# [MODULE DOC]
# Export utilities for fitted decision trees: export_graphviz (GraphViz DOT text),
# plot_tree (matplotlib rendering laid out with the Reingold-Tilford/Buchheim
# algorithm) and export_text (ASCII rule report). Internally a shared
# _BaseTreeExporter builds node label strings and fill colors, with _DOTTreeExporter
# and _MPLTreeExporter subclasses for the two graphical back ends. Import-time side
# effects: creates the module-level SENTINEL object; matplotlib is only imported
# lazily when plotting.
"""
This module defines export functions for decision trees.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from io import StringIO
from numbers import Integral

import numpy as np

from sklearn.base import is_classifier
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    _criterion,
    _tree,
)
from sklearn.tree._reingold_tilford import Tree, buchheim
from sklearn.utils._param_validation import (
    HasMethods,
    Interval,
    StrOptions,
    validate_params,
)
from sklearn.utils.validation import check_array, check_is_fitted


def _rgb_to_hexstring(rgb):
    # [FUNCTION DOC]
    # HELPER: Formats an (r, g, b) sequence of 0-255 integers as a "#rrggbb" lowercase
    # hex string with zero-padded two-digit channels.
    """Convert 8bit integer rgb color to html hexstring"""
    return "#%02x%02x%02x" % tuple(rgb)


def _color_brew(n):
    # [FUNCTION DOC]
    # HELPER: Generates a palette of n RGB colors with hues equally spaced starting at
    # 25 degrees, saturation 0.75 and value 0.9, converted from HSV to RGB with the
    # standard chroma / sextant formula and scaled to integers in 0-255.
    # Inputs: n - positive int. Outputs: list of [r, g, b] int lists.
    # Exceptions: ZeroDivisionError if n is 0.
    # KNOWN ISSUE: hues come from a floating-point arange that can include the stop
    # value for some n, so the list may contain n + 1 colors (harmless since callers
    # only index it).
    """Generate n colors with equally spaced hues.

    Parameters
    ----------
    n : int
        The number of colors required.

    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360.0 / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [
            (c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0),
        ]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list


class Sentinel:
    # [CLASS DOC]
    # HELPER: Placeholder type whose repr is the string "tree.dot" (with quotes); its
    # single instance SENTINEL is the default out_file of _DOTTreeExporter so the
    # legacy default shows nicely in signatures.
    def __repr__(self):
        # [FUNCTION DOC]
        # Returns the literal text "tree.dot" including double quotes.
        return '"tree.dot"'


SENTINEL = Sentinel()


@validate_params(
    {
        "decision_tree": [DecisionTreeClassifier, DecisionTreeRegressor],
        "max_depth": [Interval(Integral, 0, None, closed="left"), None],
        "feature_names": ["array-like", None],
        "class_names": ["array-like", "boolean", None],
        "label": [StrOptions({"all", "root", "none"})],
        "filled": ["boolean"],
        "impurity": ["boolean"],
        "node_ids": ["boolean"],
        "proportion": ["boolean"],
        "rounded": ["boolean"],
        "precision": [Interval(Integral, 0, None, closed="left"), None],
        "ax": "no_validation",  # delegate validation to matplotlib
        "fontsize": [Interval(Integral, 0, None, closed="left"), None],
    },
    prefer_skip_nested_validation=True,
)
def plot_tree(
    decision_tree,
    *,
    max_depth=None,
    feature_names=None,
    class_names=None,
    label="all",
    filled=False,
    impurity=True,
    node_ids=False,
    proportion=False,
    rounded=False,
    precision=3,
    ax=None,
    fontsize=None,
):
    # [FUNCTION DOC]
    # Public API: validates the fitted state, builds an _MPLTreeExporter with all
    # display options and returns the result of its export on the given axes.
    # Inputs: see docstring; parameter types are enforced by validate_params.
    # Outputs: list of matplotlib Annotation artists.
    # Exceptions: NotFittedError if the tree is not fitted; InvalidParameterError for
    # invalid arguments. Side effects: clears and draws on ax (or the current axes).
    # KNOWN ISSUE: the docstring says rounded switches fonts to Helvetica; for the
    # matplotlib exporter it only rounds the box corners.
    """Plot a decision tree.

    The sample counts that are shown are weighted with any sample_weights that
    might be present.

    The visualization is fit automatically to the size of the axis.
    Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control
    the size of the rendering.

    Read more in the :ref:`User Guide <tree>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    decision_tree : decision tree regressor or classifier
        The decision tree to be plotted.

    max_depth : int, default=None
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : array-like of str, default=None
        Names of each of the features.
        If None, generic names will be used ("x[0]", "x[1]", ...).

    class_names : array-like of str or True, default=None
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.
        If ``True``, shows a symbolic representation of the class name.

    label : {'all', 'root', 'none'}, default='all'
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

    filled : bool, default=False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    impurity : bool, default=True
        When set to ``True``, show the impurity at each node.

    node_ids : bool, default=False
        When set to ``True``, show the ID number on each node.

    proportion : bool, default=False
        When set to ``True``, change the display of 'values' and/or 'samples'
        to be proportions and percentages respectively.

    rounded : bool, default=False
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    precision : int, default=3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.

    ax : matplotlib axis, default=None
        Axes to plot to. If None, use current axis. Any previous content
        is cleared.

    fontsize : int, default=None
        Size of text font. If None, determined automatically to fit figure.

    Returns
    -------
    annotations : list of artists
        List containing the artists for the annotation boxes making up the
        tree.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree

    >>> clf = tree.DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()

    >>> clf = clf.fit(iris.data, iris.target)
    >>> tree.plot_tree(clf)
    [...]
    """

    check_is_fitted(decision_tree)

    exporter = _MPLTreeExporter(
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=class_names,
        label=label,
        filled=filled,
        impurity=impurity,
        node_ids=node_ids,
        proportion=proportion,
        rounded=rounded,
        precision=precision,
        fontsize=fontsize,
    )
    return exporter.export(decision_tree, ax=ax)


class _BaseTreeExporter:
    # [CLASS DOC]
    # HELPER: Shared base of the DOT and matplotlib exporters. Stores display options
    # and provides node label formatting (node_to_str) and fill color computation.
    # Subclasses must define characters (a list of 7 markup strings: id prefix, subscript
    # open/close, less-or-equal sign, line break, label end, label start) and colors
    # (dict with a "bounds" entry).
    def __init__(
        self,
        max_depth=None,
        feature_names=None,
        class_names=None,
        label="all",
        filled=False,
        impurity=True,
        node_ids=False,
        proportion=False,
        rounded=False,
        precision=3,
        fontsize=None,
    ):
        # [FUNCTION DOC]
        # Stores all display options unchanged as attributes (no validation).
        # Inputs: same meanings as in plot_tree/export_graphviz.
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.class_names = class_names
        self.label = label
        self.filled = filled
        self.impurity = impurity
        self.node_ids = node_ids
        self.proportion = proportion
        self.rounded = rounded
        self.precision = precision
        self.fontsize = fontsize

    def get_color(self, value):
        # [FUNCTION DOC]
        # HELPER: Converts a node value into a fill color blended against white.
        # Classification (colors["bounds"] is None): picks the palette color of the argmax
        # class and uses as opacity the gap between the largest and second largest value
        # relative to one minus the second largest (0 when only one value). Regression or
        # multi-output: uses the first palette color with opacity equal to the value
        # linearly rescaled between the bounds. Each channel becomes alpha times the channel
        # plus (1 - alpha) times 255, rounded.
        # Inputs: value - class proportion array or scalar. Outputs: "#rrggbb" string.
        # Exceptions: ZeroDivisionError/NaN if the top two class proportions are both 1 or
        # the bounds are equal.
        # Find the appropriate color & intensity for a node
        if self.colors["bounds"] is None:
            # Classification tree
            color = list(self.colors["rgb"][np.argmax(value)])
            sorted_values = sorted(value, reverse=True)
            if len(sorted_values) == 1:
                alpha = 0.0
            else:
                alpha = (sorted_values[0] - sorted_values[1]) / (1 - sorted_values[1])
        else:
            # Regression tree or multi-output
            color = list(self.colors["rgb"][0])
            alpha = (value - self.colors["bounds"][0]) / (
                self.colors["bounds"][1] - self.colors["bounds"][0]
            )
        # compute the color as alpha against white
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return _rgb_to_hexstring(color)

    def get_fill_color(self, tree, node_id):
        # [FUNCTION DOC]
        # HELPER: Returns the fill color of node node_id. On first call lazily initialises
        # colors["rgb"] with a palette of n_classes[0] colors and, for multi-output trees,
        # bounds from the negated max/min impurity; for regression with non-constant values,
        # bounds from the min/max of all node values. Single output: uses the node's value
        # row (unwrapped to a float for regression with bounds); multi-output: uses the
        # negated node impurity.
        # Inputs: tree - sklearn _tree.Tree; node_id - int.
        # Side effects: mutates self.colors on first call.
        # Fetch appropriate color for node
        if "rgb" not in self.colors:
            # Initialize colors and bounds if required
            self.colors["rgb"] = _color_brew(tree.n_classes[0])
            if tree.n_outputs != 1:
                # Find max and min impurities for multi-output
                # The next line uses -max(impurity) instead of min(-impurity)
                # and -min(impurity) instead of max(-impurity) on purpose, in
                # order to avoid what looks like an issue with SIMD on non
                # memory aligned arrays on 32bit OS. For more details see
                # https://github.com/scikit-learn/scikit-learn/issues/27506.
                self.colors["bounds"] = (-np.max(tree.impurity), -np.min(tree.impurity))
            elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                # Find max and min values in leaf nodes for regression
                self.colors["bounds"] = (np.min(tree.value), np.max(tree.value))
        if tree.n_outputs == 1:
            node_val = tree.value[node_id][0, :]
            if (
                tree.n_classes[0] == 1
                and isinstance(node_val, Iterable)
                and self.colors["bounds"] is not None
            ):
                # Unpack the float only for the regression tree case.
                # Classification tree requires an Iterable in `get_color`.
                node_val = node_val.item()
        else:
            # If multi-output color node by impurity
            node_val = -tree.impurity[node_id]
        return self.get_color(node_val)

    def node_to_str(self, tree, node_id, criterion):
        # [FUNCTION DOC]
        # HELPER: Builds the label of one node using the markup in self.characters:
        # optional "node #id", the split test "feature <= threshold" (rounded to precision)
        # for non-leaves, optional impurity labelled by criterion name (MSE criterion
        # objects map to squared_error, other non-string criteria to "impurity"),
        # samples (count or percentage of root), value (class counts weighted by
        # weighted_n_node_samples unless proportion; ints when all whole numbers, otherwise
        # rounded; brackets removed for single-output regression; multi-output rows joined
        # with line breaks), and for single-output classification with class_names set the
        # majority class name (or y[i] when class_names is True). Text labels ("name = ")
        # are included at all nodes, only the root, or none according to label. A trailing
        # line break is stripped and the closing character appended.
        # Inputs: tree - _tree.Tree; node_id - int; criterion - str or Criterion object.
        # Outputs: label string. Exceptions: IndexError if class_names is too short.
        # Generate the node content string
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]

        # Should labels be shown?
        labels = (self.label == "root" and node_id == 0) or self.label == "all"

        characters = self.characters
        node_string = characters[-1]

        # Write node ID
        if self.node_ids:
            if labels:
                node_string += "node "
            node_string += characters[0] + str(node_id) + characters[4]

        # Write decision criteria
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if self.feature_names is not None:
                feature = self.feature_names[tree.feature[node_id]]
                feature = self.str_escape(feature)
            else:
                feature = "x%s%s%s" % (
                    characters[1],
                    tree.feature[node_id],
                    characters[2],
                )
            node_string += "%s %s %s%s" % (
                feature,
                characters[3],
                round(tree.threshold[node_id], self.precision),
                characters[4],
            )

        # Write impurity
        if self.impurity:
            if isinstance(criterion, _criterion.MSE) or criterion == "squared_error":
                criterion = "squared_error"
            elif not isinstance(criterion, str):
                criterion = "impurity"
            if labels:
                node_string += "%s = " % criterion
            node_string += (
                str(round(tree.impurity[node_id], self.precision)) + characters[4]
            )

        # Write node sample count
        if labels:
            node_string += "samples = "
        if self.proportion:
            percent = (
                100.0 * tree.n_node_samples[node_id] / float(tree.n_node_samples[0])
            )
            node_string += str(round(percent, 1)) + "%" + characters[4]
        else:
            node_string += str(tree.n_node_samples[node_id]) + characters[4]

        # Write node class distribution / regression value
        if not self.proportion and tree.n_classes[0] != 1:
            # For classification this will show the proportion of samples
            value = value * tree.weighted_n_node_samples[node_id]
        if labels:
            node_string += "value = "
        if tree.n_classes[0] == 1:
            # Regression
            value_text = np.around(value, self.precision)
        elif self.proportion:
            # Classification
            value_text = np.around(value, self.precision)
        elif np.all(np.equal(np.mod(value, 1), 0)):
            # Classification without floating-point weights
            value_text = value.astype(int)
        else:
            # Classification with floating-point weights
            value_text = np.around(value, self.precision)
        # Strip whitespace
        value_text = str(value_text.astype("S32")).replace("b'", "'")
        value_text = value_text.replace("' '", ", ").replace("'", "")
        if tree.n_classes[0] == 1 and tree.n_outputs == 1:
            value_text = value_text.replace("[", "").replace("]", "")
        value_text = value_text.replace("\n ", characters[4])
        node_string += value_text + characters[4]

        # Write node majority class
        if (
            self.class_names is not None
            and tree.n_classes[0] != 1
            and tree.n_outputs == 1
        ):
            # Only done for single-output classification trees
            if labels:
                node_string += "class = "
            if self.class_names is not True:
                class_name = self.class_names[np.argmax(value)]
                class_name = self.str_escape(class_name)
            else:
                class_name = "y%s%s%s" % (
                    characters[1],
                    np.argmax(value),
                    characters[2],
                )
            node_string += class_name

        # Clean up any trailing newlines
        if node_string.endswith(characters[4]):
            node_string = node_string[: -len(characters[4])]

        return node_string + characters[5]

    def str_escape(self, string):
        # [FUNCTION DOC]
        # HELPER: Hook for escaping names in labels; identity in the base class.
        return string


class _DOTTreeExporter(_BaseTreeExporter):
    # [CLASS DOC]
    # HELPER: Writes a tree as GraphViz DOT text to a file-like out_file. Sets
    # characters to HTML-like markup when special_characters is True, otherwise plain
    # DOT escapes; keeps ranks (node ids per depth plus leaves) for leaves_parallel and
    # the colors cache.
    def __init__(
        self,
        out_file=SENTINEL,
        max_depth=None,
        feature_names=None,
        class_names=None,
        label="all",
        filled=False,
        leaves_parallel=False,
        impurity=True,
        node_ids=False,
        proportion=False,
        rotate=False,
        rounded=False,
        special_characters=False,
        precision=3,
        fontname="helvetica",
    ):
        # [FUNCTION DOC]
        # Stores options (via the base class plus leaves_parallel, out_file,
        # special_characters, fontname, rotate) and initialises characters, ranks and colors.
        # Inputs: out_file - writable text object (default SENTINEL, which is not writable);
        # other options as in export_graphviz.
        super().__init__(
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rounded=rounded,
            precision=precision,
        )
        self.leaves_parallel = leaves_parallel
        self.out_file = out_file
        self.special_characters = special_characters
        self.fontname = fontname
        self.rotate = rotate

        # PostScript compatibility for special characters
        if special_characters:
            self.characters = ["&#35;", "<SUB>", "</SUB>", "&le;", "<br/>", ">", "<"]
        else:
            self.characters = ["#", "[", "]", "<=", "\\n", '"', '"']

        # The depth of each node for plotting with 'leaf' option
        self.ranks = {"leaves": []}
        # The colors to render each node with
        self.colors = {"bounds": None}

    def export(self, decision_tree):
        # [FUNCTION DOC]
        # HELPER: Writes the full DOT document: header, all nodes and edges by recursion
        # from the root, then the closing tail.
        # Inputs: decision_tree - fitted tree estimator or raw _tree.Tree (then the
        # criterion is shown as "impurity").
        # Exceptions: ValueError if feature_names length differs from n_features_in_.
        # Side effects: writes to out_file and fills ranks/colors.
        # KNOWN ISSUE: the feature_names check reads n_features_in_, which a raw _tree.Tree
        # lacks, so the _tree.Tree branch fails when feature_names is given; export_graphviz
        # also rejects non-estimators earlier, making that branch effectively dead.
        # Check length of feature_names before getting into the tree node
        # Raise error if length of feature_names does not match
        # n_features_in_ in the decision_tree
        if self.feature_names is not None:
            if len(self.feature_names) != decision_tree.n_features_in_:
                raise ValueError(
                    "Length of feature_names, %d does not match number of features, %d"
                    % (len(self.feature_names), decision_tree.n_features_in_)
                )
        # each part writes to out_file
        self.head()
        # Now recurse the tree and add node & edge attributes
        if isinstance(decision_tree, _tree.Tree):
            self.recurse(decision_tree, 0, criterion="impurity")
        else:
            self.recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

        self.tail()

    def tail(self):
        # [FUNCTION DOC]
        # HELPER: If leaves_parallel, writes one "{rank=same ; ...}" line per rank key in
        # sorted (string) order; then writes the closing brace.
        # If required, draw leaf nodes at same depth as each other
        if self.leaves_parallel:
            for rank in sorted(self.ranks):
                self.out_file.write(
                    "{rank=same ; " + "; ".join(r for r in self.ranks[rank]) + "} ;\n"
                )
        self.out_file.write("}")

    def head(self):
        # [FUNCTION DOC]
        # HELPER: Writes the DOT preamble: digraph opening, node style (box, optional
        # filled/rounded style with black border, fontname), optional graph ranksep/spline
        # line for leaves_parallel, edge font, and rankdir=LR when rotate is set.
        self.out_file.write("digraph Tree {\n")

        # Specify node aesthetics
        self.out_file.write("node [shape=box")
        rounded_filled = []
        if self.filled:
            rounded_filled.append("filled")
        if self.rounded:
            rounded_filled.append("rounded")
        if len(rounded_filled) > 0:
            self.out_file.write(
                ', style="%s", color="black"' % ", ".join(rounded_filled)
            )

        self.out_file.write(', fontname="%s"' % self.fontname)
        self.out_file.write("] ;\n")

        # Specify graph & edge aesthetics
        if self.leaves_parallel:
            self.out_file.write("graph [ranksep=equally, splines=polyline] ;\n")

        self.out_file.write('edge [fontname="%s"] ;\n' % self.fontname)

        if self.rotate:
            self.out_file.write("rankdir=LR ;\n")

    def recurse(self, tree, node_id, criterion, parent=None, depth=0):
        # [FUNCTION DOC]
        # HELPER: Depth-first pre-order writer. For nodes within max_depth: records the node
        # id in ranks (leaves or its depth), writes the node with its label and optional fill
        # color, writes the edge from parent (with True/False head labels and angles flipped
        # when rotate for root children), then recurses into both children. Nodes beyond
        # max_depth are written as grey-able "(...)" placeholders with an edge from the
        # parent and are not expanded.
        # Inputs: tree - _tree.Tree; node_id - int; criterion; parent - parent id or None;
        # depth - int, default 0.
        # Exceptions: ValueError if node_id equals TREE_LEAF.
        # Side effects: writes to out_file, mutates ranks and colors.
        # KNOWN ISSUE: the truncated-node branch formats a string with no placeholders
        # using node_id; this only works because node_id is a numpy integer there (a plain
        # Python int would raise TypeError).
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # Add node with description
        if self.max_depth is None or depth <= self.max_depth:
            # Collect ranks for 'leaf' option in plot_options
            if left_child == _tree.TREE_LEAF:
                self.ranks["leaves"].append(str(node_id))
            elif str(depth) not in self.ranks:
                self.ranks[str(depth)] = [str(node_id)]
            else:
                self.ranks[str(depth)].append(str(node_id))

            self.out_file.write(
                "%d [label=%s" % (node_id, self.node_to_str(tree, node_id, criterion))
            )

            if self.filled:
                self.out_file.write(
                    ', fillcolor="%s"' % self.get_fill_color(tree, node_id)
                )
            self.out_file.write("] ;\n")

            if parent is not None:
                # Add edge to parent
                self.out_file.write("%d -> %d" % (parent, node_id))
                if parent == 0:
                    # Draw True/False labels if parent is root node
                    angles = np.array([45, -45]) * ((self.rotate - 0.5) * -2)
                    self.out_file.write(" [labeldistance=2.5, labelangle=")
                    if node_id == 1:
                        self.out_file.write('%d, headlabel="True"]' % angles[0])
                    else:
                        self.out_file.write('%d, headlabel="False"]' % angles[1])
                self.out_file.write(" ;\n")

            if left_child != _tree.TREE_LEAF:
                self.recurse(
                    tree,
                    left_child,
                    criterion=criterion,
                    parent=node_id,
                    depth=depth + 1,
                )
                self.recurse(
                    tree,
                    right_child,
                    criterion=criterion,
                    parent=node_id,
                    depth=depth + 1,
                )

        else:
            self.ranks["leaves"].append(str(node_id))

            self.out_file.write('%d [label="(...)"' % node_id)
            if self.filled:
                # color cropped nodes grey
                self.out_file.write(', fillcolor="#C0C0C0"')
            self.out_file.write("] ;\n" % node_id)

            if parent is not None:
                # Add edge to parent
                self.out_file.write("%d -> %d ;\n" % (parent, node_id))

    def str_escape(self, string):
        # [FUNCTION DOC]
        # HELPER: Escapes double quotes with a backslash for DOT labels.
        # override default escaping for graphviz
        return string.replace('"', r"\"")


class _MPLTreeExporter(_BaseTreeExporter):
    # [CLASS DOC]
    # HELPER: Renders a tree on matplotlib axes: converts the tree to a
    # _reingold_tilford.Tree of labels, lays it out with buchheim and draws one
    # annotation per node with arrows to the parent.
    def __init__(
        self,
        max_depth=None,
        feature_names=None,
        class_names=None,
        label="all",
        filled=False,
        impurity=True,
        node_ids=False,
        proportion=False,
        rounded=False,
        precision=3,
        fontsize=None,
    ):
        # [FUNCTION DOC]
        # Stores options, sets plain-text characters (newline as line break, no markup),
        # bbox_args (round box style if rounded), and arrow_args with a "<-" arrow style.
        super().__init__(
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rounded=rounded,
            precision=precision,
        )
        self.fontsize = fontsize

        # The depth of each node for plotting with 'leaf' option
        self.ranks = {"leaves": []}
        # The colors to render each node with
        self.colors = {"bounds": None}

        self.characters = ["#", "[", "]", "<=", "\n", "", ""]
        self.bbox_args = dict()
        if self.rounded:
            self.bbox_args["boxstyle"] = "round"

        self.arrow_args = dict(arrowstyle="<-")

    def _make_tree(self, node_id, et, criterion, depth=0):
        # [FUNCTION DOC]
        # HELPER: Recursively converts _tree.Tree node node_id (et) into a
        # _reingold_tilford.Tree whose label is node_to_str; children are included only for
        # split nodes with depth <= max_depth, so nodes at depth max_depth + 1 become leaves.
        # Outputs: _reingold_tilford.Tree.
        # traverses _tree.Tree recursively, builds intermediate
        # "_reingold_tilford.Tree" object
        name = self.node_to_str(et, node_id, criterion=criterion)
        if et.children_left[node_id] != _tree.TREE_LEAF and (
            self.max_depth is None or depth <= self.max_depth
        ):
            children = [
                self._make_tree(
                    et.children_left[node_id], et, criterion, depth=depth + 1
                ),
                self._make_tree(
                    et.children_right[node_id], et, criterion, depth=depth + 1
                ),
            ]
        else:
            return Tree(name, node_id)
        return Tree(name, node_id, *children)

    def export(self, decision_tree, ax=None):
        # [FUNCTION DOC]
        # HELPER: Clears ax (current axes if None) and hides its axis, builds and lays out
        # the tree, draws it via recurse in axes-fraction coordinates scaled by the layout
        # extents plus 1, then updates annotation bounding boxes with the canvas renderer.
        # When fontsize is None, rescales all fonts so the largest box fits the per-node
        # spacing (minimum of horizontal and vertical ratios).
        # Inputs: decision_tree - fitted estimator; ax - Axes or None.
        # Outputs: list of Annotation artists on ax.
        # Side effects: imports matplotlib, mutates the axes and font sizes.
        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation

        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        my_tree = self._make_tree(0, decision_tree.tree_, decision_tree.criterion)
        draw_tree = buchheim(my_tree)

        # important to make sure we're still
        # inside the axis after drawing the box
        # this makes sense because the width of a box
        # is about the same as the distance between boxes
        max_x, max_y = draw_tree.max_extents() + 1
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height

        scale_x = ax_width / max_x
        scale_y = ax_height / max_y
        self.recurse(draw_tree, decision_tree.tree_, ax, max_x, max_y)

        anns = [ann for ann in ax.get_children() if isinstance(ann, Annotation)]

        # update sizes of all bboxes
        renderer = ax.figure.canvas.get_renderer()

        for ann in anns:
            ann.update_bbox_position_size(renderer)

        if self.fontsize is None:
            # get figure to data transform
            # adjust fontsize to avoid overlap
            # get max box width and height
            extents = [
                bbox_patch.get_window_extent()
                for ann in anns
                if (bbox_patch := ann.get_bbox_patch()) is not None
            ]
            max_width = max([extent.width for extent in extents])
            max_height = max([extent.height for extent in extents])
            # width should be around scale_x in axis coordinates
            size = anns[0].get_fontsize() * min(
                scale_x / max_width, scale_y / max_height
            )
            for ann in anns:
                ann.set_fontsize(size)

        return anns

    def recurse(self, node, tree, ax, max_x, max_y, depth=0):
        # [FUNCTION DOC]
        # HELPER: Draws node (a DrawTree) and, recursively, its children. Position is
        # ((x + 0.5) / max_x, (max_y - y - 0.5) / max_y). Within max_depth it draws the
        # label box (filled color or axes face color) with an arrow from the parent, and for
        # the root's children a "True" (left child) or "False" edge label at the arrow
        # midpoint. Beyond max_depth it draws a grey "(...)" box and stops. Z-order decreases
        # with depth.
        # Inputs: node - DrawTree; tree - _tree.Tree; ax; max_x, max_y - layout extents;
        # depth - int. Side effects: adds annotations to ax.
        import matplotlib.pyplot as plt

        # kwargs for annotations without a bounding box
        common_kwargs = dict(
            zorder=100 - 10 * depth,
            xycoords="axes fraction",
        )
        if self.fontsize is not None:
            common_kwargs["fontsize"] = self.fontsize

        # kwargs for annotations with a bounding box
        kwargs = dict(
            ha="center",
            va="center",
            bbox=self.bbox_args.copy(),
            arrowprops=self.arrow_args.copy(),
            **common_kwargs,
        )
        kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]

        # offset things by .5 to center them in plot
        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)

        if self.max_depth is None or depth <= self.max_depth:
            if self.filled:
                kwargs["bbox"]["fc"] = self.get_fill_color(tree, node.tree.node_id)
            else:
                kwargs["bbox"]["fc"] = ax.get_facecolor()

            if node.parent is None:
                # root
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = (
                    (node.parent.x + 0.5) / max_x,
                    (max_y - node.parent.y - 0.5) / max_y,
                )
                ax.annotate(node.tree.label, xy_parent, xy, **kwargs)

                # Draw True/False labels if parent is root node
                if node.parent.parent is None:
                    # Adjust the position for the text to be slightly above the arrow
                    text_pos = (
                        (xy_parent[0] + xy[0]) / 2,
                        (xy_parent[1] + xy[1]) / 2,
                    )
                    # Annotate the arrow with the edge label to indicate the child
                    # where the sample-split condition is satisfied
                    if node.parent.left() == node:
                        label_text, label_ha = ("True  ", "right")
                    else:
                        label_text, label_ha = ("  False", "left")
                    ax.annotate(label_text, text_pos, ha=label_ha, **common_kwargs)
            for child in node.children:
                self.recurse(child, tree, ax, max_x, max_y, depth=depth + 1)

        else:
            xy_parent = (
                (node.parent.x + 0.5) / max_x,
                (max_y - node.parent.y - 0.5) / max_y,
            )
            kwargs["bbox"]["fc"] = "grey"
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)


@validate_params(
    {
        "decision_tree": "no_validation",
        "out_file": [str, None, HasMethods("write")],
        "max_depth": [Interval(Integral, 0, None, closed="left"), None],
        "feature_names": ["array-like", None],
        "class_names": ["array-like", "boolean", None],
        "label": [StrOptions({"all", "root", "none"})],
        "filled": ["boolean"],
        "leaves_parallel": ["boolean"],
        "impurity": ["boolean"],
        "node_ids": ["boolean"],
        "proportion": ["boolean"],
        "rotate": ["boolean"],
        "rounded": ["boolean"],
        "special_characters": ["boolean"],
        "precision": [Interval(Integral, 0, None, closed="left"), None],
        "fontname": [str],
    },
    prefer_skip_nested_validation=True,
)
def export_graphviz(
    decision_tree,
    out_file=None,
    *,
    max_depth=None,
    feature_names=None,
    class_names=None,
    label="all",
    filled=False,
    leaves_parallel=False,
    impurity=True,
    node_ids=False,
    proportion=False,
    rotate=False,
    rounded=False,
    special_characters=False,
    precision=3,
    fontname="helvetica",
):
    # [FUNCTION DOC]
    # Public API: optionally validates feature_names (all strings, converted to an
    # array) and class_names (array unless a bool), checks the tree is fitted, opens
    # out_file when given a path (utf-8, closed in a finally block) or uses a StringIO
    # when None, and runs a _DOTTreeExporter.
    # Outputs: DOT string when out_file is None, otherwise None.
    # Exceptions: ValueError for non-string feature names or wrong feature_names length;
    # NotFittedError / TypeError from check_is_fitted for unfitted or non-estimator
    # input; InvalidParameterError for invalid options.
    # Side effects: writes to out_file or a created file.
    """Export a decision tree in DOT format.

    This function generates a GraphViz representation of the decision tree,
    which is then written into `out_file`. Once exported, graphical renderings
    can be generated using, for example::

        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)

    The sample counts that are shown are weighted with any sample_weights that
    might be present.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    decision_tree : object
        The decision tree estimator to be exported to GraphViz.

    out_file : object or str, default=None
        Handle or name of the output file. If ``None``, the result is
        returned as a string.

        .. versionchanged:: 0.20
            Default of out_file changed from "tree.dot" to None.

    max_depth : int, default=None
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : array-like of shape (n_features,), default=None
        An array containing the feature names.
        If None, generic names will be used ("x[0]", "x[1]", ...).

    class_names : array-like of shape (n_classes,) or bool, default=None
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.
        If ``True``, shows a symbolic representation of the class name.

    label : {'all', 'root', 'none'}, default='all'
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

    filled : bool, default=False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    leaves_parallel : bool, default=False
        When set to ``True``, draw all leaf nodes at the bottom of the tree.

    impurity : bool, default=True
        When set to ``True``, show the impurity at each node.

    node_ids : bool, default=False
        When set to ``True``, show the ID number on each node.

    proportion : bool, default=False
        When set to ``True``, change the display of 'values' and/or 'samples'
        to be proportions and percentages respectively.

    rotate : bool, default=False
        When set to ``True``, orient tree left to right rather than top-down.

    rounded : bool, default=False
        When set to ``True``, draw node boxes with rounded corners.

    special_characters : bool, default=False
        When set to ``False``, ignore special characters for PostScript
        compatibility.

    precision : int, default=3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.

    fontname : str, default='helvetica'
        Name of font used to render text.

    Returns
    -------
    dot_data : str
        String representation of the input tree in GraphViz dot format.
        Only returned if ``out_file`` is None.

        .. versionadded:: 0.18

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree

    >>> clf = tree.DecisionTreeClassifier()
    >>> iris = load_iris()

    >>> clf = clf.fit(iris.data, iris.target)
    >>> tree.export_graphviz(clf)
    'digraph Tree {...
    """
    if feature_names is not None:
        if any((not isinstance(name, str) for name in feature_names)):
            raise ValueError("All feature names must be strings.")
        feature_names = check_array(
            feature_names, ensure_2d=False, dtype=None, ensure_min_samples=0
        )
    if class_names is not None and not isinstance(class_names, bool):
        class_names = check_array(
            class_names, ensure_2d=False, dtype=None, ensure_min_samples=0
        )

    check_is_fitted(decision_tree)
    own_file = False
    return_string = False
    try:
        if isinstance(out_file, str):
            out_file = open(out_file, "w", encoding="utf-8")
            own_file = True

        if out_file is None:
            return_string = True
            out_file = StringIO()

        exporter = _DOTTreeExporter(
            out_file=out_file,
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            leaves_parallel=leaves_parallel,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rotate=rotate,
            rounded=rounded,
            special_characters=special_characters,
            precision=precision,
            fontname=fontname,
        )
        exporter.export(decision_tree)

        if return_string:
            return exporter.out_file.getvalue()

    finally:
        if own_file:
            out_file.close()


def _compute_depth(tree, node):
    # [FUNCTION DOC]
    # HELPER: Returns the depth of the subtree rooted at node, counting node itself as
    # depth 1, by recursively visiting all descendants.
    # Inputs: tree - _tree.Tree; node - int. Outputs: int >= 1.
    """
    Returns the depth of the subtree rooted in node.
    """

    def compute_depth_(
        current_node, current_depth, children_left, children_right, depths
    ):
        # [FUNCTION DOC]
        # HELPER: Recursive visitor appending current_depth of every visited node to the
        # depths list; descends into both children when both are not -1.
        # Side effects: mutates depths.
        depths += [current_depth]
        left = children_left[current_node]
        right = children_right[current_node]
        if left != -1 and right != -1:
            compute_depth_(
                left, current_depth + 1, children_left, children_right, depths
            )
            compute_depth_(
                right, current_depth + 1, children_left, children_right, depths
            )

    depths = []
    compute_depth_(node, 1, tree.children_left, tree.children_right, depths)
    return max(depths)


@validate_params(
    {
        "decision_tree": [DecisionTreeClassifier, DecisionTreeRegressor],
        "feature_names": ["array-like", None],
        "class_names": ["array-like", None],
        "max_depth": [Interval(Integral, 0, None, closed="left"), None],
        "spacing": [Interval(Integral, 1, None, closed="left"), None],
        "decimals": [Interval(Integral, 0, None, closed="left"), None],
        "show_weights": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def export_text(
    decision_tree,
    *,
    feature_names=None,
    class_names=None,
    max_depth=10,
    spacing=3,
    decimals=2,
    show_weights=False,
):
    # [FUNCTION DOC]
    # Public API: builds an indented text report of the tree rules. Converts names to
    # arrays, checks fitting, resolves class_names (defaults to classes_), picks the
    # leaf format (weights and class for classifiers, value list for regressors) and
    # generic feature_i names when none given, then walks the tree depth-first writing
    # "name <= threshold" then the left subtree, "name >  threshold" then the right
    # subtree. Nodes deeper than max_depth + 1 are written as a leaf if their subtree is
    # a single node, otherwise as "truncated branch of depth d".
    # Outputs: report string with one line per edge or leaf.
    # Exceptions: ValueError for class_names or feature_names of wrong length;
    # NotFittedError if unfitted; InvalidParameterError for invalid options.
    # KNOWN ISSUE: right_child_fmt is used for the left (<=) branch and left_child_fmt
    # for the right (>) branch, the names are swapped. The docstring says truncated
    # branches are marked with "...", but the text "truncated branch of depth" is used.
    """Build a text report showing the rules of a decision tree.

    Note that backwards compatibility may not be supported.

    Parameters
    ----------
    decision_tree : object
        The decision tree estimator to be exported.
        It can be an instance of
        DecisionTreeClassifier or DecisionTreeRegressor.

    feature_names : array-like of shape (n_features,), default=None
        An array containing the feature names.
        If None generic names will be used ("feature_0", "feature_1", ...).

    class_names : array-like of shape (n_classes,), default=None
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.

        - if `None`, the class names are delegated to `decision_tree.classes_`;
        - otherwise, `class_names` will be used as class names instead of
          `decision_tree.classes_`. The length of `class_names` must match
          the length of `decision_tree.classes_`.

        .. versionadded:: 1.3

    max_depth : int, default=10
        Only the first max_depth levels of the tree are exported.
        Truncated branches will be marked with "...".

    spacing : int, default=3
        Number of spaces between edges. The higher it is, the wider the result.

    decimals : int, default=2
        Number of decimal digits to display.

    show_weights : bool, default=False
        If true the classification weights will be exported on each leaf.
        The classification weights are the number of samples each class.

    Returns
    -------
    report : str
        Text summary of all the rules in the decision tree.

    Examples
    --------

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.tree import export_text
    >>> iris = load_iris()
    >>> X = iris['data']
    >>> y = iris['target']
    >>> decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
    >>> decision_tree = decision_tree.fit(X, y)
    >>> r = export_text(decision_tree, feature_names=iris['feature_names'])
    >>> print(r)
    |--- petal width (cm) <= 0.80
    |   |--- class: 0
    |--- petal width (cm) >  0.80
    |   |--- petal width (cm) <= 1.75
    |   |   |--- class: 1
    |   |--- petal width (cm) >  1.75
    |   |   |--- class: 2
    """
    if feature_names is not None:
        feature_names = check_array(
            feature_names, ensure_2d=False, dtype=None, ensure_min_samples=0
        )
    if class_names is not None:
        class_names = check_array(
            class_names, ensure_2d=False, dtype=None, ensure_min_samples=0
        )

    check_is_fitted(decision_tree)
    tree_ = decision_tree.tree_
    if is_classifier(decision_tree):
        if class_names is None:
            class_names = decision_tree.classes_
        elif len(class_names) != len(decision_tree.classes_):
            raise ValueError(
                "When `class_names` is an array, it should contain as"
                " many items as `decision_tree.classes_`. Got"
                f" {len(class_names)} while the tree was fitted with"
                f" {len(decision_tree.classes_)} classes."
            )
    right_child_fmt = "{} {} <= {}\n"
    left_child_fmt = "{} {} >  {}\n"
    truncation_fmt = "{} {}\n"

    if feature_names is not None and len(feature_names) != tree_.n_features:
        raise ValueError(
            "feature_names must contain %d elements, got %d"
            % (tree_.n_features, len(feature_names))
        )

    if isinstance(decision_tree, DecisionTreeClassifier):
        value_fmt = "{}{} weights: {}\n"
        if not show_weights:
            value_fmt = "{}{}{}\n"
    else:
        value_fmt = "{}{} value: {}\n"

    if feature_names is not None:
        feature_names_ = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else None
            for i in tree_.feature
        ]
    else:
        feature_names_ = ["feature_{}".format(i) for i in tree_.feature]

    report = StringIO()

    def _add_leaf(value, weighted_n_node_samples, class_name, indent):
        # [FUNCTION DOC]
        # HELPER: Writes a leaf line. Classifiers: optional weights list (each proportion
        # times the node weighted sample count, formatted with decimals) followed by
        # " class: <name>". Regressors: list of output values formatted with decimals.
        # Side effects: writes to report.
        # KNOWN ISSUE: contains a no-op expression statement (weighted_n_node_samples on
        # its own line).
        val = ""
        if isinstance(decision_tree, DecisionTreeClassifier):
            if show_weights:
                val = [
                    "{1:.{0}f}, ".format(decimals, v * weighted_n_node_samples)
                    for v in value
                ]
                val = "[" + "".join(val)[:-2] + "]"
                weighted_n_node_samples
            val += " class: " + str(class_name)
        else:
            val = ["{1:.{0}f}, ".format(decimals, v) for v in value]
            val = "[" + "".join(val)[:-2] + "]"
        report.write(value_fmt.format(indent, "", val))

    def print_tree_recurse(report, node, depth):
        # [FUNCTION DOC]
        # HELPER: Recursive writer for node at depth (root is 1). Builds the indent from
        # "|" plus spacing spaces per level, with the last segment replaced by dashes;
        # determines the node value (first output for multi-output regressors' transposed
        # values), majority class name for single-output classifiers, then writes split or
        # leaf lines or truncation text as described in export_text.
        # Inputs: report - StringIO; node - int; depth - int. Side effects: writes to report.
        indent = ("|" + (" " * spacing)) * depth
        indent = indent[:-spacing] + "-" * spacing

        value = None
        if tree_.n_outputs == 1:
            value = tree_.value[node][0]
        else:
            value = tree_.value[node].T[0]
        class_name = np.argmax(value)

        if tree_.n_classes[0] != 1 and tree_.n_outputs == 1:
            class_name = class_names[class_name]

        weighted_n_node_samples = tree_.weighted_n_node_samples[node]

        if depth <= max_depth + 1:
            info_fmt = ""
            info_fmt_left = info_fmt
            info_fmt_right = info_fmt

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names_[node]
                threshold = tree_.threshold[node]
                threshold = "{1:.{0}f}".format(decimals, threshold)
                report.write(right_child_fmt.format(indent, name, threshold))
                report.write(info_fmt_left)
                print_tree_recurse(report, tree_.children_left[node], depth + 1)

                report.write(left_child_fmt.format(indent, name, threshold))
                report.write(info_fmt_right)
                print_tree_recurse(report, tree_.children_right[node], depth + 1)
            else:  # leaf
                _add_leaf(value, weighted_n_node_samples, class_name, indent)
        else:
            subtree_depth = _compute_depth(tree_, node)
            if subtree_depth == 1:
                _add_leaf(value, weighted_n_node_samples, class_name, indent)
            else:
                trunc_report = "truncated branch of depth %d" % subtree_depth
                report.write(truncation_fmt.format(indent, trunc_report))

    print_tree_recurse(report, 0, 1)
    return report.getvalue()
