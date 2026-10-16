# [MODULE DOC]
# HELPER: Pure-Python implementation of the Buchheim et al. (2002) linear-time
# variant of the Reingold-Tilford / Walker tidy tree layout algorithm. Used by
# sklearn.tree._export (matplotlib plotting) to compute x/y positions of tree nodes.
# Public contents: DrawTree (layout node wrapper), Tree (minimal input tree), buchheim
# (entry point) and the walk helpers. No import-time side effects besides importing
# numpy.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


class DrawTree:
    # [CLASS DOC]
    # HELPER: Mutable layout node wrapping an input Tree node. Holds the layout state
    # used by Buchheim's algorithm: x (horizontal position, -1 before layout), y (depth),
    # children (DrawTree wrappers), parent, thread (contour thread pointer), mod
    # (pending subtree shift), ancestor, change/shift (deferred shift bookkeeping),
    # number (1-based index among siblings).
    def __init__(self, tree, parent=None, depth=0, number=1):
        # [FUNCTION DOC]
        # Builds the wrapper recursively for the whole subtree of tree.
        # Inputs: tree - object with a children sequence (e.g. Tree); parent - parent
        # DrawTree or None for the root; depth - int depth (stored as y), default 0;
        # number - 1-based position among siblings, default 1.
        # Outputs: none. Side effects: sets all layout attributes, ancestor to self, and
        # recursively creates DrawTree children with depth + 1 and their sibling index.
        self.x = -1.0
        self.y = depth
        self.tree = tree
        self.children = [
            DrawTree(c, self, depth + 1, i + 1) for i, c in enumerate(tree.children)
        ]
        self.parent = parent
        self.thread = None
        self.mod = 0
        self.ancestor = self
        self.change = self.shift = 0
        self._lmost_sibling = None
        # this is the number of the node in its group of siblings 1..n
        self.number = number

    def left(self):
        # [FUNCTION DOC]
        # Returns the next node on the left contour: the thread if set, otherwise the
        # first child; returns 0 (falsy) when there are no children and no thread.
        return self.thread or (len(self.children) and self.children[0])

    def right(self):
        # [FUNCTION DOC]
        # Returns the next node on the right contour: the thread if set, otherwise the
        # last child; returns 0 (falsy) when there are no children and no thread.
        return self.thread or (len(self.children) and self.children[-1])

    def lbrother(self):
        # [FUNCTION DOC]
        # Returns the sibling immediately to the left of this node in parent.children,
        # or None if this node is the first child or has no parent.
        # Note: siblings are compared with equality (default identity).
        n = None
        if self.parent:
            for node in self.parent.children:
                if node == self:
                    return n
                else:
                    n = node
        return n

    def get_lmost_sibling(self):
        # [FUNCTION DOC]
        # Returns (and caches in _lmost_sibling) the leftmost sibling of this node, or
        # None when this node has no parent or is itself the first child.
        # Side effects: caches the result on first successful lookup.
        if not self._lmost_sibling and self.parent and self != self.parent.children[0]:
            self._lmost_sibling = self.parent.children[0]
        return self._lmost_sibling

    lmost_sibling = property(get_lmost_sibling)

    def __str__(self):
        # [FUNCTION DOC]
        # Returns a debugging string "<tree>: x=<x> mod=<mod>".
        return "%s: x=%s mod=%s" % (self.tree, self.x, self.mod)

    def __repr__(self):
        # [FUNCTION DOC]
        # Same as __str__.
        return self.__str__()

    def max_extents(self):
        # [FUNCTION DOC]
        # Recursively computes the maximum x and maximum y over this node and all its
        # descendants. Outputs: numpy array of shape (2,) holding [max_x, max_y].
        extents = [c.max_extents() for c in self.children]
        extents.append((self.x, self.y))
        return np.max(extents, axis=0)


def buchheim(tree):
    # [FUNCTION DOC]
    # Entry point: computes a tidy layout of tree. Wraps it in DrawTree, runs
    # first_walk (preliminary x and mod values), second_walk (final x via accumulated
    # mods, y = depth, returns minimum x), and if the minimum x is negative shifts the
    # whole tree right with third_walk so all x are >= 0.
    # Inputs: tree - root of a Tree-like structure (has children).
    # Outputs: the root DrawTree with x and y set on every node.
    dt = first_walk(DrawTree(tree))
    min = second_walk(dt)
    if min < 0:
        third_walk(dt, -min)
    return dt


def third_walk(tree, n):
    # [FUNCTION DOC]
    # HELPER: Recursively adds n to the x coordinate of tree and all descendants.
    # Side effects: mutates x of every node in the subtree.
    tree.x += n
    for c in tree.children:
        third_walk(c, n)


def first_walk(v, distance=1.0):
    # [FUNCTION DOC]
    # HELPER: Post-order first pass of Buchheim's algorithm. Leaves get x = left
    # brother's x + distance, or 0 if first child. Internal nodes: recursively lay out
    # each child (always with the default distance 1.0), resolve conflicts between
    # subtrees with apportion, apply deferred shifts via execute_shifts, then place the
    # node at the midpoint of its first and last child; if it has a left brother it is
    # placed at brother.x + distance and mod records the offset to the midpoint.
    # Inputs: v - DrawTree; distance - minimum horizontal separation, default 1.0.
    # Outputs: v. Side effects: sets x, mod, thread, ancestor, change, shift on nodes.
    # KNOWN ISSUE: the recursive call does not forward distance, so a non-default
    # distance only applies at the top level.
    if len(v.children) == 0:
        if v.lmost_sibling:
            v.x = v.lbrother().x + distance
        else:
            v.x = 0.0
    else:
        default_ancestor = v.children[0]
        for w in v.children:
            first_walk(w)
            default_ancestor = apportion(w, default_ancestor, distance)
        # print("finished v =", v.tree, "children")
        execute_shifts(v)

        midpoint = (v.children[0].x + v.children[-1].x) / 2

        w = v.lbrother()
        if w:
            v.x = w.x + distance
            v.mod = v.x - midpoint
        else:
            v.x = midpoint
    return v


def apportion(v, default_ancestor, distance):
    # [FUNCTION DOC]
    # HELPER: Resolves overlap between the subtree rooted at v and the subtrees of its
    # left siblings by walking the inner and outer contours (vil/vir/vol/vor with
    # accumulated mod sums) level by level; whenever the left contour plus distance
    # exceeds the right contour, moves the subtree of v right by the difference with
    # move_subtree. Afterwards sets threads so the contours of the combined forest
    # remain traversable, and updates the default ancestor.
    # Inputs: v - DrawTree being placed; default_ancestor - DrawTree; distance - float.
    # Outputs: the (possibly updated) default ancestor.
    # Side effects: mutates x, mod, thread, ancestor, change, shift of contour nodes.
    w = v.lbrother()
    if w is not None:
        # in buchheim notation:
        # i == inner; o == outer; r == right; l == left; r = +; l = -
        vir = vor = v
        vil = w
        vol = v.lmost_sibling
        sir = sor = v.mod
        sil = vil.mod
        sol = vol.mod
        while vil.right() and vir.left():
            vil = vil.right()
            vir = vir.left()
            vol = vol.left()
            vor = vor.right()
            vor.ancestor = v
            shift = (vil.x + sil) - (vir.x + sir) + distance
            if shift > 0:
                move_subtree(ancestor(vil, v, default_ancestor), v, shift)
                sir = sir + shift
                sor = sor + shift
            sil += vil.mod
            sir += vir.mod
            sol += vol.mod
            sor += vor.mod
        if vil.right() and not vor.right():
            vor.thread = vil.right()
            vor.mod += sil - sor
        else:
            if vir.left() and not vol.left():
                vol.thread = vir.left()
                vol.mod += sir - sol
            default_ancestor = v
    return default_ancestor


def move_subtree(wl, wr, shift):
    # [FUNCTION DOC]
    # HELPER: Shifts subtree wr right by shift and records the shift so that it is
    # distributed evenly over the intermediate siblings between wl and wr (via change
    # and shift counters, applied later by execute_shifts).
    # Inputs: wl, wr - sibling DrawTrees (wl left of wr); shift - positive float.
    # Exceptions: ZeroDivisionError if wl and wr have the same number.
    # Side effects: mutates wr.change, wr.shift, wr.x, wr.mod and wl.change.
    subtrees = wr.number - wl.number
    # print(wl.tree, "is conflicted with", wr.tree, 'moving', subtrees,
    # 'shift', shift)
    # print wl, wr, wr.number, wl.number, shift, subtrees, shift/subtrees
    wr.change -= shift / subtrees
    wr.shift += shift
    wl.change += shift / subtrees
    wr.x += shift
    wr.mod += shift


def execute_shifts(v):
    # [FUNCTION DOC]
    # HELPER: Applies the deferred shifts of v's children: iterating children from
    # right to left, adds the running shift to each child's x and mod, then accumulates
    # child.change into change and child.shift + change into shift.
    # Side effects: mutates x and mod of v's children.
    shift = change = 0
    for w in v.children[::-1]:
        # print("shift:", w, shift, w.change)
        w.x += shift
        w.mod += shift
        change += w.change
        shift += w.shift + change


def ancestor(vil, v, default_ancestor):
    # [FUNCTION DOC]
    # HELPER: Returns vil.ancestor if it is a sibling of v (one of v.parent.children),
    # otherwise default_ancestor. Used to find which left subtree to move.
    # the relevant text is at the bottom of page 7 of
    # "Improving Walker's Algorithm to Run in Linear Time" by Buchheim et al,
    # (2002)
    # https://citeseerx.ist.psu.edu/doc_view/pid/1f41c3c2a4880dc49238e46d555f16d28da2940d
    if vil.ancestor in v.parent.children:
        return vil.ancestor
    else:
        return default_ancestor


def second_walk(v, m=0, depth=0, min=None):
    # [FUNCTION DOC]
    # HELPER: Pre-order second pass: final x = preliminary x + sum of ancestors' mods
    # (m), sets y to depth and tracks the minimum x seen.
    # Inputs: v - DrawTree; m - accumulated mod of ancestors, default 0; depth - int,
    # default 0; min - running minimum or None.
    # Outputs: minimum x over the subtree (combined with incoming min).
    # Side effects: mutates x and y of every node in the subtree.
    v.x += m
    v.y = depth

    if min is None or v.x < min:
        min = v.x

    for w in v.children:
        min = second_walk(w, m + v.mod, depth + 1, min)

    return min


class Tree:
    # [CLASS DOC]
    # HELPER: Minimal input tree node for buchheim with label, node_id and children.
    def __init__(self, label="", node_id=-1, *children):
        # [FUNCTION DOC]
        # Inputs: label - display string, default ""; node_id - int id, default -1;
        # *children - child Tree nodes. Side effects: stores children as the given tuple,
        # or an empty list when no children are passed.
        self.label = label
        self.node_id = node_id
        if children:
            self.children = children
        else:
            self.children = []
