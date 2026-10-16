# [MODULE DOC]
# Pretty-printing machinery behind BaseEstimator.__repr__. Adapts CPython's pprint
# module so estimators render as ClassName(param=value, ...), dict items honour the
# compact flag, and long sequences/param lists are truncated with an ellipsis.
# Exposes KeyValTuple, KeyValTupleParam, _changed_params, _EstimatorPrettyPrinter and
# _safe_repr. No import-time side effects beyond building the printer dispatch table.
"""This module contains the _EstimatorPrettyPrinter class used in
BaseEstimator.__repr__ for pretty-printing estimators"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
# 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018 Python Software Foundation;
# All Rights Reserved

# Authors: Fred L. Drake, Jr. <fdrake@acm.org> (built-in CPython pprint module)
#          Nicolas Hug (scikit-learn specific changes)

# License: PSF License version 2 (see below)

# PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
# --------------------------------------------

# 1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"),
# and the Individual or Organization ("Licensee") accessing and otherwise
# using this software ("Python") in source or binary form and its associated
# documentation.

# 2. Subject to the terms and conditions of this License Agreement, PSF hereby
# grants Licensee a nonexclusive, royalty-free, world-wide license to
# reproduce, analyze, test, perform and/or display publicly, prepare
# derivative works, distribute, and otherwise use Python alone or in any
# derivative version, provided, however, that PSF's License Agreement and
# PSF's notice of copyright, i.e., "Copyright (c) 2001, 2002, 2003, 2004,
# 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
# 2017, 2018 Python Software Foundation; All Rights Reserved" are retained in
# Python alone or in any derivative version prepared by Licensee.

# 3. In the event Licensee prepares a derivative work that is based on or
# incorporates Python or any part thereof, and wants to make the derivative
# work available to others as provided herein, then Licensee hereby agrees to
# include in any such work a brief summary of the changes made to Python.

# 4. PSF is making Python available to Licensee on an "AS IS" basis. PSF MAKES
# NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED. BY WAY OF EXAMPLE, BUT
# NOT LIMITATION, PSF MAKES NO AND DISCLAIMS ANY REPRESENTATION OR WARRANTY OF
# MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF
# PYTHON WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

# 5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON FOR ANY
# INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF
# MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON, OR ANY DERIVATIVE
# THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

# 6. This License Agreement will automatically terminate upon a material
# breach of its terms and conditions.

# 7. Nothing in this License Agreement shall be deemed to create any
# relationship of agency, partnership, or joint venture between PSF and
# Licensee. This License Agreement does not grant permission to use PSF
# trademarks or trade name in a trademark sense to endorse or promote products
# or services of Licensee, or any third party.

# 8. By copying, installing or otherwise using Python, Licensee agrees to be
# bound by the terms and conditions of this License Agreement.


# Brief summary of changes to original code:
# - "compact" parameter is supported for dicts, not just lists or tuples
# - estimators have a custom handler, they're not just treated as objects
# - long sequences (lists, tuples, dict items) with more than N elements are
#   shortened using ellipsis (', ...') at the end.

import inspect
import pprint

from sklearn._config import get_config
from sklearn.base import BaseEstimator
from sklearn.utils._missing import is_scalar_nan


class KeyValTuple(tuple):
    # [CLASS DOC]
    # HELPER: Marker tuple subclass holding one (key, value) pair of a dict, so the
    # printer dispatch table can route it to _pprint_key_val_tuple and render 'key':
    # value.
    """Dummy class for correctly rendering key-value tuples from dicts."""

    def __repr__(self):
        # [FUNCTION DOC]
        # HELPER: Delegates to the plain tuple repr; defining it gives the class its own
        # __repr__ object to use as a dispatch key distinct from tuple.__repr__.
        # Outputs: string repr of the tuple.
        # needed for _dispatch[tuple.__repr__] not to be overridden
        return super().__repr__()


class KeyValTupleParam(KeyValTuple):
    # [CLASS DOC]
    # HELPER: Marker subclass of KeyValTuple for estimator (param, value) pairs;
    # rendered
    # as key=value (key quotes stripped) instead of 'key': value. Inherits __repr__, so
    # it
    # dispatches to the same key/value printer, which checks the subclass at runtime.
    """Dummy class for correctly rendering key-value tuples from parameters."""

    pass


def _changed_params(estimator):
    # [FUNCTION DOC]
    # HELPER: Computes the estimator parameters whose value differs from the default in
    # the estimator's __init__ signature (used when print_changed_only is enabled).
    # Inputs: estimator - a BaseEstimator instance.
    # Outputs: dict mapping parameter name to current value for the changed parameters.
    # Exceptions: propagates errors from get_params or from repr of parameter values.
    # Side effects: none.
    """Return dict (param_name: value) of parameters that were given to
    estimator with non-default values."""

    params = estimator.get_params(deep=False)
    init_params = inspect.signature(estimator.__init__).parameters
    init_params = {name: param.default for name, param in init_params.items()}

    def has_changed(k, v):
        # [FUNCTION DOC]
        # HELPER: Decides whether parameter k with current value v counts as changed.
        # Returns True if k is not an explicit __init__ argument (e.g. collected by
        # **kwargs),
        # if it has no default, if v is an estimator whose class differs from the
        # default's
        # class, or if the reprs of v and the default differ (unless both are scalar
        # NaN).
        # Otherwise False. Inputs: k - parameter name; v - current value. Outputs: bool.
        if k not in init_params:  # happens if k is part of a **kwargs
            return True
        if init_params[k] == inspect._empty:  # k has no default value
            return True
        # try to avoid calling repr on nested estimators
        if isinstance(v, BaseEstimator) and v.__class__ != init_params[k].__class__:
            return True
        # Use repr as a last resort. It may be expensive.
        if repr(v) != repr(init_params[k]) and not (
            is_scalar_nan(init_params[k]) and is_scalar_nan(v)
        ):
            return True
        return False

    return {k: v for k, v in params.items() if has_changed(k, v)}


class _EstimatorPrettyPrinter(pprint.PrettyPrinter):
    # [CLASS DOC]
    # PrettyPrinter subclass that renders estimators with their (optionally only
    # changed) parameters, applies compact layout to dict items, and truncates
    # sequences/params beyond n_max_elements_to_show with ', ...'. Its class-level
    # _dispatch is a copy of the builtin table extended with handlers for
    # BaseEstimator.__repr__ and KeyValTuple.__repr__.
    """Pretty Printer class for estimator objects.

    This extends the pprint.PrettyPrinter class, because:
    - we need estimators to be printed with their parameters, e.g.
      Estimator(param1=value1, ...) which is not supported by default.
    - the 'compact' parameter of PrettyPrinter is ignored for dicts, which
      may lead to very long representations that we want to avoid.

    Quick overview of pprint.PrettyPrinter (see also
    https://stackoverflow.com/questions/49565047/pprint-with-hex-numbers):

    - the entry point is the _format() method which calls format() (overridden
      here)
    - format() directly calls _safe_repr() for a first try at rendering the
      object
    - _safe_repr formats the whole object recursively, only calling itself,
      not caring about line length or anything
    - back to _format(), if the output string is too long, _format() then calls
      the appropriate _pprint_TYPE() method (e.g. _pprint_list()) depending on
      the type of the object. This where the line length and the compact
      parameters are taken into account.
    - those _pprint_TYPE() methods will internally use the format() method for
      rendering the nested objects of an object (e.g. the elements of a list)

    In the end, everything has to be implemented twice: in _safe_repr and in
    the custom _pprint_TYPE methods. Unfortunately PrettyPrinter is really not
    straightforward to extend (especially when we want a compact output), so
    the code is a bit convoluted.

    This class overrides:
    - format() to support the changed_only parameter
    - _safe_repr to support printing of estimators (for when they fit on a
      single line)
    - _format_dict_items so that dict are correctly 'compacted'
    - _format_items so that ellipsis is used on long lists and tuples

    When estimators cannot be printed on a single line, the builtin _format()
    will call _pprint_estimator() because it was registered to do so (see
    _dispatch[BaseEstimator.__repr__] = _pprint_estimator).

    both _format_dict_items() and _pprint_estimator() use the
    _format_params_or_dict_items() method that will format parameters and
    key-value pairs respecting the compact parameter. This method needs another
    subroutine _pprint_key_val_tuple() used when a parameter or a key-value
    pair is too long to fit on a single line. This subroutine is called in
    _format() and is registered as well in the _dispatch dict (just like
    _pprint_estimator). We had to create the two classes KeyValTuple and
    KeyValTupleParam for this.
    """

    def __init__(
        self,
        indent=1,
        width=80,
        depth=None,
        stream=None,
        *,
        compact=False,
        indent_at_name=True,
        n_max_elements_to_show=None,
    ):
        # [FUNCTION DOC]
        # Inputs: indent, width, depth, stream, compact - as for pprint.PrettyPrinter;
        # indent_at_name - bool, default True; when True continuation lines align after
        # the
        # estimator name and the per-level indent is forced to 1 (indent is ignored);
        # n_max_elements_to_show - int or None, max items shown before an ellipsis.
        # Side effects: reads the global config print_changed_only and stores it in
        # _changed_only; sets _indent_at_name, _indent_per_level (when indent_at_name)
        # and
        # n_max_elements_to_show.
        super().__init__(indent, width, depth, stream, compact=compact)
        self._indent_at_name = indent_at_name
        if self._indent_at_name:
            self._indent_per_level = 1  # ignore indent param
        self._changed_only = get_config()["print_changed_only"]
        # Max number of elements in a list, dict, tuple until we start using
        # ellipsis. This also affects the number of arguments of an estimators
        # (they are treated as dicts)
        self.n_max_elements_to_show = n_max_elements_to_show

    def format(self, object, context, maxlevels, level):
        # [FUNCTION DOC]
        # Override of PrettyPrinter.format used for single-line attempts: delegates to
        # the module-level _safe_repr, passing the instance's changed-only setting.
        # Inputs: object, context (dict of ids being printed), maxlevels, level.
        # Outputs: tuple (repr string, readable flag, recursive flag).
        return _safe_repr(
            object, context, maxlevels, level, changed_only=self._changed_only
        )

    def _pprint_estimator(self, object, stream, indent, allowance, context, level):
        # [FUNCTION DOC]
        # HELPER: Multi-line renderer for estimators, registered in _dispatch. Writes
        # the class name and '(', bumps indent by the name length when indent_at_name,
        # gets
        # either the changed parameters or all shallow params, sorts them by name,
        # formats
        # them via _format_params (with allowance increased by 1 for the closing paren)
        # and
        # writes ')'.
        # Inputs: object - estimator; stream - writable text stream; indent, allowance,
        # context, level - pprint layout state. Outputs: none. Side effects: writes to
        # stream.
        stream.write(object.__class__.__name__ + "(")
        if self._indent_at_name:
            indent += len(object.__class__.__name__)

        if self._changed_only:
            params = _changed_params(object)
        else:
            params = object.get_params(deep=False)

        self._format_params(
            sorted(params.items()), stream, indent, allowance + 1, context, level
        )
        stream.write(")")

    def _format_dict_items(self, items, stream, indent, allowance, context, level):
        # [FUNCTION DOC]
        # HELPER: Override of the builtin dict-item formatter; delegates to
        # _format_params_or_dict_items with is_dict True so dicts honour compact and the
        # ellipsis limit. Outputs: None. Side effects: writes to stream.
        return self._format_params_or_dict_items(
            items, stream, indent, allowance, context, level, is_dict=True
        )

    def _format_params(self, items, stream, indent, allowance, context, level):
        # [FUNCTION DOC]
        # HELPER: Formats estimator (name, value) pairs by delegating to
        # _format_params_or_dict_items with is_dict False. Side effects: writes to
        # stream.
        return self._format_params_or_dict_items(
            items, stream, indent, allowance, context, level, is_dict=False
        )

    def _format_params_or_dict_items(
        self, object, stream, indent, allowance, context, level, is_dict
    ):
        # [FUNCTION DOC]
        # HELPER: Writes a sequence of (key, value) pairs, separated by commas.
        # Algorithm: increase indent by the per-level indent; available width is
        # width minus indent plus 1. Iterate with one-item lookahead; stop and write ',
        # ...'
        # once n_max_elements_to_show items are written. On the last item, reduce the
        # width
        # budget by allowance. In compact mode, build 'key: value' (dict) or 'key=value'
        # (params, key quotes stripped) from single-line reprs; if it does not fit the
        # remaining width, reset width and switch the delimiter to a newline+indent; if
        # it
        # fits, write it and continue. Otherwise (non-compact or too long) write the
        # delimiter and recursively _format the pair wrapped as KeyValTuple or
        # KeyValTupleParam, so it dispatches to _pprint_key_val_tuple.
        # Inputs: object - iterable of 2-tuples; stream, indent, allowance, context,
        # level -
        # pprint state; is_dict - bool selecting the rendering style.
        # Outputs: None (returns early on empty input). Side effects: writes to stream.
        """Format dict items or parameters respecting the compact=True
        parameter. For some reason, the builtin rendering of dict items doesn't
        respect compact=True and will use one line per key-value if all cannot
        fit in a single line.
        Dict items will be rendered as <'key': value> while params will be
        rendered as <key=value>. The implementation is mostly copy/pasting from
        the builtin _format_items().
        This also adds ellipsis if the number of items is greater than
        self.n_max_elements_to_show.
        """
        write = stream.write
        indent += self._indent_per_level
        delimnl = ",\n" + " " * indent
        delim = ""
        width = max_width = self._width - indent + 1
        it = iter(object)
        try:
            next_ent = next(it)
        except StopIteration:
            return
        last = False
        n_items = 0
        while not last:
            if n_items == self.n_max_elements_to_show:
                write(", ...")
                break
            n_items += 1
            ent = next_ent
            try:
                next_ent = next(it)
            except StopIteration:
                last = True
                max_width -= allowance
                width -= allowance
            if self._compact:
                k, v = ent
                krepr = self._repr(k, context, level)
                vrepr = self._repr(v, context, level)
                if not is_dict:
                    krepr = krepr.strip("'")
                middle = ": " if is_dict else "="
                rep = krepr + middle + vrepr
                w = len(rep) + 2
                if width < w:
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    width -= w
                    write(delim)
                    delim = ", "
                    write(rep)
                    continue
            write(delim)
            delim = delimnl
            class_ = KeyValTuple if is_dict else KeyValTupleParam
            self._format(
                class_(ent), stream, indent, allowance if last else 1, context, level
            )

    def _format_items(self, items, stream, indent, allowance, context, level):
        # [FUNCTION DOC]
        # HELPER: Same algorithm as the builtin _format_items for lists/tuples, plus
        # truncation: after n_max_elements_to_show items writes ', ...' and stops.
        # Writes
        # extra leading spaces when the per-level indent exceeds 1. In compact mode
        # packs
        # single-line reprs on a line while they fit; otherwise each item is recursively
        # formatted on its own line.
        # Inputs: items - iterable; stream, indent, allowance, context, level - pprint
        # state.
        # Outputs: None. Side effects: writes to stream.
        """Format the items of an iterable (list, tuple...). Same as the
        built-in _format_items, with support for ellipsis if the number of
        elements is greater than self.n_max_elements_to_show.
        """
        write = stream.write
        indent += self._indent_per_level
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * " ")
        delimnl = ",\n" + " " * indent
        delim = ""
        width = max_width = self._width - indent + 1
        it = iter(items)
        try:
            next_ent = next(it)
        except StopIteration:
            return
        last = False
        n_items = 0
        while not last:
            if n_items == self.n_max_elements_to_show:
                write(", ...")
                break
            n_items += 1
            ent = next_ent
            try:
                next_ent = next(it)
            except StopIteration:
                last = True
                max_width -= allowance
                width -= allowance
            if self._compact:
                rep = self._repr(ent, context, level)
                w = len(rep) + 2
                if width < w:
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    width -= w
                    write(delim)
                    delim = ", "
                    write(rep)
                    continue
            write(delim)
            delim = delimnl
            self._format(ent, stream, indent, allowance if last else 1, context, level)

    def _pprint_key_val_tuple(self, object, stream, indent, allowance, context, level):
        # [FUNCTION DOC]
        # HELPER: Multi-line renderer for one KeyValTuple/KeyValTupleParam, registered
        # in _dispatch. Writes the key repr (quotes stripped and '=' separator for
        # params,
        # ': ' for dict items) then recursively formats the value with indent advanced
        # by
        # the key and separator length.
        # Inputs: object - 2-tuple wrapper; stream, indent, allowance, context, level.
        # Outputs: None. Side effects: writes to stream.
        """Pretty printing for key-value tuples from dict or parameters."""
        k, v = object
        rep = self._repr(k, context, level)
        if isinstance(object, KeyValTupleParam):
            rep = rep.strip("'")
            middle = "="
        else:
            middle = ": "
        stream.write(rep)
        stream.write(middle)
        self._format(
            v, stream, indent + len(rep) + len(middle), allowance, context, level
        )

    # Note: need to copy _dispatch to prevent instances of the builtin
    # PrettyPrinter class to call methods of _EstimatorPrettyPrinter (see issue
    # 12906)
    # pyrefly error: "Class `PrettyPrinter` has no class attribute `_dispatch`"
    _dispatch = (
        pprint.PrettyPrinter._dispatch.copy()  # pyrefly: ignore[missing-attribute]
    )
    _dispatch[BaseEstimator.__repr__] = _pprint_estimator
    _dispatch[KeyValTuple.__repr__] = _pprint_key_val_tuple


def _safe_repr(object, context, maxlevels, level, changed_only=False):
    # [FUNCTION DOC]
    # Single-line recursive repr used by _EstimatorPrettyPrinter.format, mirroring
    # pprint._safe_repr with estimator support. Builtin scalars use repr. Plain dicts
    # (not overriding __repr__) render as sorted 'k: v' items in braces; plain lists and
    # tuples render their elements (one-tuple gets a trailing comma); estimators render
    # as ClassName(k=v, ...) using changed or all shallow params sorted by name. For
    # containers/estimators: depth beyond maxlevels gives a '...' placeholder and an
    # id already in context gives a recursion marker; the id is added to context while
    # children are rendered and removed afterwards. Anything else uses repr.
    # Inputs: object - any; context - dict of object ids being rendered (mutated then
    # restored); maxlevels - int or falsy for unlimited; level - current depth;
    # changed_only - bool, default False.
    # Outputs: tuple (string, readable, recursive). readable is False if anything is
    # truncated or recursive, or the fallback repr starts with '<'.
    # KNOWN ISSUE: for the fallback case the readable flag is computed as a truthy
    # expression and can be the empty string instead of a bool when repr is empty.
    # Note: truncation via n_max_elements_to_show is not applied here, only in
    # multi-line mode.
    """Same as the builtin _safe_repr, with added support for Estimator
    objects."""
    typ = type(object)

    if typ in pprint._builtin_scalars:
        return repr(object), True, False

    r = getattr(typ, "__repr__", None)
    if issubclass(typ, dict) and r is dict.__repr__:
        if not object:
            return "{}", True, False
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return "{...}", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        saferepr = _safe_repr
        items = sorted(object.items(), key=pprint._safe_tuple)
        for k, v in items:
            krepr, kreadable, krecur = saferepr(
                k, context, maxlevels, level, changed_only=changed_only
            )
            vrepr, vreadable, vrecur = saferepr(
                v, context, maxlevels, level, changed_only=changed_only
            )
            append("%s: %s" % (krepr, vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return "{%s}" % ", ".join(components), readable, recursive

    if (issubclass(typ, list) and r is list.__repr__) or (
        issubclass(typ, tuple) and r is tuple.__repr__
    ):
        if issubclass(typ, list):
            if not object:
                return "[]", True, False
            format = "[%s]"
        elif len(object) == 1:
            format = "(%s,)"
        else:
            if not object:
                return "()", True, False
            format = "(%s)"
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return format % "...", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        for o in object:
            orepr, oreadable, orecur = _safe_repr(
                o, context, maxlevels, level, changed_only=changed_only
            )
            append(orepr)
            if not oreadable:
                readable = False
            if orecur:
                recursive = True
        del context[objid]
        return format % ", ".join(components), readable, recursive

    if issubclass(typ, BaseEstimator):
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return f"{typ.__name__}(...)", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        if changed_only:
            params = _changed_params(object)
        else:
            params = object.get_params(deep=False)
        components = []
        append = components.append
        level += 1
        saferepr = _safe_repr
        items = sorted(params.items(), key=pprint._safe_tuple)
        for k, v in items:
            krepr, kreadable, krecur = saferepr(
                k, context, maxlevels, level, changed_only=changed_only
            )
            vrepr, vreadable, vrecur = saferepr(
                v, context, maxlevels, level, changed_only=changed_only
            )
            append("%s=%s" % (krepr.strip("'"), vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return ("%s(%s)" % (typ.__name__, ", ".join(components)), readable, recursive)

    rep = repr(object)
    return rep, (rep and not rep.startswith("<")), False
