# [MODULE DOC]
# Vendored copy of numpydoc's docscrape module: parses numpydoc-formatted docstrings
#   into a section mapping and renders them back to reST text. scikit-learn uses it
#   (e.g. in docstring tests and deprecation/metadata-routing doc generation) without
#   depending on numpydoc. Defines Reader, ParseError, the Parameter named tuple (name,
#   type, desc), NumpyDocString and its FunctionDoc/ObjDoc/ClassDoc subclasses, plus
#   get_doc_object.
"""Extract reference documentation from the NumPy source tree."""

import copy
import inspect
import pydoc
import re
import sys
import textwrap
from collections import namedtuple
from collections.abc import Callable, Mapping
from functools import cached_property
from warnings import warn


def strip_blank_lines(l):
    # [FUNCTION DOC]
    # HELPER: Removes leading and trailing whitespace-only strings from the list l in
    #   place and returns the same list.
    "Remove leading and trailing blank lines from a list of lines"
    while l and not l[0].strip():
        del l[0]
    while l and not l[-1].strip():
        del l[-1]
    return l


class Reader:
    # [CLASS DOC]
    # HELPER: Cursor over a list of lines with helpers to read blocks delimited by empty
    #   or unindented lines; the current position is stored in _l.
    """A line-based string reader."""

    def __init__(self, data):
        # [FUNCTION DOC]
        # Stores data as a list of lines (a list is used as-is, a string is split on
        #   newlines) in _str and resets the cursor to 0.
        """
        Parameters
        ----------
        data : str
           String with lines separated by '\\n'.

        """
        if isinstance(data, list):
            self._str = data
        else:
            self._str = data.split("\n")  # store string as list of lines

        self.reset()

    def __getitem__(self, n):
        # [FUNCTION DOC]
        # Returns the line (or slice of lines) at index n of the stored list.
        return self._str[n]

    def reset(self):
        # [FUNCTION DOC]
        # Sets the cursor back to line 0.
        self._l = 0  # current line nr

    def read(self):
        # [FUNCTION DOC]
        # Returns the current line and advances the cursor; returns "" at end of input.
        if not self.eof():
            out = self[self._l]
            self._l += 1
            return out
        else:
            return ""

    def seek_next_non_empty_line(self):
        # [FUNCTION DOC]
        # Advances the cursor past whitespace-only lines.
        for l in self[self._l :]:
            if l.strip():
                break
            else:
                self._l += 1

    def eof(self):
        # [FUNCTION DOC]
        # True when the cursor is at or past the last line.
        return self._l >= len(self._str)

    def read_to_condition(self, condition_func):
        # [FUNCTION DOC]
        # Reads lines from the cursor until condition_func(line) is true and returns
        #   them (the matching line is not consumed); if input ends first, returns all
        #   remaining lines. Returns [] if already at eof.
        # Side effects: advances the cursor.
        start = self._l
        for line in self[start:]:
            if condition_func(line):
                return self[start : self._l]
            self._l += 1
            if self.eof():
                return self[start : self._l + 1]
        return []

    def read_to_next_empty_line(self):
        # [FUNCTION DOC]
        # Skips leading empty lines, then reads up to (not including) the next empty
        #   line.
        self.seek_next_non_empty_line()

        def is_empty(line):
            # [FUNCTION DOC]
            # HELPER: True for whitespace-only lines.
            return not line.strip()

        return self.read_to_condition(is_empty)

    def read_to_next_unindented_line(self):
        # [FUNCTION DOC]
        # Reads lines until a non-empty line that starts at column 0 (used to collect
        #   indented descriptions).
        def is_unindented(line):
            # [FUNCTION DOC]
            # HELPER: Truthy for a non-blank line without leading whitespace.
            return line.strip() and (len(line.lstrip()) == len(line))

        return self.read_to_condition(is_unindented)

    def peek(self, n=0):
        # [FUNCTION DOC]
        # Returns the line n positions after the cursor without moving it, or "" if out
        #   of range. KNOWN ISSUE: negative n (used as peek(-1)) at cursor 0 wraps to
        #   the last line via Python negative indexing.
        if self._l + n < len(self._str):
            return self[self._l + n]
        else:
            return ""

    def is_empty(self):
        # [FUNCTION DOC]
        # True if all lines joined together are only whitespace.
        return not "".join(self._str).strip()


class ParseError(Exception):
    # [CLASS DOC]
    # Exception for docstring parse failures; the docstring attribute may be attached by
    #   NumpyDocString.
    def __str__(self):
        # [FUNCTION DOC]
        # Returns the first argument, suffixed with " in <docstring repr>" when a
        #   docstring attribute was attached.
        message = self.args[0]
        if hasattr(self, "docstring"):
            message = f"{message} in {self.docstring!r}"
        return message


Parameter = namedtuple("Parameter", ["name", "type", "desc"])


class NumpyDocString(Mapping):
    # [CLASS DOC]
    # Parses a numpydoc docstring into a Mapping from section titles to content (lists
    #   of lines, lists of Parameter, See Also items or the index dict); sections holds
    #   the defaults deep-copied per instance. str() renders it back to numpydoc text.
    """Parses a numpydoc string to an abstract representation

    Instances define a mapping from section title to structured data.

    """

    sections = {
        "Signature": "",
        "Summary": [""],
        "Extended Summary": [],
        "Parameters": [],
        "Attributes": [],
        "Methods": [],
        "Returns": [],
        "Yields": [],
        "Receives": [],
        "Other Parameters": [],
        "Raises": [],
        "Warns": [],
        "Warnings": [],
        "See Also": [],
        "Notes": [],
        "References": "",
        "Examples": "",
        "index": {},
    }

    def __init__(self, docstring, config=None):
        # [FUNCTION DOC]
        # Dedents and splits the docstring into lines, wraps them in a Reader,
        #   initializes _parsed_data from a deep copy of sections and parses.
        # Inputs: docstring - str; config - accepted but unused.
        # Exceptions: ParseError from parsing is re-raised with the original docstring
        #   attached; ValueError for malformed docstrings (duplicate sections, bad See
        #   Also, Receives without Yields).
        orig_docstring = docstring
        docstring = textwrap.dedent(docstring).split("\n")

        self._doc = Reader(docstring)
        self._parsed_data = copy.deepcopy(self.sections)

        try:
            self._parse()
        except ParseError as e:
            e.docstring = orig_docstring
            raise

    def __getitem__(self, key):
        # [FUNCTION DOC]
        # Returns the parsed content of section key (KeyError if unknown).
        return self._parsed_data[key]

    def __setitem__(self, key, val):
        # [FUNCTION DOC]
        # Sets a known section; for an unknown section name emits a UserWarning via
        #   _error_location instead of storing it.
        if key not in self._parsed_data:
            self._error_location(f"Unknown section {key}", error=False)
        else:
            self._parsed_data[key] = val

    def __iter__(self):
        # [FUNCTION DOC]
        # Iterates over section names.
        return iter(self._parsed_data)

    def __len__(self):
        # [FUNCTION DOC]
        # Number of sections (always the fixed set of known sections).
        return len(self._parsed_data)

    def _is_at_section(self):
        # [FUNCTION DOC]
        # HELPER: Skips blank lines and decides whether the cursor is at a section
        #   header: true for an ".. index::" line, or when the next line starts with "-"
        #   or "=" repeated at least the header's length. Warns when a dash/equals
        #   underline of length >= 3 differs in length from the title. Returns False at
        #   eof.
        self._doc.seek_next_non_empty_line()

        if self._doc.eof():
            return False

        l1 = self._doc.peek().strip()  # e.g. Parameters

        if l1.startswith(".. index::"):
            return True

        l2 = self._doc.peek(1).strip()  # ---------- or ==========
        if len(l2) >= 3 and (set(l2) in ({"-"}, {"="})) and len(l2) != len(l1):
            snip = "\n".join(self._doc._str[:2]) + "..."
            self._error_location(
                f"potentially wrong underline length... \n{l1} \n{l2} in \n{snip}",
                error=False,
            )
        return l2.startswith("-" * len(l1)) or l2.startswith("=" * len(l1))

    def _strip(self, doc):
        # [FUNCTION DOC]
        # HELPER: Returns doc without leading and trailing blank lines (non-mutating).
        # KNOWN ISSUE: for an all-blank list i and j end at the last index, so a single
        #   trailing element can remain.
        i = 0
        j = 0
        for i, line in enumerate(doc):
            if line.strip():
                break

        for j, line in enumerate(doc[::-1]):
            if line.strip():
                break

        return doc[i : len(doc) - j]

    def _read_to_next_section(self):
        # [FUNCTION DOC]
        # HELPER: Reads paragraphs until the next section header or eof, re-inserting a
        #   single "" between paragraphs separated by blank lines; returns the list of
        #   lines.
        section = self._doc.read_to_next_empty_line()

        while not self._is_at_section() and not self._doc.eof():
            if not self._doc.peek(-1).strip():  # previous line was empty
                section += [""]

            section += self._doc.read_to_next_empty_line()

        return section

    def _read_sections(self):
        # [FUNCTION DOC]
        # HELPER: Generator over remaining sections: yields (name, lines after the
        #   header) for ".." index sections, (title, stripped body after the underline)
        #   for normal ones.
        # KNOWN ISSUE: for a block with fewer than 2 lines it yields the StopIteration
        #   class instead of stopping, which makes the caller's tuple unpacking fail
        #   with a TypeError.
        while not self._doc.eof():
            data = self._read_to_next_section()
            name = data[0].strip()

            if name.startswith(".."):  # index section
                yield name, data[1:]
            elif len(data) < 2:
                yield StopIteration
            else:
                yield name, self._strip(data[2:])

    def _parse_param_list(self, content, single_element_is_type=False):
        # [FUNCTION DOC]
        # HELPER: Parses a Parameters-like block into a list of Parameter tuples.
        # Behavior: dedent; for each unindented header line split on the first " : "
        #   into name and type; otherwise strip a trailing " :" and treat the line as
        #   the type (single_element_is_type, used for Returns/Raises etc.) or as the
        #   name; the following indented lines become the dedented, blank-stripped
        #   description list.
        content = dedent_lines(content)
        r = Reader(content)
        params = []
        while not r.eof():
            header = r.read().strip()
            if " : " in header:
                arg_name, arg_type = header.split(" : ", maxsplit=1)
            else:
                # NOTE: param line with single element should never have a
                # a " :" before the description line, so this should probably
                # warn.
                header = header.removesuffix(" :")
                if single_element_is_type:
                    arg_name, arg_type = "", header
                else:
                    arg_name, arg_type = header, ""

            desc = r.read_to_next_unindented_line()
            desc = dedent_lines(desc)
            desc = strip_blank_lines(desc)

            params.append(Parameter(arg_name, arg_type, desc))

        return params

    # See also supports the following formats.
    #
    # <FUNCNAME>
    # <FUNCNAME> SPACE* COLON SPACE+ <DESC> SPACE*
    # <FUNCNAME> ( COMMA SPACE+ <FUNCNAME>)+ (COMMA | PERIOD)? SPACE*
    # <FUNCNAME> ( COMMA SPACE+ <FUNCNAME>)* SPACE* COLON SPACE+ <DESC> SPACE*

    # <FUNCNAME> is one of
    #   <PLAIN_FUNCNAME>
    #   COLON <ROLE> COLON BACKTICK <PLAIN_FUNCNAME> BACKTICK
    # where
    #   <PLAIN_FUNCNAME> is a legal function name, and
    #   <ROLE> is any nonempty sequence of word characters.
    # Examples: func_f1  :meth:`func_h1` :obj:`~baz.obj_r` :class:`class_j`
    # <DESC> is a string describing the function.

    _role = r":(?P<role>(py:)?\w+):"
    _funcbacktick = r"`(?P<name>(?:~\w+\.)?[a-zA-Z0-9_\.-]+)`"
    _funcplain = r"(?P<name2>[a-zA-Z0-9_\.-]+)"
    _funcname = r"(" + _role + _funcbacktick + r"|" + _funcplain + r")"
    _funcnamenext = _funcname.replace("role", "rolenext")
    _funcnamenext = _funcnamenext.replace("name", "namenext")
    _description = r"(?P<description>\s*:(\s+(?P<desc>\S+.*))?)?\s*$"
    _func_rgx = re.compile(r"^\s*" + _funcname + r"\s*")
    _line_rgx = re.compile(
        r"^\s*"
        + r"(?P<allfuncs>"
        + _funcname  # group for all function names
        + r"(?P<morefuncs>([,]\s+"
        + _funcnamenext
        + r")*)"
        + r")"
        + r"(?P<trailing>[,\.])?"  # end of "allfuncs"
        + _description  # Some function lists have a trailing comma (or period)  '\s*'
    )

    # Empty <DESC> elements are replaced with '..'
    empty_description = ".."

    def _parse_see_also(self, content):
        # [FUNCTION DOC]
        # HELPER: Parses See Also lines into a list of (funcs, description_lines) where
        #   funcs is a list of (name, role or None).
        # Behavior: skip blank lines; match each line with _line_rgx; a non-matching or
        #   description-less indented line is appended to the description of the
        #   previous item; otherwise split the function list on commas, extracting each
        #   name/role with parse_item_name, and start a new item whose description is
        #   the text after the colon. Warns on a trailing comma/period followed by a
        #   description.
        # Exceptions: ValueError (via _error_location) for unparseable entries.
        # KNOWN ISSUE: rest is rebound to a new list per item, so continuation lines
        #   extend the latest item's description only through that shared list object,
        #   which works but is fragile.
        """
        func_name : Descriptive text
            continued text
        another_func_name : Descriptive text
        func_name1, func_name2, :meth:`func_name`, func_name3

        """

        content = dedent_lines(content)

        items = []

        def parse_item_name(text):
            # [FUNCTION DOC]
            # HELPER: Matches one function reference (":role:`name`" or plain name) at
            #   the start of text and returns (name, role, end offset). Raises
            #   ValueError via _error_location if nothing matches (references the
            #   enclosing loop variable line in the message).
            """Match ':role:`name`' or 'name'."""
            m = self._func_rgx.match(text)
            if not m:
                self._error_location(f"Error parsing See Also entry {line!r}")
            role = m.group("role")
            name = m.group("name") if role else m.group("name2")
            return name, role, m.end()

        rest = []
        for line in content:
            if not line.strip():
                continue

            line_match = self._line_rgx.match(line)
            description = None
            if line_match:
                description = line_match.group("desc")
                if line_match.group("trailing") and description:
                    self._error_location(
                        "Unexpected comma or period after function list at index %d of "
                        'line "%s"' % (line_match.end("trailing"), line),
                        error=False,
                    )
            if not description and line.startswith(" "):
                rest.append(line.strip())
            elif line_match:
                funcs = []
                text = line_match.group("allfuncs")
                while True:
                    if not text.strip():
                        break
                    name, role, match_end = parse_item_name(text)
                    funcs.append((name, role))
                    text = text[match_end:].strip()
                    if text and text[0] == ",":
                        text = text[1:].strip()
                rest = list(filter(None, [description]))
                items.append((funcs, rest))
            else:
                self._error_location(f"Error parsing See Also entry {line!r}")
        return items

    def _parse_index(self, section, content):
        # [FUNCTION DOC]
        # HELPER: Parses an ".. index:: default" section into a dict: "default" maps to
        #   the first comma-separated entry after "::", and each ":key: a, b" content
        #   line maps key to the stripped list of values.
        """
        .. index:: default
           :refguide: something, else, and more

        """

        def strip_each_in(lst):
            # [FUNCTION DOC]
            # HELPER: Returns a list with every string stripped.
            return [s.strip() for s in lst]

        out = {}
        section = section.split("::")
        if len(section) > 1:
            out["default"] = strip_each_in(section[1].split(","))[0]
        for line in content:
            line = line.split(":")
            if len(line) > 2:
                out[line[1]] = strip_each_in(line[2].split(","))
        return out

    def _parse_summary(self):
        # [FUNCTION DOC]
        # HELPER: Reads the optional signature and summary at the top. Repeatedly reads
        #   paragraphs: a paragraph that looks like a call signature is stored as
        #   Signature and reading continues; the first non-signature paragraph becomes
        #   Summary; then, unless a section header follows, the rest up to the next
        #   section becomes Extended Summary. Returns early if the docstring starts with
        #   a section.
        # KNOWN ISSUE: the "summary is not None" check is always true, and if the last
        #   paragraph was a signature followed by a section, it is stored both as
        #   Signature and Summary.
        """Grab signature (if given) and summary"""
        if self._is_at_section():
            return

        # If several signatures present, take the last one
        while True:
            summary = self._doc.read_to_next_empty_line()
            summary_str = " ".join([s.strip() for s in summary]).strip()
            compiled = re.compile(r"^([\w., ]+=)?\s*[\w\.]+\(.*\)$")
            if compiled.match(summary_str):
                self["Signature"] = summary_str
                if not self._is_at_section():
                    continue
            break

        if summary is not None:
            self["Summary"] = summary

        if not self._is_at_section():
            self["Extended Summary"] = self._read_to_next_section()

    def _parse(self):
        # [FUNCTION DOC]
        # HELPER: Top-level parse: reset the reader, parse the summary, collect all
        #   sections, reject Receives without Yields (ValueError), normalize each title
        #   to capitalized words, error on duplicate sections, and dispatch: parameter-
        #   style sections to _parse_param_list (Returns/Yields/Raises/Warns/Receives
        #   with single_element_is_type), index sections to _parse_index, See Also to
        #   _parse_see_also, everything else stored as raw lines. Unknown titles only
        #   warn (via __setitem__).
        self._doc.reset()
        self._parse_summary()

        sections = list(self._read_sections())
        section_names = {section for section, content in sections}

        has_yields = "Yields" in section_names
        # We could do more tests, but we are not. Arbitrarily.
        if not has_yields and "Receives" in section_names:
            msg = "Docstring contains a Receives section but not Yields."
            raise ValueError(msg)

        for section, content in sections:
            if not section.startswith(".."):
                section = (s.capitalize() for s in section.split(" "))
                section = " ".join(section)
                if self.get(section):
                    self._error_location(
                        "The section %s appears twice in  %s"
                        % (section, "\n".join(self._doc._str))
                    )

            if section in ("Parameters", "Other Parameters", "Attributes", "Methods"):
                self[section] = self._parse_param_list(content)
            elif section in ("Returns", "Yields", "Raises", "Warns", "Receives"):
                self[section] = self._parse_param_list(
                    content, single_element_is_type=True
                )
            elif section.startswith(".. index::"):
                self["index"] = self._parse_index(section, content)
            elif section == "See Also":
                self["See Also"] = self._parse_see_also(content)
            else:
                self[section] = content

    @property
    def _obj(self):
        # [FUNCTION DOC]
        # Property: the documented object (_cls if set, else _f, else None), used for
        #   error messages.
        if hasattr(self, "_cls"):
            return self._cls
        elif hasattr(self, "_f"):
            return self._f
        return None

    def _error_location(self, msg, error=True):
        # [FUNCTION DOC]
        # HELPER: Augments msg with the documented object's name and source file (when
        #   available) and either raises ValueError (error=True) or emits a UserWarning
        #   with stacklevel 3.
        if self._obj is not None:
            # we know where the docs came from:
            try:
                filename = inspect.getsourcefile(self._obj)
            except TypeError:
                filename = None
            # Make UserWarning more descriptive via object introspection.
            # Skip if introspection fails
            name = getattr(self._obj, "__name__", None)
            if name is None:
                name = getattr(getattr(self._obj, "__class__", None), "__name__", None)
            if name is not None:
                msg += f" in the docstring of {name}"
            msg += f" in {filename}." if filename else ""
        if error:
            raise ValueError(msg)
        else:
            warn(msg, stacklevel=3)

    # string conversion routines

    def _str_header(self, name, symbol="-"):
        # [FUNCTION DOC]
        # HELPER: Returns [name, underline] with the underline made of symbol repeated
        #   len(name) times.
        return [name, len(name) * symbol]

    def _str_indent(self, doc, indent=4):
        # [FUNCTION DOC]
        # HELPER: Returns the lines prefixed with indent spaces.
        return [" " * indent + line for line in doc]

    def _str_signature(self):
        # [FUNCTION DOC]
        # HELPER: Returns [signature with "*" escaped, ""] or [""] if no signature.
        if self["Signature"]:
            return [self["Signature"].replace("*", r"\*")] + [""]
        return [""]

    def _str_summary(self):
        # [FUNCTION DOC]
        # HELPER: Returns the summary lines plus a blank line, or [] if empty.
        if self["Summary"]:
            return self["Summary"] + [""]
        return []

    def _str_extended_summary(self):
        # [FUNCTION DOC]
        # HELPER: Returns the extended summary lines plus a blank line, or [].
        if self["Extended Summary"]:
            return self["Extended Summary"] + [""]
        return []

    def _str_param_list(self, name):
        # [FUNCTION DOC]
        # HELPER: Renders a Parameter list section: header, then per item "name : type"
        #   (omitting empty parts) and its description indented by 4 when non-blank,
        #   then a blank line; [] if the section is empty.
        out = []
        if self[name]:
            out += self._str_header(name)
            for param in self[name]:
                parts = []
                if param.name:
                    parts.append(param.name)
                if param.type:
                    parts.append(param.type)
                out += [" : ".join(parts)]
                if param.desc and "".join(param.desc).strip():
                    out += self._str_indent(param.desc)
            out += [""]
        return out

    def _str_section(self, name):
        # [FUNCTION DOC]
        # HELPER: Renders a raw-lines section as header, lines and a blank line, or []
        #   if empty.
        out = []
        if self[name]:
            out += self._str_header(name)
            out += self[name]
            out += [""]
        return out

    def _str_see_also(self, func_role):
        # [FUNCTION DOC]
        # HELPER: Renders See Also: each function list becomes comma-joined links
        #   (":role:`f`" using the item role, else func_role, else "`f`_"), followed by
        #   its indented description or ".." when absent; ends with blank lines. Returns
        #   [] if empty.
        if not self["See Also"]:
            return []
        out = []
        out += self._str_header("See Also")
        out += [""]
        last_had_desc = True
        for funcs, desc in self["See Also"]:
            assert isinstance(funcs, list)
            links = []
            for func, role in funcs:
                if role:
                    link = f":{role}:`{func}`"
                elif func_role:
                    link = f":{func_role}:`{func}`"
                else:
                    link = f"`{func}`_"
                links.append(link)
            link = ", ".join(links)
            out += [link]
            if desc:
                out += self._str_indent([" ".join(desc)])
                last_had_desc = True
            else:
                last_had_desc = False
                out += self._str_indent([self.empty_description])

        if last_had_desc:
            out += [""]
        out += [""]
        return out

    def _str_index(self):
        # [FUNCTION DOC]
        # HELPER: Renders the ".. index::" directive with the default entry and
        #   ":section: refs" lines; returns "" when there is neither a default nor any
        #   other entry.
        idx = self["index"]
        out = []
        output_index = False
        default_index = idx.get("default", "")
        if default_index:
            output_index = True
        out += [f".. index:: {default_index}"]
        for section, references in idx.items():
            if section == "default":
                continue
            output_index = True
            out += [f"   :{section}: {', '.join(references)}"]
        if output_index:
            return out
        return ""

    def __str__(self, func_role=""):
        # [FUNCTION DOC]
        # Renders all sections in canonical numpydoc order (signature, summaries,
        #   parameter lists, Warnings, See Also, Notes, References, Examples, index) and
        #   joins them with newlines. func_role is the default role for See Also links.
        out = []
        out += self._str_signature()
        out += self._str_summary()
        out += self._str_extended_summary()
        out += self._str_param_list("Parameters")
        for param_list in ("Attributes", "Methods"):
            out += self._str_param_list(param_list)
        for param_list in (
            "Returns",
            "Yields",
            "Receives",
            "Other Parameters",
            "Raises",
            "Warns",
        ):
            out += self._str_param_list(param_list)
        out += self._str_section("Warnings")
        out += self._str_see_also(func_role)
        for s in ("Notes", "References", "Examples"):
            out += self._str_section(s)
        out += self._str_index()
        return "\n".join(out)


def dedent_lines(lines):
    # [FUNCTION DOC]
    # HELPER: Removes the common leading indentation from a list of lines and returns a
    #   new list.
    """Deindent a list of lines maximally"""
    return textwrap.dedent("\n".join(lines)).split("\n")


class FunctionDoc(NumpyDocString):
    # [CLASS DOC]
    # NumpyDocString for a function or method object; rendering prefixes a Sphinx
    #   function/method directive.
    def __init__(self, func, role="func", doc=None, config=None):
        # [FUNCTION DOC]
        # Stores func and role; when doc is None uses inspect.getdoc(func) (or "") and
        #   raises ValueError if func is also None; then parses.
        self._f = func
        self._role = role  # e.g. "func" or "meth"

        if doc is None:
            if func is None:
                raise ValueError("No function or docstring given")
            doc = inspect.getdoc(func) or ""
        if config is None:
            config = {}
        NumpyDocString.__init__(self, doc, config)

    def get_func(self):
        # [FUNCTION DOC]
        # Returns (callable, name): for classes the class's __call__ (or __init__),
        #   otherwise the object itself; name defaults to this class's name when the
        #   object has no __name__.
        func_name = getattr(self._f, "__name__", self.__class__.__name__)
        if inspect.isclass(self._f):
            func = getattr(self._f, "__call__", self._f.__init__)
        else:
            func = self._f
        return func, func_name

    def __str__(self):
        # [FUNCTION DOC]
        # Prepends ".. function:: name" / ".. method:: name" (empty directive name for
        #   unknown roles, after printing a warning to stdout) to the base rendering
        #   with the role used for See Also links.
        out = ""

        func, func_name = self.get_func()

        roles = {"func": "function", "meth": "method"}

        if self._role:
            if self._role not in roles:
                print(f"Warning: invalid role {self._role}")
            out += f".. {roles.get(self._role, '')}:: {func_name}\n    \n\n"

        out += super().__str__(func_role=self._role)
        return out


class ObjDoc(NumpyDocString):
    # [CLASS DOC]
    # NumpyDocString for an arbitrary object with an explicit docstring.
    def __init__(self, obj, doc=None, config=None):
        # [FUNCTION DOC]
        # Stores obj as _f and parses doc. KNOWN ISSUE: doc=None is not resolved here,
        #   so a direct ObjDoc(obj) call fails in textwrap.dedent; get_doc_object fills
        #   doc first.
        self._f = obj
        if config is None:
            config = {}
        NumpyDocString.__init__(self, doc, config=config)


class ClassDoc(NumpyDocString):
    # [CLASS DOC]
    # NumpyDocString for a class that auto-fills empty Methods and Attributes sections
    #   from the class's public members.
    extra_public_methods = ["__call__"]

    def __init__(self, cls, doc=None, modulename="", func_doc=FunctionDoc, config=None):
        # [FUNCTION DOC]
        # Validates cls is a class or None (ValueError otherwise), reads its docstring
        #   with pydoc.getdoc when doc is None (ValueError if both None), parses it,
        #   then unless show_class_members is False or exclude-members is Sphinx's ALL,
        #   fills empty Methods/Attributes with Parameter(name, "", docstring lines) for
        #   sorted public members, honouring members and exclude-members from config.
        # Inputs: cls; doc; modulename stored with a trailing dot in _mod; func_doc
        #   (unused); config dict.
        # Side effects: sets _cls, _mod, show_inherited_members; imports
        #   sphinx.ext.autodoc only if sphinx is already imported.
        if not inspect.isclass(cls) and cls is not None:
            raise ValueError(f"Expected a class or None, but got {cls!r}")
        self._cls = cls

        if "sphinx" in sys.modules:
            from sphinx.ext.autodoc import ALL
        else:
            ALL = object()

        if config is None:
            config = {}
        self.show_inherited_members = config.get("show_inherited_class_members", True)

        if modulename and not modulename.endswith("."):
            modulename += "."
        self._mod = modulename

        if doc is None:
            if cls is None:
                raise ValueError("No class or documentation string given")
            doc = pydoc.getdoc(cls)

        NumpyDocString.__init__(self, doc)

        _members = config.get("members", [])
        if _members is ALL:
            _members = None
        _exclude = config.get("exclude-members", [])

        if config.get("show_class_members", True) and _exclude is not ALL:

            def splitlines_x(s):
                # [FUNCTION DOC]
                # HELPER: Splits s into lines, returning [] for empty or None.
                if not s:
                    return []
                else:
                    return s.splitlines()

            for field, items in [
                ("Methods", self.methods),
                ("Attributes", self.properties),
            ]:
                if not self[field]:
                    doc_list = []
                    for name in sorted(items):
                        if name in _exclude or (_members and name not in _members):
                            continue
                        try:
                            doc_item = pydoc.getdoc(getattr(self._cls, name))
                            doc_list.append(Parameter(name, "", splitlines_x(doc_item)))
                        except AttributeError:
                            pass  # method doesn't exist
                    self[field] = doc_list

    @property
    def methods(self):
        # [FUNCTION DOC]
        # Property: names of callable members that are public (or __call__) and shown
        #   under the inherited-member policy; [] when no class.
        if self._cls is None:
            return []
        return [
            name
            for name, func in inspect.getmembers(self._cls)
            if (
                (not name.startswith("_") or name in self.extra_public_methods)
                and isinstance(func, Callable)
                and self._is_show_member(name)
            )
        ]

    @property
    def properties(self):
        # [FUNCTION DOC]
        # Property: names of public members that are None, property, cached_property or
        #   other data descriptors, excluding named tuple fields and honouring the
        #   inherited-member policy; [] when no class.
        if self._cls is None:
            return []
        return [
            name
            for name, func in inspect.getmembers(self._cls)
            if (
                not name.startswith("_")
                and not self._should_skip_member(name, self._cls)
                and (
                    func is None
                    or isinstance(func, (property, cached_property))
                    or inspect.isdatadescriptor(func)
                )
                and self._is_show_member(name)
            )
        ]

    @staticmethod
    def _should_skip_member(name, klass):
        # [FUNCTION DOC]
        # HELPER: True when klass is a namedtuple (tuple subclass with _asdict and
        #   _fields) and name is one of its fields.
        return (
            # Namedtuples should skip everything in their ._fields as the
            # docstrings for each of the members is: "Alias for field number X"
            issubclass(klass, tuple)
            and hasattr(klass, "_asdict")
            and hasattr(klass, "_fields")
            and name in klass._fields
        )

    def _is_show_member(self, name):
        # [FUNCTION DOC]
        # HELPER: True if inherited members are shown or name is defined directly in the
        #   class __dict__.
        return (
            # show all class members
            self.show_inherited_members
            # or class member is not inherited
            or name in self._cls.__dict__
        )


def get_doc_object(
    obj,
    what=None,
    doc=None,
    config=None,
    class_doc=ClassDoc,
    func_doc=FunctionDoc,
    obj_doc=ObjDoc,
):
    # [FUNCTION DOC]
    # Factory: infers what ("class", "module", "function" or "object") from obj when not
    #   given, then returns class_doc for classes, func_doc for functions/methods, or
    #   obj_doc (with doc defaulting to pydoc.getdoc(obj)) otherwise. Config defaults to
    #   an empty dict.
    if what is None:
        if inspect.isclass(obj):
            what = "class"
        elif inspect.ismodule(obj):
            what = "module"
        elif isinstance(obj, Callable):
            what = "function"
        else:
            what = "object"
    if config is None:
        config = {}

    if what == "class":
        return class_doc(obj, func_doc=func_doc, doc=doc, config=config)
    elif what in ("function", "method"):
        return func_doc(obj, doc=doc, config=config)
    else:
        if doc is None:
            doc = pydoc.getdoc(obj)
        return obj_doc(obj, doc, config=config)