# [MODULE DOC]
# Vendored liac-arff 2.4.0: reader/writer for Weka ARFF text files, used by
#   sklearn.datasets.fetch_openml. Provides load/loads/dump/dumps,
#   ArffDecoder/ArffEncoder, the return-type constants (DENSE, COO, LOD, DENSE_GEN,
#   LOD_GEN), the ArffException hierarchy, and regex/escaping helpers. Builds the dense
#   and sparse value regexes and escape maps at import time.
# =============================================================================
# Federal University of Rio Grande do Sul (UFRGS)
# Connectionist Artificial Intelligence Laboratory (LIAC)
# Renato de Pontes Pereira - rppereira@inf.ufrgs.br
# =============================================================================
# Copyright (c) 2011 Renato de Pontes Pereira, renato.ppontes at gmail dot com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

'''
The liac-arff module implements functions to read and write ARFF files in
Python. It was created in the Connectionist Artificial Intelligence Laboratory
(LIAC), which takes place at the Federal University of Rio Grande do Sul
(UFRGS), in Brazil.

ARFF (Attribute-Relation File Format) is an file format specially created for
describe datasets which are commonly used for machine learning experiments and
software. This file format was created to be used in Weka, the best
representative software for machine learning automated experiments.

An ARFF file can be divided into two sections: header and data. The Header
describes the metadata of the dataset, including a general description of the
dataset, its name and its attributes. The source below is an example of a
header section in a XOR dataset::

    %
    % XOR Dataset
    %
    % Created by Renato Pereira
    %            rppereira@inf.ufrgs.br
    %            http://inf.ufrgs.br/~rppereira
    %
    %
    @RELATION XOR

    @ATTRIBUTE input1 REAL
    @ATTRIBUTE input2 REAL
    @ATTRIBUTE y REAL

The Data section of an ARFF file describes the observations of the dataset, in
the case of XOR dataset::

    @DATA
    0.0,0.0,0.0
    0.0,1.0,1.0
    1.0,0.0,1.0
    1.0,1.0,0.0
    %
    %
    %

Notice that several lines are starting with an ``%`` symbol, denoting a
comment, thus, lines with ``%`` at the beginning will be ignored, except by the
description part at the beginning of the file. The declarations ``@RELATION``,
``@ATTRIBUTE``, and ``@DATA`` are all case insensitive and obligatory.

For more information and details about the ARFF file description, consult
http://www.cs.waikato.ac.nz/~ml/weka/arff.html


ARFF Files in Python
~~~~~~~~~~~~~~~~~~~~

This module uses built-ins python objects to represent a deserialized ARFF
file. A dictionary is used as the container of the data and metadata of ARFF,
and have the following keys:

- **description**: (OPTIONAL) a string with the description of the dataset.
- **relation**: (OBLIGATORY) a string with the name of the dataset.
- **attributes**: (OBLIGATORY) a list of attributes with the following
  template::

    (attribute_name, attribute_type)

  the attribute_name is a string, and attribute_type must be an string
  or a list of strings.
- **data**: (OBLIGATORY) a list of data instances. Each data instance must be
  a list with values, depending on the attributes.

The above keys must follow the case which were described, i.e., the keys are
case sensitive. The attribute type ``attribute_type`` must be one of these
strings (they are not case sensitive): ``NUMERIC``, ``INTEGER``, ``REAL`` or
``STRING``. For nominal attributes, the ``atribute_type`` must be a list of
strings.

In this format, the XOR dataset presented above can be represented as a python
object as::

    xor_dataset = {
        'description': 'XOR Dataset',
        'relation': 'XOR',
        'attributes': [
            ('input1', 'REAL'),
            ('input2', 'REAL'),
            ('y', 'REAL'),
        ],
        'data': [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ]
    }


Features
~~~~~~~~

This module provides several features, including:

- Read and write ARFF files using python built-in structures, such dictionaries
  and lists;
- Supports `scipy.sparse.coo <http://docs.scipy
  .org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix>`_
  and lists of dictionaries as used by SVMLight
- Supports the following attribute types: NUMERIC, REAL, INTEGER, STRING, and
  NOMINAL;
- Has an interface similar to other built-in modules such as ``json``, or
  ``zipfile``;
- Supports read and write the descriptions of files;
- Supports missing values and names with spaces;
- Supports unicode values and names;
- Fully compatible with Python 2.7+, Python 3.5+, pypy and pypy3;
- Under `MIT License <http://opensource.org/licenses/MIT>`_

'''
__author__ = 'Renato de Pontes Pereira, Matthias Feurer, Joel Nothman'
__author_email__ = ('renato.ppontes@gmail.com, '
                    'feurerm@informatik.uni-freiburg.de, '
                    'joel.nothman@gmail.com')
__version__ = '2.4.0'

import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple

# CONSTANTS ===================================================================
_SIMPLE_TYPES = ['NUMERIC', 'REAL', 'INTEGER', 'STRING']

_TK_DESCRIPTION = '%'
_TK_COMMENT     = '%'
_TK_RELATION    = '@RELATION'
_TK_ATTRIBUTE   = '@ATTRIBUTE'
_TK_DATA        = '@DATA'

_RE_RELATION     = re.compile(r'^([^\{\}%,\s]*|\".*\"|\'.*\')$', re.UNICODE)
_RE_ATTRIBUTE    = re.compile(r'^(\".*\"|\'.*\'|[^\{\}%,\s]*)\s+(.+)$', re.UNICODE)
_RE_QUOTE_CHARS = re.compile(r'["\'\\\s%,\000-\031]', re.UNICODE)
_RE_ESCAPE_CHARS = re.compile(r'(?=["\'\\%])|[\n\r\t\000-\031]')
_RE_SPARSE_LINE = re.compile(r'^\s*\{.*\}\s*$', re.UNICODE)
_RE_NONTRIVIAL_DATA = re.compile('["\'{}\\s]', re.UNICODE)

ArffDenseDataType = Iterator[List]
ArffSparseDataType = Tuple[List, ...]


if TYPE_CHECKING:
    # For Pyrefly, typing_extensions symbols are available within TYPE_CHECKING blocks.
    from typing_extensions import TypedDict

    class ArffContainerType(TypedDict):
        # [CLASS DOC]
        # Typing-only TypedDict (defined only under TYPE_CHECKING; a plain Dict alias at
        #   runtime) describing the decoded/encodable object: description, relation,
        #   attributes (list of (name, type) pairs) and data.
        description: str
        relation: str
        attributes: List
        data: Union[ArffDenseDataType, ArffSparseDataType]

else:
    ArffContainerType = Dict[str, Any]


def _build_re_values():
    # [FUNCTION DOC]
    # HELPER: Builds the two regexes used to split data lines. A value is a double- or
    #   single-quoted string (with backslash escapes) or a run of characters without
    #   commas, whitespace, quotes or braces. The dense regex, applied to "," + line,
    #   captures either a (possibly empty) value after each comma in group 1 or
    #   unparseable remaining text in group 2. The sparse regex captures (index, value)
    #   pairs after "{" or ",", and matches any other non-trivial text as an error
    #   (empty groups), except a closing "}" or an empty "{}".
    # Outputs: tuple (dense_regex, sparse_regex).
    quoted_re = r'''
                    "      # open quote followed by zero or more of:
                    (?:
                        (?<!\\)    # no additional backslash
                        (?:\\\\)*  # maybe escaped backslashes
                        \\"        # escaped quote
                    |
                        \\[^"]     # escaping a non-quote
                    |
                        [^"\\]     # non-quote char
                    )*
                    "      # close quote
                    '''
    # a value is surrounded by " or by ' or contains no quotables
    value_re = r'''(?:
        %s|          # a value may be surrounded by "
        %s|          # or by '
        [^,\s"'{}]+  # or may contain no characters requiring quoting
        )''' % (quoted_re,
                quoted_re.replace('"', "'"))

    # This captures (value, error) groups. Because empty values are allowed,
    # we cannot just look for empty values to handle syntax errors.
    # We presume the line has had ',' prepended...
    dense = re.compile(r'''(?x)
        ,                # may follow ','
        \s*
        ((?=,)|$|{value_re})  # empty or value
        |
        (\S.*)           # error
        '''.format(value_re=value_re))

    # This captures (key, value) groups and will have an empty key/value
    # in case of syntax errors.
    # It does not ensure that the line starts with '{' or ends with '}'.
    sparse = re.compile(r'''(?x)
        (?:^\s*\{|,)   # may follow ',', or '{' at line start
        \s*
        (\d+)          # attribute key
        \s+
        (%(value_re)s) # value
        |
        (?!}\s*$)      # not an error if it's }$
        (?!^\s*{\s*}\s*$)  # not an error if it's ^{}$
        \S.*           # error
        ''' % {'value_re': value_re})
    return dense, sparse



_RE_DENSE_VALUES, _RE_SPARSE_KEY_VALUES = _build_re_values()


_ESCAPE_SUB_MAP = {
    '\\\\': '\\',
    '\\"': '"',
    "\\'": "'",
    '\\t': '\t',
    '\\n': '\n',
    '\\r': '\r',
    '\\b': '\b',
    '\\f': '\f',
    '\\%': '%',
}
_UNESCAPE_SUB_MAP = {chr(i): '\\%03o' % i for i in range(32)}
_UNESCAPE_SUB_MAP.update({v: k for k, v in _ESCAPE_SUB_MAP.items()})
_UNESCAPE_SUB_MAP[''] = '\\'
_ESCAPE_SUB_MAP.update({'\\%d' % i: chr(i) for i in range(10)})


def _escape_sub_callback(match):
    # [FUNCTION DOC]
    # HELPER: re.sub callback that decodes one backslash escape inside a quoted value:
    #   two-character escapes via _ESCAPE_SUB_MAP (including \0-\9 as control chars),
    #   "\uXXXX" as a hex code point, otherwise "\NNN" as an octal code.
    # Exceptions: ValueError for an unknown two-character escape.
    s = match.group()
    if len(s) == 2:
        try:
            return _ESCAPE_SUB_MAP[s]
        except KeyError:
            raise ValueError('Unsupported escape sequence: %s' % s)
    if s[1] == 'u':
        return chr(int(s[2:], 16))
    else:
        return chr(int(s[1:], 8))


def _unquote(v):
    # [FUNCTION DOC]
    # HELPER: Converts a raw token to its value: quoted tokens have the quotes removed
    #   and escapes decoded; "?" or empty become None (missing); anything else is
    #   returned unchanged.
    if v[:1] in ('"', "'"):
        return re.sub(r'\\([0-9]{1,3}|u[0-9a-f]{4}|.)', _escape_sub_callback,
                      v[1:-1])
    elif v in ('?', ''):
        return None
    else:
        return v


def _parse_values(s):
    # [FUNCTION DOC]
    # HELPER: Splits one data line (or nominal declaration body) into values.
    # Behavior: if the line has no quotes, braces or whitespace, use the csv reader and
    #   map "?"/"" to None. Otherwise apply the dense regex; if no error group matched,
    #   return the unquoted values list. If it fails and the line looks like "{...}",
    #   parse as sparse into a dict {int index: unquoted value}. Otherwise locate the
    #   offending text for the error.
    # Outputs: list (dense) or dict (sparse).
    # Exceptions: BadLayout when the line cannot be parsed (with the offending fragment
    #   when found).
    '''(INTERNAL) Split a line into a list of values'''
    if not _RE_NONTRIVIAL_DATA.search(s):
        # Fast path for trivial cases (unfortunately we have to handle missing
        # values because of the empty string case :(.)
        return [None if s in ('?', '') else s
                for s in next(csv.reader([s]))]

    # _RE_DENSE_VALUES tokenizes despite quoting, whitespace, etc.
    values, errors = zip(*_RE_DENSE_VALUES.findall(',' + s))
    if not any(errors):
        return [_unquote(v) for v in values]
    if _RE_SPARSE_LINE.match(s):
        try:
            return {int(k): _unquote(v)
                    for k, v in _RE_SPARSE_KEY_VALUES.findall(s)}
        except ValueError:
            # an ARFF syntax error in sparse data
            for match in _RE_SPARSE_KEY_VALUES.finditer(s):
                if not match.group(1):
                    raise BadLayout('Error parsing %r' % match.group())
            raise BadLayout('Unknown parsing error')
    else:
        # an ARFF syntax error
        for match in _RE_DENSE_VALUES.finditer(s):
            if match.group(2):
                raise BadLayout('Error parsing %r' % match.group())
        raise BadLayout('Unknown parsing error')


DENSE = 0     # Constant value representing a dense matrix
COO = 1       # Constant value representing a sparse matrix in coordinate format
LOD = 2       # Constant value representing a sparse matrix in list of
              # dictionaries format
DENSE_GEN = 3 # Generator of dictionaries
LOD_GEN = 4   # Generator of dictionaries
_SUPPORTED_DATA_STRUCTURES = [DENSE, COO, LOD, DENSE_GEN, LOD_GEN]


# EXCEPTIONS ==================================================================
class ArffException(Exception):
    # [CLASS DOC]
    # Base class for ARFF errors. Subclasses provide message with a "%d" placeholder
    #   filled with the line number stored in line (initialized to -1, set by
    #   ArffDecoder.decode).
    message: Optional[str] = None

    def __init__(self):
        # [FUNCTION DOC]
        # Sets line to -1. Takes no message argument; subclasses set message.
        self.line = -1

    def __str__(self):
        # [FUNCTION DOC]
        # Returns message formatted with line.
        return self.message%self.line

class BadRelationFormat(ArffException):
    # [CLASS DOC]
    # Raised for an invalid @RELATION declaration.
    '''Error raised when the relation declaration is in an invalid format.'''
    message = 'Bad @RELATION format, at line %d.'

class BadAttributeFormat(ArffException):
    # [CLASS DOC]
    # Raised for a malformed @ATTRIBUTE declaration.
    '''Error raised when some attribute declaration is in an invalid format.'''
    message = 'Bad @ATTRIBUTE format, at line %d.'

class BadDataFormat(ArffException):
    # [CLASS DOC]
    # Raised when a data row has the wrong number of values or out-of-range sparse
    #   indices.
    '''Error raised when some data instance is in an invalid format.'''
    def __init__(self, value):
        # [FUNCTION DOC]
        # Builds the message embedding the offending row text after the line-number
        #   placeholder.
        # KNOWN ISSUE: a "%" inside value is not escaped, so __str__ can fail or
        #   misformat for rows containing "%" (BadLayout escapes it, this class does
        #   not).
        super().__init__()
        self.message = (
            'Bad @DATA instance format in line %d: ' +
            ('%s' % value)
        )

class BadAttributeType(ArffException):
    # [CLASS DOC]
    # Raised for an unknown attribute type or a malformed nominal value list.
    '''Error raised when some invalid type is provided into the attribute
    declaration.'''
    message = 'Bad @ATTRIBUTE type, at line %d.'

class BadAttributeName(ArffException):
    # [CLASS DOC]
    # Raised when an attribute name is declared twice.
    '''Error raised when an attribute name is provided twice the attribute
    declaration.'''

    def __init__(self, value, value2):
        # [FUNCTION DOC]
        # Builds the message with the duplicated name and the line of its first
        #   declaration (value2), leaving a placeholder for the current line. Same
        #   unescaped "%" caveat as BadDataFormat.
        super().__init__()
        self.message = (
            ('Bad @ATTRIBUTE name %s at line' % value) +
            ' %d, this name is already in use in line' +
            (' %d.' % value2)
        )

class BadNominalValue(ArffException):
    # [CLASS DOC]
    # Raised when a data value is not among the declared nominal values.
    '''Error raised when a value in used in some data instance but is not
    declared into it respective attribute declaration.'''

    def __init__(self, value):
        # [FUNCTION DOC]
        # Builds the message embedding the offending value.
        super().__init__()
        self.message = (
            ('Data value %s not found in nominal declaration, ' % value)
            + 'at line %d.'
        )

class BadNominalFormatting(ArffException):
    # [CLASS DOC]
    # Raised when a nominal value containing spaces is not quoted (not raised anywhere
    #   in this module).
    '''Error raised when a nominal value with space is not properly quoted.'''
    def __init__(self, value):
        # [FUNCTION DOC]
        # Builds the message embedding the offending value.
        super().__init__()
        self.message = (
            ('Nominal data value "%s" not properly quoted in line ' % value) +
            '%d.'
        )

class BadNumericalValue(ArffException):
    # [CLASS DOC]
    # Raised when a numeric attribute value cannot be converted to float.
    '''Error raised when and invalid numerical value is used in some data
    instance.'''
    message = 'Invalid numerical value, at line %d.'

class BadStringValue(ArffException):
    # [CLASS DOC]
    # Raised for an unquoted string with spaces (not raised anywhere in this module).
    '''Error raise when a string contains space but is not quoted.'''
    message = 'Invalid string value at line %d.'

class BadLayout(ArffException):
    # [CLASS DOC]
    # Raised when declarations appear in the wrong order, @DATA is missing, or a data
    #   line cannot be parsed.
    '''Error raised when the layout of the ARFF file has something wrong.'''
    message = 'Invalid layout of the ARFF file, at line %d.'

    def __init__(self, msg=''):
        # [FUNCTION DOC]
        # Appends msg (with "%" escaped as "%%") to the default message when given.
        super().__init__()
        if msg:
            self.message = BadLayout.message + ' ' + msg.replace('%', '%%')


class BadObject(ArffException):
    # [CLASS DOC]
    # Raised by the encoder when the Python object to serialize is invalid. Its string
    #   form does not include a line number.
    '''Error raised when the object representing the ARFF file has something
    wrong.'''
    def __init__(self, msg='Invalid object.'):
        # [FUNCTION DOC]
        # Stores msg. Does not call the base __init__, so no line attribute is set until
        #   decode assigns one.
        self.msg = msg

    def __str__(self):
        # [FUNCTION DOC]
        # Returns msg.
        return '%s' % self.msg

# =============================================================================

# INTERNAL ====================================================================
def _unescape_sub_callback(match):
    # [FUNCTION DOC]
    # HELPER: re.sub callback mapping a character (or the empty lookahead match before a
    #   quote/backslash/percent) to its escaped form via _UNESCAPE_SUB_MAP: control
    #   characters to octal "\NNN" or named escapes, and the empty match to a backslash.
    return _UNESCAPE_SUB_MAP[match.group()]


def encode_string(s):
    # [FUNCTION DOC]
    # Quotes a string for ARFF output: if it contains quotes, backslashes, whitespace,
    #   "%", "," or control characters, escape special characters and wrap it in single
    #   quotes; otherwise return it unchanged.
    if _RE_QUOTE_CHARS.search(s):
        return "'%s'" % _RE_ESCAPE_CHARS.sub(_unescape_sub_callback, s)
    return s


class EncodedNominalConversor:
    # [CLASS DOC]
    # Converter for nominal attributes when encode_nominal is True: maps each declared
    #   value to its index.
    def __init__(self, values):
        # [FUNCTION DOC]
        # Builds the value-to-index dict and also maps the integer 0 to index 0 (sparse
        #   rows supply 0 for absent entries).
        self.values = {v: i for i, v in enumerate(values)}
        self.values[0] = 0

    def __call__(self, value):
        # [FUNCTION DOC]
        # Returns the index of value; raises BadNominalValue if undeclared.
        try:
            return self.values[value]
        except KeyError:
            raise BadNominalValue(value)


class NominalConversor:
    # [CLASS DOC]
    # Converter for nominal attributes that returns the string value after validating it
    #   against the declaration.
    def __init__(self, values):
        # [FUNCTION DOC]
        # Stores the set of declared values and the first declared value as the
        #   replacement for sparse zeros.
        self.values = set(values)
        self.zero_value = values[0]

    def __call__(self, value):
        # [FUNCTION DOC]
        # Returns str(value) if declared; the integer 0 (an implicit sparse zero) maps
        #   to the first declared value; otherwise raises BadNominalValue.
        if value not in self.values:
            if value == 0:
                # Sparse decode
                # See issue #52: nominals should take their first value when
                # unspecified in a sparse matrix. Naturally, this is consistent
                # with EncodedNominalConversor.
                return self.zero_value
            raise BadNominalValue(value)
        return str(value)


class DenseGeneratorData:
    # [CLASS DOC]
    # HELPER: Data strategy yielding each decoded row as a list (lazy generator) and
    #   encoding rows of lists to CSV-style lines.
    '''Internal helper class to allow for different matrix types without
    making the code a huge collection of if statements.'''

    def decode_rows(self, stream, conversors):
        # [FUNCTION DOC]
        # Generator: parses each line with _parse_values; sparse dict lines are expanded
        #   to dense lists filling missing indices with 0 (BadDataFormat if an index is
        #   >= number of attributes); dense lines must have exactly one value per
        #   attribute (BadDataFormat otherwise); yields converted values.
        # Side effects: exceptions surface only during iteration.
        for row in stream:
            values = _parse_values(row)

            if isinstance(values, dict):
                if values and max(values) >= len(conversors):
                    raise BadDataFormat(row)
                # XXX: int 0 is used for implicit values, not '0'
                values = [values[i] if i in values else 0 for i in
                          range(len(conversors))]
            else:
                if len(values) != len(conversors):
                    raise BadDataFormat(row)

            yield self._decode_values(values, conversors)

    @staticmethod
    def _decode_values(values, conversors):
        # [FUNCTION DOC]
        # HELPER: Applies each attribute's conversor to the matching value, keeping None
        #   for missing values. Raises BadNumericalValue when a float conversion fails.
        # KNOWN ISSUE: any other ValueError (e.g. from int() or a non-float message) is
        #   swallowed and the unconverted raw values are returned silently.
        try:
            values = [None if value is None else conversor(value)
                      for conversor, value
                      in zip(conversors, values)]
        except ValueError as exc:
            if 'float: ' in str(exc):
                raise BadNumericalValue()
        return values

    def encode_data(self, data, attributes):
        # [FUNCTION DOC]
        # Generator yielding one comma-separated line per instance; None, "" and NaN
        #   become "?", other values are str()-ed and quoted via encode_string. Raises
        #   BadObject if an instance's length differs from the number of attributes.
        '''(INTERNAL) Encodes a line of data.

        Data instances follow the csv format, i.e, attribute values are
        delimited by commas. After converted from csv.

        :param data: a list of values.
        :param attributes: a list of attributes. Used to check if data is valid.
        :return: a string with the encoded data line.
        '''
        current_row = 0

        for inst in data:
            if len(inst) != len(attributes):
                raise BadObject(
                    'Instance %d has %d attributes, expected %d' %
                     (current_row, len(inst), len(attributes))
                )

            new_data = []
            for value in inst:
                if value is None or value == '' or value != value:
                    s = '?'
                else:
                    s = encode_string(str(value))
                new_data.append(s)

            current_row += 1
            yield ','.join(new_data)


class _DataListMixin:
    # [CLASS DOC]
    # HELPER: Mixin whose decode_rows materializes the parent's generator into a list.
    """Mixin to return a list from decode_rows instead of a generator"""
    def decode_rows(self, stream, conversors):
        # [FUNCTION DOC]
        # Returns list(super().decode_rows(stream, conversors)).
        return list(super().decode_rows(stream, conversors))


class Data(_DataListMixin, DenseGeneratorData):
    # [CLASS DOC]
    # Dense list-of-lists strategy (DENSE return type): DenseGeneratorData with list
    #   materialization.
    pass


class COOData:
    # [CLASS DOC]
    # Sparse coordinate strategy (COO return type).
    def decode_rows(self, stream, conversors):
        # [FUNCTION DOC]
        # Parses sparse lines into three parallel lists (data, rows, cols). For each
        #   line (row index = line position, including empty "{}" lines which are
        #   skipped), sort entries by column, convert each non-None value with its
        #   column's conversor, and append.
        # Outputs: tuple (data, rows, cols).
        # Exceptions: BadLayout for a dense line; BadNumericalValue for float conversion
        #   failures; BadDataFormat if a column index exceeds the attribute count; other
        #   ValueErrors propagate.
        data, rows, cols = [], [], []
        for i, row in enumerate(stream):
            values = _parse_values(row)
            if not isinstance(values, dict):
                raise BadLayout()
            if not values:
                continue
            row_cols, values = zip(*sorted(values.items()))
            try:
                values = [value if value is None else conversors[key](value)
                          for key, value in zip(row_cols, values)]
            except ValueError as exc:
                if 'float: ' in str(exc):
                    raise BadNumericalValue()
                raise
            except IndexError:
                # conversor out of range
                raise BadDataFormat(row)

            data.extend(values)
            rows.extend([i] * len(values))
            cols.extend(row_cols)

        return data, rows, cols

    def encode_data(self, data, attributes):
        # [FUNCTION DOC]
        # Generator encoding a scipy COO matrix as sparse lines "{ col value,... }", one
        #   per row up to the last row containing a stored entry (empty rows in between
        #   produce "{  }" lines). Missing values (None, "", NaN) become "?".
        # Exceptions: ValueError if row indices are not sorted; BadObject if a column
        #   index is >= number of attributes.
        # KNOWN ISSUE: trailing all-zero rows after the last stored entry are not
        #   emitted, so the row count is not preserved.
        num_attributes = len(attributes)
        new_data = []
        current_row = 0

        row = data.row
        col = data.col
        data = data.data

        # Check if the rows are sorted
        if not all(row[i] <= row[i + 1] for i in range(len(row) - 1)):
            raise ValueError("liac-arff can only output COO matrices with "
                             "sorted rows.")

        for v, col, row in zip(data, col, row):
            if row > current_row:
                # Add empty rows if necessary
                while current_row < row:
                    yield " ".join(["{", ','.join(new_data), "}"])
                    new_data = []
                    current_row += 1

            if col >= num_attributes:
                raise BadObject(
                    'Instance %d has at least %d attributes, expected %d' %
                    (current_row, col + 1, num_attributes)
                )

            if v is None or v == '' or v != v:
                s = '?'
            else:
                s = encode_string(str(v))
            new_data.append("%d %s" % (col, s))

        yield " ".join(["{", ','.join(new_data), "}"])

class LODGeneratorData:
    # [CLASS DOC]
    # Lazy list-of-dicts strategy (LOD_GEN return type).
    def decode_rows(self, stream, conversors):
        # [FUNCTION DOC]
        # Generator yielding for each sparse line a dict {column index: converted value}
        #   (None kept). Raises BadLayout for dense lines, BadNumericalValue for float
        #   failures, BadDataFormat for out-of-range indices.
        for row in stream:
            values = _parse_values(row)

            if not isinstance(values, dict):
                raise BadLayout()
            try:
                yield {key: None if value is None else conversors[key](value)
                       for key, value in values.items()}
            except ValueError as exc:
                if 'float: ' in str(exc):
                    raise BadNumericalValue()
                raise
            except IndexError:
                # conversor out of range
                raise BadDataFormat(row)

    def encode_data(self, data, attributes):
        # [FUNCTION DOC]
        # Generator encoding each dict row as "{ col value,... }" with columns sorted;
        #   missing values become "?". Raises BadObject if the largest key is >= number
        #   of attributes.
        current_row = 0

        num_attributes = len(attributes)
        for row in data:
            new_data = []

            if len(row) > 0 and max(row) >= num_attributes:
                raise BadObject(
                    'Instance %d has %d attributes, expected %d' %
                    (current_row, max(row) + 1, num_attributes)
                )

            for col in sorted(row):
                v = row[col]
                if v is None or v == '' or v != v:
                    s = '?'
                else:
                    s = encode_string(str(v))
                new_data.append("%d %s" % (col, s))

            current_row += 1
            yield " ".join(["{", ','.join(new_data), "}"])

class LODData(_DataListMixin, LODGeneratorData):
    # [CLASS DOC]
    # Eager list-of-dicts strategy (LOD return type).
    pass


def _get_data_object_for_decoding(matrix_type):
    # [FUNCTION DOC]
    # HELPER: Returns a new strategy object for the given return-type constant (DENSE to
    #   Data, COO to COOData, LOD to LODData, DENSE_GEN to DenseGeneratorData, LOD_GEN
    #   to LODGeneratorData); ValueError otherwise.
    if matrix_type == DENSE:
        return Data()
    elif matrix_type == COO:
        return COOData()
    elif matrix_type == LOD:
        return LODData()
    elif matrix_type == DENSE_GEN:
        return DenseGeneratorData()
    elif matrix_type == LOD_GEN:
        return LODGeneratorData()
    else:
        raise ValueError("Matrix type %s not supported." % str(matrix_type))

def _get_data_object_for_encoding(matrix):
    # [FUNCTION DOC]
    # HELPER: Picks the encoding strategy from the data: objects with a format attribute
    #   must be COO sparse matrices (ValueError otherwise); a sequence whose first
    #   element is a dict uses LODData; anything else Data.
    # KNOWN ISSUE: empty data (an empty list) raises IndexError when inspecting the
    #   first element.
    # Probably a scipy.sparse
    if hasattr(matrix, 'format'):
        if matrix.format == 'coo':
            return COOData()
        else:
            raise ValueError('Cannot guess matrix format!')
    elif isinstance(matrix[0], dict):
        return LODData()
    else:
        return Data()

# =============================================================================

# ADVANCED INTERFACE ==========================================================
class ArffDecoder:
    # [CLASS DOC]
    # Stateful ARFF parser tracking the current line number and the per-attribute
    #   converters.
    '''An ARFF decoder.'''

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes an empty converter list and the line counter.
        '''Constructor.'''
        self._conversors = []
        self._current_line = 0

    def _decode_comment(self, s):
        # [FUNCTION DOC]
        # HELPER: Removes the leading "%" and at most one following space from a comment
        #   line.
        '''(INTERNAL) Decodes a comment line.

        Comments are single line strings starting, obligatorily, with the ``%``
        character, and can have any symbol, including whitespaces or special
        characters.

        This method must receive a normalized string, i.e., a string without
        padding, including the "\r\n" characters.

        :param s: a normalized string.
        :return: a string with the decoded comment.
        '''
        res = re.sub(r'^\%( )?', '', s)
        return res

    def _decode_relation(self, s):
        # [FUNCTION DOC]
        # HELPER: Takes the text after the first space of a @RELATION line, validates it
        #   with _RE_RELATION (BadRelationFormat if invalid) and returns it with
        #   surrounding quotes stripped.
        # KNOWN ISSUE: a line consisting only of "@RELATION" (no space) raises
        #   ValueError from unpacking instead of BadRelationFormat.
        '''(INTERNAL) Decodes a relation line.

        The relation declaration is a line with the format ``@RELATION
        <relation-name>``, where ``relation-name`` is a string. The string must
        start with alphabetic character and must be quoted if the name includes
        spaces, otherwise this method will raise a `BadRelationFormat` exception.

        This method must receive a normalized string, i.e., a string without
        padding, including the "\r\n" characters.

        :param s: a normalized string.
        :return: a string with the decoded relation name.
        '''
        _, v = s.split(' ', 1)
        v = v.strip()

        if not _RE_RELATION.match(v):
            raise BadRelationFormat()

        res = str(v.strip('"\''))
        return res

    def _decode_attribute(self, s):
        # [FUNCTION DOC]
        # HELPER: Parses an @ATTRIBUTE line into (name, type). The name is unquoted; a
        #   type in braces is parsed with _parse_values into the list of nominal values
        #   (BadAttributeType on parse failure or sparse syntax); otherwise the
        #   uppercased type must be NUMERIC, REAL, INTEGER or STRING (BadAttributeType
        #   otherwise). BadAttributeFormat if the line does not match _RE_ATTRIBUTE.
        #   Dates are not supported.
        '''(INTERNAL) Decodes an attribute line.

        The attribute is the most complex declaration in an arff file. All
        attributes must follow the template::

             @attribute <attribute-name> <datatype>

        where ``attribute-name`` is a string, quoted if the name contains any
        whitespace, and ``datatype`` can be:

        - Numerical attributes as ``NUMERIC``, ``INTEGER`` or ``REAL``.
        - Strings as ``STRING``.
        - Dates (NOT IMPLEMENTED).
        - Nominal attributes with format:

            {<nominal-name1>, <nominal-name2>, <nominal-name3>, ...}

        The nominal names follow the rules for the attribute names, i.e., they
        must be quoted if the name contains whitespaces.

        This method must receive a normalized string, i.e., a string without
        padding, including the "\r\n" characters.

        :param s: a normalized string.
        :return: a tuple (ATTRIBUTE_NAME, TYPE_OR_VALUES).
        '''
        _, v = s.split(' ', 1)
        v = v.strip()

        # Verify the general structure of declaration
        m = _RE_ATTRIBUTE.match(v)
        if not m:
            raise BadAttributeFormat()

        # Extracts the raw name and type
        name, type_ = m.groups()

        # Extracts the final name
        name = str(name.strip('"\''))

        # Extracts the final type
        if type_[:1] == "{" and type_[-1:] == "}":
            try:
                type_ = _parse_values(type_.strip('{} '))
            except Exception:
                raise BadAttributeType()
            if isinstance(type_, dict):
                raise BadAttributeType()

        else:
            # If not nominal, verify the type name
            type_ = str(type_).upper()
            if type_ not in ['NUMERIC', 'REAL', 'INTEGER', 'STRING']:
                raise BadAttributeType()

        return (name, type_)

    def _decode(self, s, encode_nominal=False, matrix_type=DENSE):
        # [FUNCTION DOC]
        # HELPER: Core decoder (its docstring wrongly says "encode").
        # Behavior: reset the line counter; split a string input into lines (any other
        #   iterable of lines is used as-is); scan header lines, skipping blanks:
        #   leading "%" lines form the description; @RELATION must come in the
        #   description state; @ATTRIBUTE must follow the relation or another attribute,
        #   duplicates raise BadAttributeName; each attribute appends a converter
        #   (nominal: EncodedNominalConversor or NominalConversor; STRING str, INTEGER
        #   int-of-float, NUMERIC/REAL float); @DATA must follow attributes and ends the
        #   header; other "%" lines are ignored. If the input ends without @DATA, raise
        #   BadLayout. Remaining non-empty, non-comment lines are fed lazily to the data
        #   strategy's decode_rows.
        # Outputs: dict with description (trailing newline removed), relation,
        #   attributes, data.
        # Side effects: appends to self._conversors and updates self._current_line.
        # KNOWN ISSUE: self._conversors is never cleared, so reusing a decoder for a
        #   second file keeps the previous file's converters in front. Unrecognized
        #   header lines (not starting with "%" or "@") are silently ignored.
        '''Do the job the ``encode``.'''

        # Make sure this method is idempotent
        self._current_line = 0

        # If string, convert to a list of lines
        if isinstance(s, str):
            s = s.strip('\r\n ').replace('\r\n', '\n').split('\n')

        # Create the return object
        obj: ArffContainerType = {
            'description': '',
            'relation': '',
            'attributes': [],
            'data': []
        }
        attribute_names = {}

        # Create the data helper object
        data = _get_data_object_for_decoding(matrix_type)

        # Read all lines
        STATE = _TK_DESCRIPTION
        s = iter(s)
        for row in s:
            self._current_line += 1
            # Ignore empty lines
            row = row.strip(' \r\n')
            if not row: continue

            u_row = row.upper()

            # DESCRIPTION -----------------------------------------------------
            if u_row.startswith(_TK_DESCRIPTION) and STATE == _TK_DESCRIPTION:
                obj['description'] += self._decode_comment(row) + '\n'
            # -----------------------------------------------------------------

            # RELATION --------------------------------------------------------
            elif u_row.startswith(_TK_RELATION):
                if STATE != _TK_DESCRIPTION:
                    raise BadLayout()

                STATE = _TK_RELATION
                obj['relation'] = self._decode_relation(row)
            # -----------------------------------------------------------------

            # ATTRIBUTE -------------------------------------------------------
            elif u_row.startswith(_TK_ATTRIBUTE):
                if STATE != _TK_RELATION and STATE != _TK_ATTRIBUTE:
                    raise BadLayout()

                STATE = _TK_ATTRIBUTE

                attr = self._decode_attribute(row)
                if attr[0] in attribute_names:
                    raise BadAttributeName(attr[0], attribute_names[attr[0]])
                else:
                    attribute_names[attr[0]] = self._current_line
                obj['attributes'].append(attr)

                if isinstance(attr[1], (list, tuple)):
                    if encode_nominal:
                        conversor = EncodedNominalConversor(attr[1])
                    else:
                        conversor = NominalConversor(attr[1])
                else:
                    CONVERSOR_MAP = {'STRING': str,
                                     'INTEGER': lambda x: int(float(x)),
                                     'NUMERIC': float,
                                     'REAL': float}
                    conversor = CONVERSOR_MAP[attr[1]]

                self._conversors.append(conversor)
            # -----------------------------------------------------------------

            # DATA ------------------------------------------------------------
            elif u_row.startswith(_TK_DATA):
                if STATE != _TK_ATTRIBUTE:
                    raise BadLayout()

                break
            # -----------------------------------------------------------------

            # COMMENT ---------------------------------------------------------
            elif u_row.startswith(_TK_COMMENT):
                pass
            # -----------------------------------------------------------------
        else:
            # Never found @DATA
            raise BadLayout()

        def stream():
            # [FUNCTION DOC]
            # HELPER: Generator over the remaining input lines that increments the line
            #   counter and yields stripped lines that are non-empty and not comments.
            for row in s:
                self._current_line += 1
                row = row.strip()
                # Ignore empty lines and comment lines.
                if row and not row.startswith(_TK_COMMENT):
                    yield row

        # Alter the data object
        obj['data'] = data.decode_rows(stream(), self._conversors)
        if obj['description'].endswith('\n'):
            obj['description'] = obj['description'][:-1]

        return obj

    def decode(self, s, encode_nominal=False, return_type=DENSE):
        # [FUNCTION DOC]
        # Public entry point: calls _decode and, on ArffException, sets the exception's
        #   line to the current line before re-raising.
        # Inputs: s - string or iterable of lines (e.g. file object); encode_nominal -
        #   bool; return_type - one of the type constants.
        # KNOWN ISSUE: for the generator return types (DENSE_GEN, LOD_GEN) errors raised
        #   while iterating the data escape this handler, so their line stays -1.
        '''Returns the Python representation of a given ARFF file.

        When a file object is passed as an argument, this method reads lines
        iteratively, avoiding to load unnecessary information to the memory.

        :param s: a string or file object with the ARFF file.
        :param encode_nominal: boolean, if True perform a label encoding
            while reading the .arff file.
        :param return_type: determines the data structure used to store the
            dataset. Can be one of `arff.DENSE`, `arff.COO`, `arff.LOD`,
            `arff.DENSE_GEN` or `arff.LOD_GEN`.
            Consult the sections on `working with sparse data`_ and `loading
            progressively`_.
        '''
        try:
            return self._decode(s, encode_nominal=encode_nominal,
                                matrix_type=return_type)
        except ArffException as e:
            e.line = self._current_line
            raise e


class ArffEncoder:
    # [CLASS DOC]
    # Serializer from the ARFF dict representation to text lines.
    '''An ARFF encoder.'''

    def _encode_comment(self, s=''):
        # [FUNCTION DOC]
        # HELPER: Returns "% " followed by s, or just "%" for an empty string.
        '''(INTERNAL) Encodes a comment line.

        Comments are single line strings starting, obligatorily, with the ``%``
        character, and can have any symbol, including whitespaces or special
        characters.

        If ``s`` is None, this method will simply return an empty comment.

        :param s: (OPTIONAL) string.
        :return: a string with the encoded comment line.
        '''
        if s:
            return '%s %s'%(_TK_COMMENT, s)
        else:
            return '%s' % _TK_COMMENT

    def _encode_relation(self, name):
        # [FUNCTION DOC]
        # HELPER: Returns "@RELATION name", double-quoting the name when it contains a
        #   space, "%", braces or a comma. (Docstring says "Decodes".) KNOWN ISSUE:
        #   embedded double quotes are not escaped.
        '''(INTERNAL) Decodes a relation line.

        The relation declaration is a line with the format ``@RELATION
        <relation-name>``, where ``relation-name`` is a string.

        :param name: a string.
        :return: a string with the encoded relation declaration.
        '''
        for char in ' %{},':
            if char in name:
                name = '"%s"'%name
                break

        return '%s %s'%(_TK_RELATION, name)

    def _encode_attribute(self, name, type_):
        # [FUNCTION DOC]
        # HELPER: Returns "@ATTRIBUTE name type", quoting the name like
        #   _encode_relation; a list/tuple type is rendered as "{v1, v2, ...}" with each
        #   value passed through encode_string.
        '''(INTERNAL) Encodes an attribute line.

        The attribute follow the template::

             @attribute <attribute-name> <datatype>

        where ``attribute-name`` is a string, and ``datatype`` can be:

        - Numerical attributes as ``NUMERIC``, ``INTEGER`` or ``REAL``.
        - Strings as ``STRING``.
        - Dates (NOT IMPLEMENTED).
        - Nominal attributes with format:

            {<nominal-name1>, <nominal-name2>, <nominal-name3>, ...}

        This method must receive a the name of the attribute and its type, if
        the attribute type is nominal, ``type`` must be a list of values.

        :param name: a string.
        :param type_: a string or a list of string.
        :return: a string with the encoded attribute declaration.
        '''
        for char in ' %{},':
            if char in name:
                name = '"%s"'%name
                break

        if isinstance(type_, (tuple, list)):
            type_tmp = ['%s' % encode_string(type_k) for type_k in type_]
            type_ = '{%s}'%(', '.join(type_tmp))

        return '%s %s %s'%(_TK_ATTRIBUTE, name, type_)

    def encode(self, obj):
        # [FUNCTION DOC]
        # Returns all lines from iter_encode joined with newlines.
        '''Encodes a given object to an ARFF file.

        :param obj: the object containing the ARFF information.
        :return: the ARFF file as an string.
        '''
        data = [row for row in self.iter_encode(obj)]

        return '\n'.join(data)

    def iter_encode(self, obj):
        # [FUNCTION DOC]
        # Generator producing the ARFF lines: description lines as comments, the
        #   relation (BadObject if missing), a blank line, attribute declarations
        #   (BadObject if attributes are missing, malformed, of an unknown type, or
        #   duplicated), a blank line, "@DATA", the encoded data rows when data is
        #   present, and a final empty line.
        '''The iterative version of `arff.ArffEncoder.encode`.

        This encodes iteratively a given object and return, one-by-one, the
        lines of the ARFF file.

        :param obj: the object containing the ARFF information.
        :return: (yields) the ARFF file as strings.
        '''
        # DESCRIPTION
        if obj.get('description', None):
            for row in obj['description'].split('\n'):
                yield self._encode_comment(row)

        # RELATION
        if not obj.get('relation'):
            raise BadObject('Relation name not found or with invalid value.')

        yield self._encode_relation(obj['relation'])
        yield ''

        # ATTRIBUTES
        if not obj.get('attributes'):
            raise BadObject('Attributes not found.')

        attribute_names = set()
        for attr in obj['attributes']:
            # Verify for bad object format
            if not isinstance(attr, (tuple, list)) or \
               len(attr) != 2 or \
               not isinstance(attr[0], str):
                raise BadObject('Invalid attribute declaration "%s"'%str(attr))

            if isinstance(attr[1], str):
                # Verify for invalid types
                if attr[1] not in _SIMPLE_TYPES:
                    raise BadObject('Invalid attribute type "%s"'%str(attr))

            # Verify for bad object format
            elif not isinstance(attr[1], (tuple, list)):
                raise BadObject('Invalid attribute type "%s"'%str(attr))

            # Verify attribute name is not used twice
            if attr[0] in attribute_names:
                raise BadObject('Trying to use attribute name "%s" for the '
                                'second time.' % str(attr[0]))
            else:
                attribute_names.add(attr[0])

            yield self._encode_attribute(attr[0], attr[1])
        yield ''
        attributes = obj['attributes']

        # DATA
        yield _TK_DATA
        if 'data' in obj:
            data = _get_data_object_for_encoding(obj.get('data'))
            yield from data.encode_data(obj.get('data'), attributes)

        yield ''

# =============================================================================

# BASIC INTERFACE =============================================================
def load(fp, encode_nominal=False, return_type=DENSE):
    # [FUNCTION DOC]
    # Decodes an ARFF document from a file-like object (iterated line by line) with a
    #   new ArffDecoder; returns the decoded dict.
    '''Load a file-like object containing the ARFF document and convert it into
    a Python object.

    :param fp: a file-like object.
    :param encode_nominal: boolean, if True perform a label encoding
        while reading the .arff file.
    :param return_type: determines the data structure used to store the
        dataset. Can be one of `arff.DENSE`, `arff.COO`, `arff.LOD`,
        `arff.DENSE_GEN` or `arff.LOD_GEN`.
        Consult the sections on `working with sparse data`_ and `loading
        progressively`_.
    :return: a dictionary.
     '''
    decoder = ArffDecoder()
    return decoder.decode(fp, encode_nominal=encode_nominal,
                          return_type=return_type)

def loads(s, encode_nominal=False, return_type=DENSE):
    # [FUNCTION DOC]
    # Decodes an ARFF document from a string with a new ArffDecoder; returns the decoded
    #   dict.
    '''Convert a string instance containing the ARFF document into a Python
    object.

    :param s: a string object.
    :param encode_nominal: boolean, if True perform a label encoding
        while reading the .arff file.
    :param return_type: determines the data structure used to store the
        dataset. Can be one of `arff.DENSE`, `arff.COO`, `arff.LOD`,
        `arff.DENSE_GEN` or `arff.LOD_GEN`.
        Consult the sections on `working with sparse data`_ and `loading
        progressively`_.
    :return: a dictionary.
    '''
    decoder = ArffDecoder()
    return decoder.decode(s, encode_nominal=encode_nominal,
                          return_type=return_type)

def dump(obj, fp):
    # [FUNCTION DOC]
    # Writes the encoded ARFF lines to fp separated by newlines (no trailing newline
    #   after the last line) and returns fp.
    # Exceptions: BadObject/ValueError from encoding; partial output may already be
    #   written when an error occurs.
    '''Serialize an object representing the ARFF document to a given file-like
    object.

    :param obj: a dictionary.
    :param fp: a file-like object.
    '''
    encoder = ArffEncoder()
    generator = encoder.iter_encode(obj)

    last_row = next(generator)
    for row in generator:
        fp.write(last_row + '\n')
        last_row = row
    fp.write(last_row)

    return fp

def dumps(obj):
    # [FUNCTION DOC]
    # Returns the ARFF document as a string using ArffEncoder.encode.
    '''Serialize an object representing the ARFF document, returning a string.

    :param obj: a dictionary.
    :return: a string with the ARFF document.
    '''
    encoder = ArffEncoder()
    return encoder.encode(obj)
# =============================================================================
