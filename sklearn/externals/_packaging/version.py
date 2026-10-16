# [MODULE DOC]
# Vendored copy of pypa/packaging's version module (PEP 440 version parsing and
#   ordering), used by scikit-learn's fixes/utility code to compare dependency versions
#   without requiring the packaging distribution.
# Defines parse, Version, LegacyVersion (deprecated fallback), InvalidVersion,
#   VERSION_PATTERN, the comparison-key builders and type aliases for the keys. The
#   private namedtuple _Version stores the parsed fields (epoch, release, dev, pre,
#   post, local).
"""Vendored from
https://github.com/pypa/packaging/blob/main/packaging/version.py
"""
# Copyright (c) Donald Stufft and individual contributors.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.

#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import collections
import itertools
import re
import warnings
from typing import Callable, Iterator, List, Optional, SupportsInt, Tuple, Union

from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType

__all__ = ["parse", "Version", "LegacyVersion", "InvalidVersion", "VERSION_PATTERN"]

InfiniteTypes = Union[InfinityType, NegativeInfinityType]
PrePostDevType = Union[InfiniteTypes, Tuple[str, int]]
SubLocalType = Union[InfiniteTypes, int, str]
LocalType = Union[
    NegativeInfinityType,
    Tuple[
        Union[
            SubLocalType,
            Tuple[SubLocalType, str],
            Tuple[NegativeInfinityType, SubLocalType],
        ],
        ...,
    ],
]
CmpKey = Tuple[
    int, Tuple[int, ...], PrePostDevType, PrePostDevType, PrePostDevType, LocalType
]
LegacyCmpKey = Tuple[int, Tuple[str, ...]]
VersionComparisonMethod = Callable[
    [Union[CmpKey, LegacyCmpKey], Union[CmpKey, LegacyCmpKey]], bool
]

_Version = collections.namedtuple(
    "_Version", ["epoch", "release", "dev", "pre", "post", "local"]
)


def parse(version: str) -> Union["LegacyVersion", "Version"]:
    # [FUNCTION DOC]
    # Parses a version string, returning a Version when it is PEP 440 compliant and
    #   otherwise falling back to a LegacyVersion.
    # Inputs: version - string. Outputs: Version or LegacyVersion instance.
    # Exceptions: none (InvalidVersion is caught). Side effects: the LegacyVersion
    #   fallback emits a DeprecationWarning.
    """Parse the given version from a string to an appropriate class.

    Parameters
    ----------
    version : str
        Version in a string format, eg. "0.9.1" or "1.2.dev0".

    Returns
    -------
    version : :class:`Version` object or a :class:`LegacyVersion` object
        Returned class depends on the given version: if is a valid
        PEP 440 version or a legacy version.
    """
    try:
        return Version(version)
    except InvalidVersion:
        return LegacyVersion(version)


class InvalidVersion(ValueError):
    # [CLASS DOC]
    # Exception (subclass of ValueError) raised by Version when a string does not match
    #   the PEP 440 pattern.
    """
    An invalid version was found, users should refer to PEP 440.
    """


class _BaseVersion:
    # [CLASS DOC]
    # HELPER: Base class implementing hashing and rich comparisons for version objects
    #   by delegating to the precomputed comparison key stored in the _key attribute by
    #   subclasses.
    _key: Union[CmpKey, LegacyCmpKey]

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Hash of the comparison key, so equal versions hash equally.
        return hash(self._key)

    # Please keep the duplicated `isinstance` check
    # in the six comparisons hereunder
    # unless you find a way to avoid adding overhead function calls.
    def __lt__(self, other: "_BaseVersion") -> bool:
        # [FUNCTION DOC]
        # Returns whether this version's _key is less than other's _key; returns
        #   NotImplemented when other is not a _BaseVersion so Python can try the
        #   reflected operation.
        # KNOWN ISSUE: comparing a Version with a LegacyVersion compares keys of
        #   different shapes; it works only because the leading epoch differs (-1 vs >=
        #   0).
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key < other._key

    def __le__(self, other: "_BaseVersion") -> bool:
        # [FUNCTION DOC]
        # Returns whether this version's _key is less than or equal to other's _key;
        #   returns NotImplemented when other is not a _BaseVersion so Python can try
        #   the reflected operation.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key <= other._key

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Returns whether this version's _key is equal to other's _key; returns
        #   NotImplemented when other is not a _BaseVersion so Python can try the
        #   reflected operation.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key == other._key

    def __ge__(self, other: "_BaseVersion") -> bool:
        # [FUNCTION DOC]
        # Returns whether this version's _key is greater than or equal to other's _key;
        #   returns NotImplemented when other is not a _BaseVersion so Python can try
        #   the reflected operation.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key >= other._key

    def __gt__(self, other: "_BaseVersion") -> bool:
        # [FUNCTION DOC]
        # Returns whether this version's _key is greater than other's _key; returns
        #   NotImplemented when other is not a _BaseVersion so Python can try the
        #   reflected operation.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key > other._key

    def __ne__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Returns whether this version's _key is not equal to other's _key; returns
        #   NotImplemented when other is not a _BaseVersion so Python can try the
        #   reflected operation.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key != other._key


class LegacyVersion(_BaseVersion):
    # [CLASS DOC]
    # Deprecated fallback representation for strings that are not valid PEP 440
    #   versions, ordered with the setuptools pre-PEP 440 scheme and always before every
    #   PEP 440 Version (epoch -1). Exposes the same attribute API as Version with
    #   constant/empty values.
    def __init__(self, version: str) -> None:
        # [FUNCTION DOC]
        # Stores the string form of version and computes _key with _legacy_cmpkey.
        # Side effects: emits a DeprecationWarning on every instantiation.
        self._version = str(version)
        self._key = _legacy_cmpkey(self._version)

        warnings.warn(
            "Creating a LegacyVersion has been deprecated and will be "
            "removed in the next major release",
            DeprecationWarning,
        )

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # Returns the original version string.
        return self._version

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns "<LegacyVersion('...')>" with the original string.
        return f"<LegacyVersion('{self}')>"

    @property
    def public(self) -> str:
        # [FUNCTION DOC]
        # Property: the original version string (no local segment concept).
        return self._version

    @property
    def base_version(self) -> str:
        # [FUNCTION DOC]
        # Property: the original version string.
        return self._version

    @property
    def epoch(self) -> int:
        # [FUNCTION DOC]
        # Property: always -1 so legacy versions sort before PEP 440 versions.
        return -1

    @property
    def release(self) -> None:
        # [FUNCTION DOC]
        # Property: always None.
        return None

    @property
    def pre(self) -> None:
        # [FUNCTION DOC]
        # Property: always None.
        return None

    @property
    def post(self) -> None:
        # [FUNCTION DOC]
        # Property: always None.
        return None

    @property
    def dev(self) -> None:
        # [FUNCTION DOC]
        # Property: always None.
        return None

    @property
    def local(self) -> None:
        # [FUNCTION DOC]
        # Property: always None.
        return None

    @property
    def is_prerelease(self) -> bool:
        # [FUNCTION DOC]
        # Property: always False.
        return False

    @property
    def is_postrelease(self) -> bool:
        # [FUNCTION DOC]
        # Property: always False.
        return False

    @property
    def is_devrelease(self) -> bool:
        # [FUNCTION DOC]
        # Property: always False.
        return False


_legacy_version_component_re = re.compile(r"(\d+ | [a-z]+ | \.| -)", re.VERBOSE)

_legacy_version_replacement_map = {
    "pre": "c",
    "preview": "c",
    "-": "final-",
    "rc": "c",
    "dev": "@",
}


def _parse_version_parts(s: str) -> Iterator[str]:
    # [FUNCTION DOC]
    # HELPER: Tokenizes a legacy version string for _legacy_cmpkey.
    # Behavior: splits on runs of digits, runs of letters, dots and dashes (keeping
    #   them); maps "pre"/"preview"/"rc" to "c", "-" to "final-" and "dev" to "@"; skips
    #   empty parts and dots; yields numeric parts zero-padded to 8 characters and other
    #   parts prefixed with "*"; finally yields "*final".
    # Inputs: s - lowercased version string. Outputs: generator of strings.
    for part in _legacy_version_component_re.split(s):
        part = _legacy_version_replacement_map.get(part, part)

        if not part or part == ".":
            continue

        if part[:1] in "0123456789":
            # pad for numeric comparison
            yield part.zfill(8)
        else:
            yield "*" + part

    # ensure that alpha/beta/candidate are before final
    yield "*final"


def _legacy_cmpkey(version: str) -> LegacyCmpKey:
    # [FUNCTION DOC]
    # HELPER: Builds the comparison key of a LegacyVersion following old
    #   pkg_resources.parse_version.
    # Behavior: epoch fixed to -1; iterate the parts of the lowercased string; before
    #   appending a "*" part, if it sorts before "*final" (a pre-release tag) drop
    #   trailing "*final-" parts, and in all cases drop trailing "00000000" numeric
    #   parts; append the part.
    # Inputs: version - string. Outputs: tuple (-1, tuple of part strings).

    # We hardcode an epoch of -1 here. A PEP 440 version can only have a epoch
    # greater than or equal to 0. This will effectively put the LegacyVersion,
    # which uses the defacto standard originally implemented by setuptools,
    # as before all PEP 440 versions.
    epoch = -1

    # This scheme is taken from pkg_resources.parse_version setuptools prior to
    # it's adoption of the packaging library.
    parts: List[str] = []
    for part in _parse_version_parts(version.lower()):
        if part.startswith("*"):
            # remove "-" before a prerelease tag
            if part < "*final":
                while parts and parts[-1] == "*final-":
                    parts.pop()

            # remove trailing zeros from each series of numeric parts
            while parts and parts[-1] == "00000000":
                parts.pop()

        parts.append(part)

    return epoch, tuple(parts)


# Deliberately not anchored to the start and end of the string, to make it
# easier for 3rd party code to reuse
VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""


class Version(_BaseVersion):
    # [CLASS DOC]
    # PEP 440 version object. Parses the string with VERSION_PATTERN (anchored, case-
    #   insensitive, surrounding whitespace allowed), stores normalized fields in
    #   _version and a sort key in _key; offers normalized str() and accessors for every
    #   segment.

    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)

    def __init__(self, version: str) -> None:
        # [FUNCTION DOC]
        # Parses version: raise InvalidVersion if the regex does not match; epoch
        #   defaults to 0; release is a tuple of ints; pre/post/dev are normalized via
        #   _parse_letter_version (post may come from the implicit "-N" form); local is
        #   parsed with _parse_local_version; then compute _key with _cmpkey.
        # Inputs: version - string. Exceptions: InvalidVersion. Side effects: sets
        #   _version and _key.

        # Validate the version and parse it into pieces
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")

        # Store the parsed out pieces of the version
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,
            release=tuple(int(i) for i in match.group("release").split(".")),
            pre=_parse_letter_version(match.group("pre_l"), match.group("pre_n")),
            post=_parse_letter_version(
                match.group("post_l"), match.group("post_n1") or match.group("post_n2")
            ),
            dev=_parse_letter_version(match.group("dev_l"), match.group("dev_n")),
            local=_parse_local_version(match.group("local")),
        )

        # Generate a key which will be used for sorting
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns "<Version('...')>" with the normalized string.
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # Builds the normalized PEP 440 string: "E!" if epoch is non-zero, dot-joined
        #   release, concatenated pre letter and number, ".postN", ".devN", and "+local"
        #   when present.
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self.release))

        # Pre-release
        if self.pre is not None:
            parts.append("".join(str(x) for x in self.pre))

        # Post-release
        if self.post is not None:
            parts.append(f".post{self.post}")

        # Development release
        if self.dev is not None:
            parts.append(f".dev{self.dev}")

        # Local version segment
        if self.local is not None:
            parts.append(f"+{self.local}")

        return "".join(parts)

    @property
    def epoch(self) -> int:
        # [FUNCTION DOC]
        # Property: integer epoch (0 when absent).
        _epoch: int = self._version.epoch
        return _epoch

    @property
    def release(self) -> Tuple[int, ...]:
        # [FUNCTION DOC]
        # Property: tuple of ints of the release segment (trailing zeros kept).
        _release: Tuple[int, ...] = self._version.release
        return _release

    @property
    def pre(self) -> Optional[Tuple[str, int]]:
        # [FUNCTION DOC]
        # Property: (letter, number) with letter in a/b/rc, or None.
        _pre: Optional[Tuple[str, int]] = self._version.pre
        return _pre

    @property
    def post(self) -> Optional[int]:
        # [FUNCTION DOC]
        # Property: post-release number or None.
        return self._version.post[1] if self._version.post else None

    @property
    def dev(self) -> Optional[int]:
        # [FUNCTION DOC]
        # Property: dev-release number or None.
        return self._version.dev[1] if self._version.dev else None

    @property
    def local(self) -> Optional[str]:
        # [FUNCTION DOC]
        # Property: local segment re-joined with dots (lowercased, numbers as ints) or
        #   None.
        if self._version.local:
            return ".".join(str(x) for x in self._version.local)
        else:
            return None

    @property
    def public(self) -> str:
        # [FUNCTION DOC]
        # Property: normalized string without the "+local" part.
        return str(self).split("+", 1)[0]

    @property
    def base_version(self) -> str:
        # [FUNCTION DOC]
        # Property: normalized "epoch!release" string without pre/post/dev/local parts.
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self.release))

        return "".join(parts)

    @property
    def is_prerelease(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if the version has a dev or pre segment.
        return self.dev is not None or self.pre is not None

    @property
    def is_postrelease(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if the version has a post segment.
        return self.post is not None

    @property
    def is_devrelease(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if the version has a dev segment.
        return self.dev is not None

    @property
    def major(self) -> int:
        # [FUNCTION DOC]
        # Property: first release component, or 0 if absent.
        return self.release[0] if len(self.release) >= 1 else 0

    @property
    def minor(self) -> int:
        # [FUNCTION DOC]
        # Property: second release component, or 0 if absent.
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    def micro(self) -> int:
        # [FUNCTION DOC]
        # Property: third release component, or 0 if absent.
        return self.release[2] if len(self.release) >= 3 else 0


def _parse_letter_version(
    letter: str, number: Union[str, bytes, SupportsInt]
) -> Optional[Tuple[str, int]]:
    # [FUNCTION DOC]
    # HELPER: Normalizes a (letter, number) pre/post/dev segment.
    # Behavior: if letter is given, number defaults to 0, letter is lowercased and
    #   aliases are mapped (alpha to a, beta to b, c/pre/preview to rc, rev/r to post);
    #   returns (letter, int(number)). If no letter but a number is given, it is the
    #   implicit post-release form and returns ("post", int(number)). Otherwise returns
    #   None.
    # Inputs: letter - str or None; number - str/int or None.
    # Exceptions: ValueError if number cannot be converted to int.

    if letter:
        # We consider there to be an implicit 0 in a pre-release if there is
        # not a numeral associated with it.
        if number is None:
            number = 0

        # We normalize any letters to their lower case form
        letter = letter.lower()

        # We consider some words to be alternate spellings of other words and
        # in those cases we want to normalize the spellings to our preferred
        # spelling.
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        return letter, int(number)
    if not letter and number:
        # We assume if we are given a number, but we are not given a letter
        # then this is using the implicit post release syntax (e.g. 1.0-1)
        letter = "post"

        return letter, int(number)

    return None


_local_version_separators = re.compile(r"[\._-]")


def _parse_local_version(local: str) -> Optional[LocalType]:
    # [FUNCTION DOC]
    # HELPER: Splits a local version label on ".", "_" or "-" into a tuple where digit-
    #   only parts become ints and others are lowercased strings; returns None when
    #   local is None.
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_separators.split(local)
        )
    return None


def _cmpkey(
    epoch: int,
    release: Tuple[int, ...],
    pre: Optional[Tuple[str, int]],
    post: Optional[Tuple[str, int]],
    dev: Optional[Tuple[str, int]],
    local: Optional[Tuple[SubLocalType]],
) -> CmpKey:
    # [FUNCTION DOC]
    # HELPER: Builds the sort key of a Version.
    # Behavior: strip trailing zeros from release; pre-segment key is NegativeInfinity
    #   for dev-only versions (so 1.0.dev0 sorts before 1.0a0), Infinity when there is
    #   no pre segment, otherwise the pre tuple; post key is NegativeInfinity when
    #   absent; dev key is Infinity when absent; local key is NegativeInfinity when
    #   absent, else a tuple mapping each int part to (int, "") and each string part to
    #   (NegativeInfinity, string) so alphanumeric parts sort before numeric ones.
    # Inputs: the parsed segments. Outputs: tuple (epoch, release, pre, post, dev,
    #   local).

    # When we compare a release version, we want to compare it with all of the
    # trailing zeros removed. So we'll use a reverse the list, drop all the now
    # leading zeros until we come to something non zero, then take the rest
    # re-reverse it back into the correct order and make it a tuple and use
    # that for our sorting key.
    _release = tuple(
        reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release))))
    )

    # We need to "trick" the sorting algorithm to put 1.0.dev0 before 1.0a0.
    # We'll do this by abusing the pre segment, but we _only_ want to do this
    # if there is not a pre or a post segment. If we have one of those then
    # the normal sorting rules will handle this case correctly.
    if pre is None and post is None and dev is not None:
        _pre: PrePostDevType = NegativeInfinity
    # Versions without a pre-release (except as noted above) should sort after
    # those with one.
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre

    # Versions without a post segment should sort before those with one.
    if post is None:
        _post: PrePostDevType = NegativeInfinity

    else:
        _post = post

    # Versions without a development segment should sort after those with one.
    if dev is None:
        _dev: PrePostDevType = Infinity

    else:
        _dev = dev

    if local is None:
        # Versions without a local segment should sort before those with one.
        _local: LocalType = NegativeInfinity
    else:
        # Versions with a local segment need that segment parsed to implement
        # the sorting rules in PEP440.
        # - Alpha numeric segments sort before numeric segments
        # - Alpha numeric segments sort lexicographically
        # - Numeric segments sort numerically
        # - Shorter versions sort before longer versions when the prefixes
        #   match exactly
        _local = tuple(
            (i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local
        )

    return epoch, _release, _pre, _post, _dev, _local
