# [MODULE DOC]
# HELPER: Vendored from pypa/packaging; defines sentinel singletons Infinity and
#   NegativeInfinity that compare greater than / less than every other object. They are
#   used inside version comparison keys to order missing pre/post/dev/local segments.
# Module attributes: Infinity (an InfinityType instance) and NegativeInfinity (a
#   NegativeInfinityType instance).
"""Vendoered from
https://github.com/pypa/packaging/blob/main/packaging/_structures.py
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


class InfinityType:
    # [CLASS DOC]
    # HELPER: Sentinel type whose instances compare greater than any other object; only
    #   equal to other instances of the same class.
    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the string "Infinity".
        return "Infinity"

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Hashes the repr string, so all instances share one hash.
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Always returns False regardless of other (comparison with anything, including
        #   another instance of the same type).
        return False

    def __le__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Always returns False regardless of other (comparison with anything, including
        #   another instance of the same type).
        # KNOWN ISSUE: for two instances of the same class this is inconsistent with
        #   __eq__ in one of the directions (e.g. Infinity <= Infinity is False while
        #   Infinity == Infinity is True), inherited from upstream.
        return False

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Returns True only if other is an instance of the same class (or a subclass).
        return isinstance(other, self.__class__)

    def __ne__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Negation of __eq__: True unless other is an instance of the same class.
        return not isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Always returns True regardless of other (comparison with anything, including
        #   another instance of the same type).
        return True

    def __ge__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Always returns True regardless of other (comparison with anything, including
        #   another instance of the same type).
        # KNOWN ISSUE: for two instances of the same class this is inconsistent with
        #   __eq__ in one of the directions (e.g. Infinity <= Infinity is False while
        #   Infinity == Infinity is True), inherited from upstream.
        return True

    def __neg__(self: object) -> "NegativeInfinityType":
        # [FUNCTION DOC]
        # Returns the opposite module-level singleton (NegativeInfinity).
        return NegativeInfinity


Infinity = InfinityType()


class NegativeInfinityType:
    # [CLASS DOC]
    # HELPER: Sentinel type whose instances compare smaller than any other object; only
    #   equal to other instances of the same class.
    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the string "-Infinity".
        return "-Infinity"

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Hashes the repr string, so all instances share one hash.
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Always returns True regardless of other (comparison with anything, including
        #   another instance of the same type).
        return True

    def __le__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Always returns True regardless of other (comparison with anything, including
        #   another instance of the same type).
        # KNOWN ISSUE: for two instances of the same class this is inconsistent with
        #   __eq__ in one of the directions (e.g. Infinity <= Infinity is False while
        #   Infinity == Infinity is True), inherited from upstream.
        return True

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Returns True only if other is an instance of the same class (or a subclass).
        return isinstance(other, self.__class__)

    def __ne__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Negation of __eq__: True unless other is an instance of the same class.
        return not isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Always returns False regardless of other (comparison with anything, including
        #   another instance of the same type).
        return False

    def __ge__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Always returns False regardless of other (comparison with anything, including
        #   another instance of the same type).
        # KNOWN ISSUE: for two instances of the same class this is inconsistent with
        #   __eq__ in one of the directions (e.g. Infinity <= Infinity is False while
        #   Infinity == Infinity is True), inherited from upstream.
        return False

    def __neg__(self: object) -> InfinityType:
        # [FUNCTION DOC]
        # Returns the opposite module-level singleton (Infinity).
        return Infinity


NegativeInfinity = NegativeInfinityType()
