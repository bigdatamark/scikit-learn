#!/usr/bin/env python
# [MODULE DOC]
# Maintenance script: reads changelog ("what's new") lines from stdin, keeps only
# entry lines (starting with "- ") and continuation lines (starting with a space),
# splits them into entries at each newline followed by "- ", and buckets each entry
# by the sklearn submodules referenced in Sphinx roles (:func:, :meth:, :mod:,
# :class:). One module -> ":mod:`sklearn.<name>`" bucket, several -> "Multiple
# modules", none -> "Miscellaneous". Buckets are printed sorted by key, each with
# a dotted underline heading, entries ordered by label via entry_sort_key, all
# separated by blank lines.
# Side effects: reads stdin, writes stdout.
# KNOWN ISSUE: the trailing reassignment of entry after appending it has no effect
# (dead code), so entries are not stripped/newline-terminated as apparently intended.
# Sorts what's new entries with per-module headings.
# Pass what's new entries on stdin.

import re
import sys
from collections import defaultdict

LABEL_ORDER = ["MajorFeature", "Feature", "Efficiency", "Enhancement", "Fix", "API"]


def entry_sort_key(s):
    # [FUNCTION DOC]
    # HELPER: Sort key for an entry string: entries starting with "- |" return the index
    # of their label (text between the first two "|") in LABEL_ORDER; others return -1
    # so they sort first.
    # Exceptions: ValueError if the label is not in LABEL_ORDER.
    if s.startswith("- |"):
        return LABEL_ORDER.index(s.split("|")[1])
    else:
        return -1


# discard headings and other non-entry lines
text = "".join(l for l in sys.stdin if l.startswith("- ") or l.startswith(" "))

bucketed = defaultdict(list)

for entry in re.split("\n(?=- )", text.strip()):
    modules = re.findall(
        r":(?:func|meth|mod|class):`(?:[^<`]*<|~)?(?:sklearn.)?([a-z]\w+)", entry
    )
    modules = set(modules)
    if len(modules) > 1:
        key = "Multiple modules"
    elif modules:
        key = ":mod:`sklearn.%s`" % next(iter(modules))
    else:
        key = "Miscellaneous"
    bucketed[key].append(entry)
    entry = entry.strip() + "\n"

everything = []
for key, bucket in sorted(bucketed.items()):
    everything.append(key + "\n" + "." * len(key))
    bucket.sort(key=entry_sort_key)
    everything.extend(bucket)
print("\n\n".join(everything))
