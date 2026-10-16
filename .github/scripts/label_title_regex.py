# [MODULE DOC]
# HELPER: CI script (pull_request_target workflow) that labels a PR from its title.
# Inputs: env var CONTEXT_GITHUB holding the JSON GitHub context (repository, token,
# event.number).
# Behavior: searches the title for the whole word "DOC" (adds "Documentation") and
# "CI" (adds "Build / CI"); adds all matching labels in one API call.
# Exceptions: TypeError if CONTEXT_GITHUB is unset; KeyError on missing context keys.
# Side effects: network calls to GitHub, adds labels to the PR.
"""Labels PRs based on title. Must be run in a github action with the
pull_request_target event."""

import json
import os
import re

from github import Auth, Github

context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

repo = context_dict["repository"]
g = Github(auth=Auth.Token(context_dict["token"]))
repo = g.get_repo(repo)
pr_number = context_dict["event"]["number"]
issue = repo.get_issue(number=pr_number)
title = issue.title


regex_to_labels = [(r"\bDOC\b", "Documentation"), (r"\bCI\b", "Build / CI")]

labels_to_add = [label for regex, label in regex_to_labels if re.search(regex, title)]

if labels_to_add:
    issue.add_to_labels(*labels_to_add)
