# [MODULE DOC]
# Sphinx extension registering an "autoshortsummary" directive that renders only the
# first non-empty line (short summary) of an object's processed docstring, with no
# directive header, options or members. Used by the API reference templates.
# Side effects: none at import time; setup registers the documenter with the app.
from sphinx.ext.autodoc import ModuleLevelDocumenter


class ShortSummaryDocumenter(ModuleLevelDocumenter):
    # [CLASS DOC]
    # Module-level autodoc documenter (objtype "shortsummary") that outputs a single
    # line of
    # text. Class attributes: content_indent is empty so the line is not indented;
    # priority
    # -99 keeps it from being auto-selected for members since it accepts every object.
    """An autodocumenter that only renders the short summary of the object."""

    # Defines the usage: .. autoshortsummary:: {{ object }}
    objtype = "shortsummary"

    # Disable content indentation
    content_indent = ""

    # Avoid being selected as the default documenter for some objects, because we are
    # returning `can_document_member` as True for all objects
    priority = -99

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        # [FUNCTION DOC]
        # Accepts any member unconditionally. Inputs follow the autodoc documenter
        # protocol
        # (member object, its name, attribute flag, parent documenter). Outputs: always
        # True.
        """Allow documenting any object."""
        return True

    def get_object_members(self, want_all):
        # [FUNCTION DOC]
        # Reports that no members are documented. Input want_all is ignored.
        # Outputs: tuple (False, empty list).
        """Document no members."""
        return (False, [])

    def add_directive_header(self, sig):
        # [FUNCTION DOC]
        # No-op override so that no directive header/options are emitted. Input sig
        # ignored.
        """Override default behavior to add no directive header or options."""
        pass

    def add_content(self, more_content):
        # [FUNCTION DOC]
        # Fetches the object's docstrings via get_doc; if None, emits nothing. If the
        # list is
        # empty, appends an empty block so processing still runs. Runs process_doc and
        # takes the
        # first non-empty line, falling back to the literal text "<no summary>", then
        # adds it as
        # a single content line at indentation 0. Input more_content is ignored.
        # Side effects: appends one line to the directive result; may mutate the
        # docstrings
        # list returned by get_doc.
        """Override default behavior to add only the first line of the docstring.

        Modified based on the part of processing docstrings in the original
        implementation of this method.

        https://github.com/sphinx-doc/sphinx/blob/faa33a53a389f6f8bc1f6ae97d6015fa92393c4a/sphinx/ext/autodoc/__init__.py#L609-L622
        """
        sourcename = self.get_sourcename()
        docstrings = self.get_doc()

        if docstrings is not None:
            if not docstrings:
                docstrings.append([])
            # Get the first non-empty line of the processed docstring; this could lead
            # to unexpected results if the object does not have a short summary line.
            short_summary = next(
                (s for s in self.process_doc(docstrings) if s), "<no summary>"
            )
            self.add_line(short_summary, sourcename, 0)


def setup(app):
    # [FUNCTION DOC]
    # Sphinx extension entry point: registers ShortSummaryDocumenter on app.
    # Outputs: None (no extension metadata dict, so parallel safety is not declared).
    app.add_autodocumenter(ShortSummaryDocumenter)
