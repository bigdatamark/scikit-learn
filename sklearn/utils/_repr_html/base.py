# [MODULE DOC]
# Base building blocks for HTML representations: a sequential id generator, the
# mixin that builds links to the online API documentation, and the mixin that wires
# _repr_html_/_repr_mimebundle_ to an object's _html_repr callable depending on the
# global 'display' configuration.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import itertools

from sklearn import __version__
from sklearn._config import get_config
from sklearn.utils.fixes import parse_version


class _IDCounter:
    # [CLASS DOC]
    # HELPER: Stateful counter producing unique HTML element ids of the form
    # '<prefix>-<n>', n starting at 1.
    """Generate sequential ids with a prefix."""

    def __init__(self, prefix):
        # [FUNCTION DOC]
        # Inputs: prefix - str prepended to every id.
        # Side effects: sets prefix and resets count to 0.
        self.prefix = prefix
        self.count = 0

    def get_id(self):
        # [FUNCTION DOC]
        # Increments count and returns the next id string 'prefix-count'.
        # Side effects: mutates self.count.
        self.count += 1
        return f"{self.prefix}-{self.count}"


class _HTMLDocumentationLinkMixin:
    # [CLASS DOC]
    # Mixin giving estimators a _get_doc_link method. Class attributes:
    # _doc_link_module (default 'sklearn'), _doc_link_url_param_generator (default None,
    # else a zero-argument callable returning template keyword values), and the
    # _doc_link_template property.
    """Mixin class allowing to generate a link to the API documentation.

    This mixin relies on three attributes:
    - `_doc_link_module`: it corresponds to the root module (e.g. `sklearn`). Using this
      mixin, the default value is `sklearn`.
    - `_doc_link_template`: it corresponds to the template used to generate the
      link to the API documentation. Using this mixin, the default value is
      `"https://scikit-learn.org/{version_url}/modules/generated/
      {estimator_module}.{estimator_name}.html"`.
    - `_doc_link_url_param_generator`: it corresponds to a function that generates the
      parameters to be used in the template when the estimator module and name are not
      sufficient.

    The method :meth:`_get_doc_link` generates the link to the API documentation for a
    given estimator.

    This mixin provides all the necessary states for
    :func:`sklearn.utils.estimator_html_repr` to generate a link to the API
    documentation for the estimator HTML diagram.

    Examples
    --------
    If the default values for `_doc_link_module`, `_doc_link_template` are not suitable,
    then you can override them and provide a method to generate the URL parameters:
    >>> from sklearn.base import BaseEstimator
    >>> doc_link_template = "https://address.local/{single_param}.html"
    >>> def url_param_generator(estimator):
    ...     return {"single_param": estimator.__class__.__name__}
    >>> class MyEstimator(BaseEstimator):
    ...     # use "builtins" since it is the associated module when declaring
    ...     # the class in a docstring
    ...     _doc_link_module = "builtins"
    ...     _doc_link_template = doc_link_template
    ...     _doc_link_url_param_generator = url_param_generator
    >>> estimator = MyEstimator()
    >>> estimator._get_doc_link()
    'https://address.local/MyEstimator.html'

    If instead of overriding the attributes inside the class definition, you want to
    override a class instance, you can use `types.MethodType` to bind the method to the
    instance:
    >>> import types
    >>> estimator = BaseEstimator()
    >>> estimator._doc_link_template = doc_link_template
    >>> estimator._doc_link_url_param_generator = types.MethodType(
    ...     url_param_generator, estimator)
    >>> estimator._get_doc_link()
    'https://address.local/BaseEstimator.html'
    """

    _doc_link_module = "sklearn"
    _doc_link_url_param_generator = None

    @property
    def _doc_link_template(self):
        # [FUNCTION DOC]
        # Property getter. Returns an instance override stored under the literal
        # attribute
        # name '__doc_link_template' if set, else the default scikit-learn URL template
        # whose
        # version segment is 'major.minor' for releases or 'dev' for dev builds, leaving
        # {estimator_module} and {estimator_name} placeholders unformatted.
        # Outputs: str template.
        sklearn_version = parse_version(__version__)
        if sklearn_version.dev is None:
            version_url = f"{sklearn_version.major}.{sklearn_version.minor}"
        else:
            version_url = "dev"
        return getattr(
            self,
            "__doc_link_template",
            (
                f"https://scikit-learn.org/{version_url}/modules/generated/"
                "{estimator_module}.{estimator_name}.html"
            ),
        )

    @_doc_link_template.setter
    def _doc_link_template(self, value):
        # [FUNCTION DOC]
        # Property setter: stores value on the instance under the attribute name
        # '__doc_link_template' (set via setattr, so no name mangling occurs).
        # Side effects: mutates the instance.
        setattr(self, "__doc_link_template", value)

    def _get_doc_link(self):
        # [FUNCTION DOC]
        # Builds the documentation URL. Returns '' if the top-level package of the
        # class's
        # module differs from _doc_link_module. Without a param generator, formats the
        # template with estimator_name (class name) and estimator_module (the dotted
        # module
        # path truncated before the first component starting with '_'). With a
        # generator,
        # formats the template with the dict it returns.
        # Outputs: str URL or ''.
        # Exceptions: KeyError/IndexError from str.format if placeholders are not
        # supplied.
        """Generates a link to the API documentation for a given estimator.

        This method generates the link to the estimator's documentation page
        by using the template defined by the attribute `_doc_link_template`.

        Returns
        -------
        url : str
            The URL to the API documentation for this estimator. If the estimator does
            not belong to module `_doc_link_module`, the empty string (i.e. `""`) is
            returned.
        """
        if self.__class__.__module__.split(".")[0] != self._doc_link_module:
            return ""

        if self._doc_link_url_param_generator is None:
            estimator_name = self.__class__.__name__
            # Construct the estimator's module name, up to the first private submodule.
            # This works because in scikit-learn all public estimators are exposed at
            # that level, even if they actually live in a private sub-module.
            estimator_module = ".".join(
                itertools.takewhile(
                    lambda part: not part.startswith("_"),
                    self.__class__.__module__.split("."),
                )
            )
            return self._doc_link_template.format(
                estimator_module=estimator_module, estimator_name=estimator_name
            )
        return self._doc_link_template.format(**self._doc_link_url_param_generator())


class ReprHTMLMixin:
    # [CLASS DOC]
    # Mixin providing Jupyter display hooks. Subclasses must define _html_repr, a
    # callable returning the HTML string.
    """Mixin to handle consistently the HTML representation.

    When inheriting from this class, you need to define an attribute `_html_repr`
    which is a callable that returns the HTML representation to be shown.
    """

    @property
    def _repr_html_(self):
        # [FUNCTION DOC]
        # Property so that hasattr reflects the config: returns the bound method
        # _repr_html_inner when display config is 'diagram'.
        # Exceptions: AttributeError when the 'display' config is not 'diagram'.
        """HTML representation of estimator.
        This is redundant with the logic of `_repr_mimebundle_`. The latter
        should be favored in the long term, `_repr_html_` is only
        implemented for consumers who do not interpret `_repr_mimbundle_`.
        """
        if get_config()["display"] != "diagram":
            raise AttributeError(
                "_repr_html_ is only defined when the "
                "'display' configuration option is set to "
                "'diagram'"
            )
        return self._repr_html_inner

    def _repr_html_inner(self):
        # [FUNCTION DOC]
        # Returns the result of self._html_repr(), the HTML string.
        """This function is returned by the @property `_repr_html_` to make
        `hasattr(estimator, "_repr_html_") return `True` or `False` depending
        on `get_config()["display"]`.
        """
        return self._html_repr()

    def _repr_mimebundle_(self, **kwargs):
        # [FUNCTION DOC]
        # Builds the mime bundle: always 'text/plain' with repr(self); adds 'text/html'
        # from _html_repr when display config is 'diagram'.
        # Inputs: **kwargs - ignored (Jupyter include/exclude). Outputs: dict.
        """Mime bundle used by jupyter kernels to display estimator"""
        output = {"text/plain": repr(self)}
        if get_config()["display"] == "diagram":
            output["text/html"] = self._html_repr()
        return output
