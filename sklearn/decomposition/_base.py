# [MODULE DOC]
# Shared base class for PCA-like estimators (PCA, IncrementalPCA, ...). Provides the
# probabilistic-PCA covariance/precision, projection, inverse projection and output
# feature naming, written against the array API namespace of the fitted attributes.
"""Principal Component Analysis Base Classes"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import linalg

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils._array_api import (
    _add_to_diagonal,
    check_same_namespace,
    device,
    get_namespace,
)
from sklearn.utils.validation import check_array, check_is_fitted, validate_data


class _BasePCA(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, metaclass=ABCMeta
):
    # [CLASS DOC]
    # HELPER: Abstract base mixing feature-name prefixing, transformer and estimator
    # behavior. Subclasses must implement fit and set components_ (n_components,
    # n_features), explained_variance_ (n_components,), noise_variance_ (scalar), mean_
    # (n_features,), n_components_ and the whiten flag; all methods below read these.
    """Base class for PCA methods.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def get_covariance(self):
        # [FUNCTION DOC]
        # Builds the probabilistic PCA covariance: if whiten, first rescales each
        # component row by the square root of its explained variance; computes per-
        # component variance minus noise_variance_, set to zero where the variance does
        # not exceed the noise; returns the sum over components of that difference times
        # the outer product of the component with itself, then adds noise_variance_ to
        # the diagonal.
        # Inputs: none (uses fitted attributes).
        # Outputs: (n_features, n_features) array in the namespace/device of
        # components_.
        # Exceptions: attribute errors if not fitted.
        # Side effects: none.
        """Compute data covariance with the generative model.

        ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
        where S**2 contains the explained variances, and sigma2 contains the
        noise variances.

        Returns
        -------
        cov : array of shape=(n_features, n_features)
            Estimated covariance of data.
        """
        xp, _ = get_namespace(self.components_)

        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * xp.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = exp_var - self.noise_variance_
        exp_var_diff = xp.where(
            exp_var > self.noise_variance_,
            exp_var_diff,
            xp.asarray(0.0, device=device(exp_var), dtype=exp_var.dtype),
        )
        cov = (components_.T * exp_var_diff) @ components_
        _add_to_diagonal(cov, self.noise_variance_, xp)
        return cov

    def get_precision(self):
        # [FUNCTION DOC]
        # Inverse of get_covariance. If n_components_ is 0 returns identity divided by
        # the noise variance; if the noise variance is 0 directly inverts get_covariance
        # (array API inv or scipy inv). Otherwise applies the matrix inversion
        # (Woodbury) lemma: with (possibly whiten-rescaled) components W and clipped
        # variance differences d, forms W times W transposed divided by noise, adds 1/d
        # to its diagonal, inverts this small (n_components square) matrix, sandwiches
        # it between W transposed and W, divides by minus noise squared, and adds
        # 1/noise to the diagonal.
        # Outputs: (n_features, n_features) precision array.
        # Exceptions: LinAlgError from the inversion if singular.
        # Side effects: none.
        # KNOWN ISSUE: when some explained variance does not exceed the noise variance,
        # d is clipped to 0 and 1/d becomes infinite on the diagonal (division warning,
        # inf/NaN risk); the zero-dtype in this where call also omits the dtype used in
        # get_covariance.
        """Compute data precision matrix with the generative model.

        Equals the inverse of the covariance but computed with
        the matrix inversion lemma for efficiency.

        Returns
        -------
        precision : array, shape=(n_features, n_features)
            Estimated precision of data.
        """
        xp, is_array_api_compliant = get_namespace(self.components_)

        n_features = self.components_.shape[1]

        # handle corner cases first
        if self.n_components_ == 0:
            return xp.eye(n_features) / self.noise_variance_

        if is_array_api_compliant:
            linalg_inv = xp.linalg.inv
        else:
            linalg_inv = linalg.inv

        if self.noise_variance_ == 0.0:
            return linalg_inv(self.get_covariance())

        # Get precision using matrix inversion lemma
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * xp.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = exp_var - self.noise_variance_
        exp_var_diff = xp.where(
            exp_var > self.noise_variance_,
            exp_var_diff,
            xp.asarray(0.0, device=device(exp_var)),
        )
        precision = components_ @ components_.T / self.noise_variance_
        _add_to_diagonal(precision, 1.0 / exp_var_diff, xp)
        precision = components_.T @ linalg_inv(precision) @ components_
        precision /= -(self.noise_variance_**2)
        _add_to_diagonal(precision, 1.0 / self.noise_variance_, xp)
        return precision

    @abstractmethod
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Abstract placeholder; subclasses fit the model on X (n_samples, n_features), y
        # ignored, set the fitted attributes listed on the class, and return self.
        """Placeholder for fit. Subclasses should implement this method!

        Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

    def transform(self, X):
        # [FUNCTION DOC]
        # Public projection. Checks X shares the namespace of components_, that the
        # estimator is fitted, validates X (float64/float32, dense or CSR/CSC sparse,
        # n_features must match, reset=False) and delegates to _transform with
        # x_is_centered False.
        # Inputs: X (n_samples, n_features).
        # Outputs: (n_samples, n_components) array.
        # Exceptions: NotFittedError; ValueError on feature-count mismatch or bad input;
        # namespace mismatch error from check_same_namespace.
        # Side effects: none.
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Projection of X in the first principal components, where `n_samples`
            is the number of samples and `n_components` is the number of the components.
        """
        check_same_namespace(X, self, attribute="components_", method="transform")
        xp, _ = get_namespace(X)

        check_is_fitted(self)

        X = validate_data(
            self,
            X,
            dtype=[xp.float64, xp.float32],
            accept_sparse=("csr", "csc"),
            reset=False,
        )
        return self._transform(X, xp=xp, x_is_centered=False)

    def _transform(self, X, xp, x_is_centered=False):
        # [FUNCTION DOC]
        # HELPER: Projects X onto components_ (X times components transposed). Unless
        # x_is_centered, subtracts the projection of mean_ afterwards (keeps sparse X
        # sparse and avoids copying dense X). If whiten, divides each output column by
        # the square root of its explained variance, with the scale clipped below at the
        # dtype machine epsilon to avoid non-finite values.
        # Inputs: X validated dense or sparse array; xp the array namespace;
        # x_is_centered bool, default False.
        # Outputs: (n_samples, n_components) dense array.
        # Side effects: none on X or self.
        X_transformed = X @ self.components_.T
        if not x_is_centered:
            # Apply the centering after the projection.
            # For dense X this avoids copying or mutating the data passed by
            # the caller.
            # For sparse X it keeps sparsity and avoids having to wrap X into
            # a linear operator.
            X_transformed -= xp.reshape(self.mean_, (1, -1)) @ self.components_.T
        if self.whiten:
            # For some solvers (such as "arpack" and "covariance_eigh"), on
            # rank deficient data, some components can have a variance
            # arbitrarily close to zero, leading to non-finite results when
            # whitening. To avoid this problem we clip the variance below.
            scale = xp.sqrt(self.explained_variance_)
            min_scale = xp.finfo(scale.dtype).eps
            scale[scale < min_scale] = min_scale
            X_transformed /= scale
        return X_transformed

    def inverse_transform(self, X):
        # [FUNCTION DOC]
        # Maps reduced data back: checks namespace and fitted state, validates X as
        # float64 or float32, then returns X times components_ plus mean_; if whiten,
        # the components are first scaled by the square root of explained_variance_ so
        # whitening is undone.
        # Inputs: X (n_samples, n_components).
        # Outputs: (n_samples, n_features) array.
        # Exceptions: NotFittedError, validation errors.
        # KNOWN ISSUE: unlike _transform, no epsilon clipping of the variance is
        # applied, so the inverse is not exact for components whose scale was clipped.
        """Transform data back to its original space.

        In other words, return an input `X_original` whose transform would be X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where `n_samples` is the number of samples
            and `n_components` is the number of components.

        Returns
        -------
        X_original : array-like of shape (n_samples, n_features)
            Original data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Notes
        -----
        If whitening is enabled, inverse_transform will compute the
        exact inverse operation, which includes reversing whitening.
        """
        check_same_namespace(
            X, self, attribute="components_", method="inverse_transform"
        )
        xp, _ = get_namespace(X)

        check_is_fitted(self)

        X = check_array(X, input_name="X", dtype=[xp.float64, xp.float32])

        if self.whiten:
            scaled_components = (
                xp.sqrt(self.explained_variance_[:, np.newaxis]) * self.components_
            )
            return X @ scaled_components + self.mean_
        else:
            return X @ self.components_ + self.mean_

    @property
    def _n_features_out(self):
        # [FUNCTION DOC]
        # HELPER: Property giving the number of output features (rows of components_),
        # used to generate get_feature_names_out names.
        """Number of transformed output features."""
        return self.components_.shape[0]
