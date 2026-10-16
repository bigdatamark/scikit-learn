# [MODULE DOC]
# HELPER: Scoring helpers for ASV benchmarks. The make_*_scorers functions attach
# train_scorer and test_scorer callables (signature (y_true, y_pred)) to a benchmark
# instance, used by Estimator.track_train_score / track_test_score.
import numpy as np

from sklearn.metrics import balanced_accuracy_score, r2_score


def neg_mean_inertia(X, labels, centers):
    # [FUNCTION DOC]
    # HELPER: Negative mean squared distance of each sample to its assigned center.
    # Inputs: X (n_samples, n_features) dense or sparse; labels int array (n_samples,);
    # centers (n_clusters, n_features). Outputs: float (higher is better).
    return -(np.asarray(X - centers[labels]) ** 2).sum(axis=1).mean()


def make_gen_classif_scorers(caller):
    # [FUNCTION DOC]
    # HELPER: Sets caller.train_scorer and test_scorer to balanced_accuracy_score.
    caller.train_scorer = balanced_accuracy_score
    caller.test_scorer = balanced_accuracy_score


def make_gen_reg_scorers(caller):
    # [FUNCTION DOC]
    # HELPER: Sets caller.train_scorer and test_scorer to r2_score.
    caller.test_scorer = r2_score
    caller.train_scorer = r2_score


def neg_mean_data_error(X, U, V):
    # [FUNCTION DOC]
    # HELPER: Negative root mean squared reconstruction error of X by U times V.
    # Inputs: X (n, p); U (n, k); V (k, p). Outputs: float.
    return -np.sqrt(((X - U.dot(V)) ** 2).mean())


def make_dict_learning_scorers(caller):
    # [FUNCTION DOC]
    # HELPER: Sets scorers that ignore their arguments and return the negative RMS
    # reconstruction error of caller.X (train) or caller.X_val (test) using the
    # estimator's transform output and components_. Evaluated lazily at score time.
    caller.train_scorer = lambda _, __: (
        neg_mean_data_error(
            caller.X, caller.estimator.transform(caller.X), caller.estimator.components_
        )
    )
    caller.test_scorer = lambda _, __: (
        neg_mean_data_error(
            caller.X_val,
            caller.estimator.transform(caller.X_val),
            caller.estimator.components_,
        )
    )


def explained_variance_ratio(Xt, X):
    # [FUNCTION DOC]
    # HELPER: Ratio of the summed per-column variance of Xt to that of X. Outputs: float.
    return np.var(Xt, axis=0).sum() / np.var(X, axis=0).sum()


def make_pca_scorers(caller):
    # [FUNCTION DOC]
    # HELPER: Sets train_scorer to the sum of the fitted explained_variance_ratio_ and
    # test_scorer to explained_variance_ratio of transformed X_val vs X_val (arguments
    # ignored).
    caller.train_scorer = lambda _, __: caller.estimator.explained_variance_ratio_.sum()
    caller.test_scorer = lambda _, __: (
        explained_variance_ratio(caller.estimator.transform(caller.X_val), caller.X_val)
    )
