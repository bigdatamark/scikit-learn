# [MODULE DOC]
# Cython declaration file for sklearn._loss._loss, cimported by other Cython
# modules (e.g. SGD, SAG) that need nogil single-sample loss evaluations.
# Declares the fused input/output float types, the double_pair return struct,
# and the cdef interfaces of the Cy* loss extension types implemented in
# _loss.pyx.tp.
# Fused types for input like y_true, raw_prediction, sample_weights.
ctypedef fused floating_in:
    double
    float


# Fused types for output like gradient and hessian
# We use a different fused types for input (floating_in) and output (floating_out), such
# that input and output can have different dtypes in the same function call. A single
# fused type can only take on one single value (type) for all arguments in one function
# call.
ctypedef fused floating_out:
    double
    float


# Struct to return 2 doubles
ctypedef struct double_pair:
    # [CLASS DOC]
    # Plain C struct for returning two doubles from nogil kernels, e.g. (loss,
    # gradient) or (gradient, hessian), or (max, sum of exps) in the softmax helper.
    double val1
    double val2


# C base class for loss functions
cdef class CyLossFunction:
    # [CLASS DOC]
    # Abstract base extension type for 1D-target losses; declares the three
    # nogil single-sample methods that subclasses implement (see _loss.pyx.tp).
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # Abstract; returns the CyLossFunction loss for one sample given y_true and raw_prediction
    # (link space). noexcept nogil: cannot raise; no side effects.
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # Abstract; returns d loss / d raw_prediction of CyLossFunction for one sample.
    # noexcept nogil: cannot raise; no side effects.
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # Abstract; returns (gradient, hessian) of CyLossFunction w.r.t. raw_prediction for one
    # sample as a double_pair. noexcept nogil: cannot raise; no side effects.


cdef class CyHalfSquaredError(CyLossFunction):
    # [CLASS DOC]
    # Half squared error 0.5 (raw - y)^2, identity link.
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns the CyHalfSquaredError loss for one sample given y_true and raw_prediction
    # (link space). noexcept nogil: cannot raise; no side effects.
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns d loss / d raw_prediction of CyHalfSquaredError for one sample.
    # noexcept nogil: cannot raise; no side effects.
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns (gradient, hessian) of CyHalfSquaredError w.r.t. raw_prediction for one
    # sample as a double_pair. noexcept nogil: cannot raise; no side effects.


cdef class CyAbsoluteError(CyLossFunction):
    # [CLASS DOC]
    # Absolute error |raw - y|, identity link.
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns the CyAbsoluteError loss for one sample given y_true and raw_prediction
    # (link space). noexcept nogil: cannot raise; no side effects.
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns d loss / d raw_prediction of CyAbsoluteError for one sample.
    # noexcept nogil: cannot raise; no side effects.
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns (gradient, hessian) of CyAbsoluteError w.r.t. raw_prediction for one
    # sample as a double_pair. noexcept nogil: cannot raise; no side effects.


cdef class CyPinballLoss(CyLossFunction):
    # [CLASS DOC]
    # Pinball (quantile) loss, identity link; quantile in (0, 1) is a readonly
    # attribute visible from Python.
    cdef readonly double quantile  # readonly makes it accessible from Python
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns the CyPinballLoss loss for one sample given y_true and raw_prediction
    # (link space). noexcept nogil: cannot raise; no side effects.
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns d loss / d raw_prediction of CyPinballLoss for one sample.
    # noexcept nogil: cannot raise; no side effects.
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns (gradient, hessian) of CyPinballLoss w.r.t. raw_prediction for one
    # sample as a double_pair. noexcept nogil: cannot raise; no side effects.


cdef class CyHuberLoss(CyLossFunction):
    # [CLASS DOC]
    # Huber loss, identity link; delta > 0 is a public (read/write) attribute.
    cdef public double delta  # public makes it accessible from Python
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns the CyHuberLoss loss for one sample given y_true and raw_prediction
    # (link space). noexcept nogil: cannot raise; no side effects.
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns d loss / d raw_prediction of CyHuberLoss for one sample.
    # noexcept nogil: cannot raise; no side effects.
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns (gradient, hessian) of CyHuberLoss w.r.t. raw_prediction for one
    # sample as a double_pair. noexcept nogil: cannot raise; no side effects.


cdef class CyHalfPoissonLoss(CyLossFunction):
    # [CLASS DOC]
    # Half Poisson deviance with log link (constants dropped).
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns the CyHalfPoissonLoss loss for one sample given y_true and raw_prediction
    # (link space). noexcept nogil: cannot raise; no side effects.
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns d loss / d raw_prediction of CyHalfPoissonLoss for one sample.
    # noexcept nogil: cannot raise; no side effects.
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns (gradient, hessian) of CyHalfPoissonLoss w.r.t. raw_prediction for one
    # sample as a double_pair. noexcept nogil: cannot raise; no side effects.


cdef class CyHalfGammaLoss(CyLossFunction):
    # [CLASS DOC]
    # Half Gamma deviance with log link (constants dropped).
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns the CyHalfGammaLoss loss for one sample given y_true and raw_prediction
    # (link space). noexcept nogil: cannot raise; no side effects.
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns d loss / d raw_prediction of CyHalfGammaLoss for one sample.
    # noexcept nogil: cannot raise; no side effects.
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns (gradient, hessian) of CyHalfGammaLoss w.r.t. raw_prediction for one
    # sample as a double_pair. noexcept nogil: cannot raise; no side effects.


cdef class CyHalfTweedieLoss(CyLossFunction):
    # [CLASS DOC]
    # Half Tweedie deviance with log link (constants dropped); power is a
    # readonly attribute.
    cdef readonly double power  # readonly makes it accessible from Python
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns the CyHalfTweedieLoss loss for one sample given y_true and raw_prediction
    # (link space). noexcept nogil: cannot raise; no side effects.
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns d loss / d raw_prediction of CyHalfTweedieLoss for one sample.
    # noexcept nogil: cannot raise; no side effects.
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns (gradient, hessian) of CyHalfTweedieLoss w.r.t. raw_prediction for one
    # sample as a double_pair. noexcept nogil: cannot raise; no side effects.


cdef class CyHalfTweedieLossIdentity(CyLossFunction):
    # [CLASS DOC]
    # Half Tweedie deviance with identity link (full deviance); power is a
    # readonly attribute.
    cdef readonly double power  # readonly makes it accessible from Python
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns the CyHalfTweedieLossIdentity loss for one sample given y_true and raw_prediction
    # (link space). noexcept nogil: cannot raise; no side effects.
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns d loss / d raw_prediction of CyHalfTweedieLossIdentity for one sample.
    # noexcept nogil: cannot raise; no side effects.
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns (gradient, hessian) of CyHalfTweedieLossIdentity w.r.t. raw_prediction for one
    # sample as a double_pair. noexcept nogil: cannot raise; no side effects.


cdef class CyHalfBinomialLoss(CyLossFunction):
    # [CLASS DOC]
    # Half binomial deviance (log loss) with logit link.
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns the CyHalfBinomialLoss loss for one sample given y_true and raw_prediction
    # (link space). noexcept nogil: cannot raise; no side effects.
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns d loss / d raw_prediction of CyHalfBinomialLoss for one sample.
    # noexcept nogil: cannot raise; no side effects.
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns (gradient, hessian) of CyHalfBinomialLoss w.r.t. raw_prediction for one
    # sample as a double_pair. noexcept nogil: cannot raise; no side effects.


cdef class CyExponentialLoss(CyLossFunction):
    # [CLASS DOC]
    # Exponential (AdaBoost) loss with half-logit link.
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns the CyExponentialLoss loss for one sample given y_true and raw_prediction
    # (link space). noexcept nogil: cannot raise; no side effects.
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns d loss / d raw_prediction of CyExponentialLoss for one sample.
    # noexcept nogil: cannot raise; no side effects.
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil
    # [FUNCTION DOC]
    # returns (gradient, hessian) of CyExponentialLoss w.r.t. raw_prediction for one
    # sample as a double_pair. noexcept nogil: cannot raise; no side effects.


cdef class CyHalfMultinomialLoss():
    # [CLASS DOC]
    # Half multinomial (softmax cross-entropy) loss; standalone type (not a
    # CyLossFunction subclass) because its single-sample gradient is vector valued.
    cdef void cy_gradient(
        self,
        const floating_in y_true,
        const floating_in[::1] raw_prediction,
        const floating_in sample_weight,
        floating_out[::1] gradient_out,
    ) noexcept nogil
    # [FUNCTION DOC]
    # Single-sample gradient over all classes: writes (softmax(raw_prediction)[k] -
    # [y_true == k]) * sample_weight into gradient_out (length n_classes).
    # noexcept nogil: cannot raise; mutates only gradient_out.
