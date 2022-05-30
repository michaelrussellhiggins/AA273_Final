import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import typing as T

from gpflow.base import MeanAndVariance, Parameter, TensorType

from tqdm import trange

if T.TYPE_CHECKING:
    import keras.api._v2.keras as tfk

class MultivariateGaussianDiag(gpf.likelihoods.Likelihood):

    DEFAULT_VARIANCE_LOWER_BOUND = 1e-6

    def __init__(self,
        variances: TensorType,
        variance_lower_bound: float = DEFAULT_VARIANCE_LOWER_BOUND,
    ) -> None:
        super().__init__(latent_dim=variance.shape[0], observation_dim=variance.shape[0])

        if tf.reduce_any(variances <= variance_lower_bound):
            raise ValueError(f"All variances must be strictly greater than {variance_lower_bound}")

        self.variances = Parameter(variances,
                                   transform=gpf.utilities.positive(lower=variance_lower_bound))

    def _log_prob(self, F: TensorType, Y: TensorType) -> tf.Tensor:
        a = self.latent_dim * np.log(2 * np.pi)
        b = tf.reduce_sum(tf.math.log(self.variances))
        c = tf.reduce_sum((F - Y) ** 2 / self.variances, axis=-1)
        return -0.5 * (a + b + c)

    def _conditional_mean(self, F: TensorType) -> tf.Tensor:
        return tf.identity(F)

    def _conditional_variance(self, F: TensorType) -> tf.Tensor:
        return tf.broadcast_to(self.variances, tf.shape(F))

    def _predict_mean_and_var(self, Fmu: TensorType, Fvar: TensorType) -> MeanAndVariance:
        tf.identity(Fmu), F_var + self.variances

    def _predict_log_density(self, Fmu: TensorType, Fvar: TensorType, Y: TensorType) -> tf.Tensor:
        pass

class RDLearner:
    """
    Residual Dynamic Learner
    """
    def __init__(self,
        state_dim: int,
        control_dim: int,
        nominal_model: T.Callable[[np.ndarray, np.ndarray], np.ndarray],
        num_induced_points: int = 50,
        batch_size: int = 100,
    ) -> None:
        self.nominal_model = nominal_model

        self.num_induced_points = num_induced_points
        self.batch_size = batch_size

        self.input_dim = state_dim + control_dim
        self.output_dim = state_dim

        self.X = np.zeros((0, self.input_dim))
        self.Y = np.zeros((0, self.output_dim))

        self.model = None
        self.posterior = None

    def build_model(self) -> None:
        # multi-output kernel
        kern_list = [
            gpf.kernels.SquaredExponential(lengthscales=tf.ones(self.input_dim)) + \
            gpf.kernels.Linear()
                for _ in range(self.output_dim)]
        kernel = gpf.kernels.LinearCoregionalization(
            kern_list, W=np.random.randn(self.output_dim, self.output_dim))

        # induced points
        Z = gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(np.copy(self.X[:self.num_induced_points]))
        )

        # variational posterior
        q_mu = np.zeros((self.num_induced_points, self.output_dim))
        q_sqrt = np.repeat(np.eye(self.num_induced_points)[None, ...], self.output_dim, axis=0)

        self.model = gpf.models.SVGP(
            kernel=kernel,
            likelihood=gpf.likelihoods.Gaussian(),
            inducing_variable=Z,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
        )
        self.variational_vars = [(self.model.q_mu, self.model.q_sqrt)]
        self.adam_vars = self.model.kernel.trainable_variables + \
                         self.model.likelihood.trainable_variables + \
                         self.model.inducing_variable.trainable_variables

    def optimize(self, num_iter: int) -> np.ndarray:
        losses = np.zeros(num_iter)

        adam_opt = tfk.optimizers.Adam(0.01)
        nat_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

        @tf.function
        def train_step(train_iter: tf.data.Iterator):
            data = next(train_iter)
            train_loss = self.model.training_loss_closure(data)

            nat_opt.minimize(train_loss, var_list=self.variational_vars)
            adam_opt.minimize(train_loss, var_list=self.adam_vars)

            return train_loss()

        print("Start optimizing with existing data buffer...")
        for step in trange(num_iter):
            losses[step] = train_step(self.train_iter)

        self.posterior = self.model.posterior()

        return losses

    def predict(self, state_n: TensorType, control_m: TensorType, with_jacobian:bool = True) -> tf.Tensor:
        inputs_k = tf.concat([state_n, control_m], axis=-1)

        inputs_1k = inputs_k[tf.newaxis]
        outputs_1n, cov_1nn = self.posterior.predict_f(inputs_1k, full_output_cov=True)
        outputs_n = outputs_1n[0]
        cov_nn = cov_1nn[0]

        return outputs_n, cov_nn

    def add_data(self,
        states_tn: TensorType,
        controls_tm: TensorType,
        relearn_iter: T.Optional[int] = None,
    ) -> T.Optional[np.ndarray]:
        next_states_tn = self.nominal_model(states_tn, controls_tm)
        residual_tp = states_tn[1:] - next_states_tn[:-1]

        X_new = np.concatenate([states_tn, controls_tm], axis=-1)[:-1]
        Y_new = residual_tp

        self.X = np.concatenate([self.X, X_new])
        self.Y = np.concatenate([self.Y, Y_new])

        num_data = self.X.shape[0]
        indics = np.arange(num_data)
        np.random.shuffle(indics)

        self.X = self.X[indics]
        self.Y = self.Y[indics]

        # construct new training dataset
        train_ds = tf.data.Dataset.from_tensor_slices((self.X, self.Y))
        train_ds = train_ds.repeat().shuffle(num_data)
        train_ds = train_ds.batch(self.batch_size)
        self.train_iter = iter(train_ds)

        if self.model is None:
            self.build_model()

        if relearn_iter is not None:
            return self.optimize(relearn_iter)

        return None

