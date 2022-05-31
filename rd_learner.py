import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import typing as T

from sklearn.cluster import KMeans
from gpflow.base import MeanAndVariance, Parameter, TensorType

from tqdm import trange, tqdm

if T.TYPE_CHECKING:
    import keras.api._v2.keras as tfk

class MultivariateGaussianDiag(gpf.likelihoods.Likelihood):

    DEFAULT_VARIANCE_LOWER_BOUND = 1e-6

    def __init__(self,
        variances: TensorType,
        variance_lower_bound: float = DEFAULT_VARIANCE_LOWER_BOUND,
    ) -> None:
        super().__init__(latent_dim=variances.shape[0], observation_dim=variances.shape[0])

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
        return tf.reduce_sum(gpf.logdensities.gaussian(Y, Fmu, Fvar + self.variances), axis=-1)

    def _variational_expectations(self, Fmu: TensorType, Fvar: TensorType, Y: TensorType) -> tf.Tensor:
        return tf.reduce_sum(
            -0.5 * np.log(2 * np.pi)
            -0.5 * tf.math.log(self.variances)
            -0.5 * ((Y - Fmu) ** 2 + Fvar) / self.variances,
            axis=-1,
        )

class RDLearner2:
    def __init__(self,
        state_dim: int,
        control_dim: int,
        nominal_model: T.Callable[[np.ndarray, np.ndarray], np.ndarray],
        num_induced_points: int = 50,
    ) -> None:
        self.nominal_model = nominal_model
        self.num_induced_points = num_induced_points

        self.input_dim = state_dim + control_dim
        self.output_dim = state_dim

        self.X = np.zeros((0, self.input_dim))
        self.Y = np.zeros((0, self.output_dim))

        self.models = None
        self.posteriors = None

    def build_models(self) -> None:
        # induced variables
        kmeans = KMeans(n_clusters=self.num_induced_points).fit(self.X)
        Z = kmeans.cluster_centers_

        # build model for each output
        self.models = []
        for i in range(self.output_dim):
            kernel = gpf.kernels.SquaredExponential(lengthscales=tf.ones(self.input_dim)) + \
                     gpf.kernels.Linear()
            model = gpf.models.SGPR(
                data=(self.X, self.Y[:, i, None]),
                kernel=kernel,
                inducing_variable=Z.copy(),
                mean_function=gpf.mean_functions.Constant(),
            )
            self.models.append(model)

    def add_data(self,
        states_tn: np.ndarray,
        controls_tm: np.ndarray,
    ) -> None:
        next_states_tn = np.zeros_like(states_tn)
        for i in range(states_tn.shape[0]):
            next_states_tn[i] = self.nominal_model(states_tn[i], controls_tm[i])

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

        if self.models is None:
            self.build_models()

    def optimize(self, max_iter: int = 300) -> T.List[T.Any]:
        assert self.models is not None, "models have not been built"

        optimizers = [gpf.optimizers.Scipy() for _ in self.models]
        opt_stats = []
        self.posteriors = []
        for opt, model in tqdm(zip(optimizers, self.models)):
            opt_stats.append(opt.minimize(model.training_loss, model.trainable_variables,
                                          options=dict(maxiter=max_iter)))
            self.posteriors.append(model.posterior())

        return opt_stats

    def predict(self, state_n: TensorType, control_m: TensorType) -> T.Tuple[tf.Tensor, tf.Tensor]:
        outputs_1n, cov_1n = self.predict_batch(state_n[tf.newaxis], control_m[tf.newaxis])

        return outputs_1n[0], cov_1n[0]

    def predict_batch(self,
        state_bn: TensorType,
        control_bm: TensorType
    ) -> T.Tuple[tf.Tensor, tf.Tensor]:
        inputs_bk = tf.concat([state_bn, control_bm], axis=-1)

        outputs_bn = []
        cov_bn = []
        models = self.posteriors if self.posteriors is not None else self.models
        for model in models:
            outputs_b1, cov_b1 = model.predict_f(inputs_bk)
            outputs_bn.append(outputs_b1)
            cov_bn.append(cov_b1)

        return tf.concat(outputs_bn, axis=-1), tf.concat(cov_bn, axis=-1)

    def variances(self) -> tf.Tensor:
        return tf.concat([m.likelihood.variance for m in self.models], axis=0)

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

        # induced points from kmeans
        kmeans = KMeans(n_clusters=self.num_induced_points).fit(self.X)
        Z = gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(kmeans.cluster_centers_)
        )

        # variational posterior
        q_mu = np.zeros((self.num_induced_points, self.output_dim))
        q_sqrt = np.repeat(np.eye(self.num_induced_points)[None, ...], self.output_dim, axis=0)

        self.model = gpf.models.SVGP(
            kernel=kernel,
            likelihood=MultivariateGaussianDiag(np.ones(self.output_dim)),
            inducing_variable=Z,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            mean_function=gpf.mean_functions.Constant(tf.zeros(self.output_dim)),
        )

    def optimize(self,
        num_iter: int,
        adam_lr: float = 1e-2,
        nat_lr: float = 5e-2,
    ) -> np.ndarray:
        losses = np.zeros(num_iter)

        adam_opt = tfk.optimizers.Adam(adam_lr)
        nat_opt = gpf.optimizers.NaturalGradient(gamma=nat_lr)

        variational_vars = [(self.model.q_mu, self.model.q_sqrt)]
        adam_vars = self.model.kernel.trainable_variables + \
                    self.model.likelihood.trainable_variables + \
                    self.model.inducing_variable.trainable_variables + \
                    self.model.mean_function.trainable_variables

        @tf.function
        def train_step(train_iter: tf.data.Iterator):
            data = next(train_iter)
            train_loss = self.model.training_loss_closure(data)

            nat_opt.minimize(train_loss, var_list=variational_vars)
            adam_opt.minimize(train_loss, var_list=adam_vars)

            return train_loss()

        print("Start optimizing with existing data buffer...")
        for step in trange(num_iter):
            losses[step] = train_step(self.train_iter)

        self.posterior = self.model.posterior()

        return losses

    def predict(self, state_n: TensorType, control_m: TensorType) -> T.Tuple[tf.Tensor, tf.Tensor]:
        inputs_k = tf.concat([state_n, control_m], axis=-1)

        inputs_1k = inputs_k[tf.newaxis]
        outputs_1n, cov_1nn = self.posterior.predict_f(inputs_1k, full_output_cov=True)
        outputs_n = outputs_1n[0]
        cov_nn = cov_1nn[0]

        return outputs_n, cov_nn

    def predict_batch(self, state_bn: TensorType, control_bm: TensorType) -> tf.Tensor:
        inputs_bk = tf.concat([state_bn, control_bm], axis=-1)

        return self.posterior.predict_f(inputs_bk, full_output_cov=True)


    def add_data(self,
        states_tn: np.ndarray,
        controls_tm: np.ndarray,
        relearn_iter: T.Optional[int] = None,
    ) -> T.Optional[np.ndarray]:
        next_states_tn = np.zeros_like(states_tn)
        for i in range(states_tn.shape[0]):
            next_states_tn[i] = self.nominal_model(states_tn[i], controls_tm[i])

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

