import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import typing as T

if T.TYPE_CHECKING:
    import keras.api._v2.keras as tfk

class RDLearner:
    """
    Residual Dynamic Learner
    """
    def __init__(self,
        state_dim: int,
        control_dim: int,
        nominal_model: T.Callable[[np.ndarray, np.ndarray], np.ndarray],
        num_induced_points: int = 50,
        batch_size: int = 50,
    ) -> None:
        self.nominal_model = nominal_model

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.num_induced_points = num_induced_points
        self.batch_size = batch_size

        self.X = np.zeros((0, self.state_dim + self.control_dim))
        self.Y = np.zerso((0, self.state_dim))

        self.model = None

    def build_model(self) -> None:
        # multi-output kernel
        kern_list = [gpf.kernels.SquaredExponential() + gpf.kernels.Linear()
                     for _ in range(self.state_dim)]
        kernel = gpf.kernels.LinearCoregionalization(kern_list, W=np.random.randn(P, L))

        # induced points
        Z = gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(np.copy(self.X[:self.num_induced_points]))
        )

        # variational posterior
        q_mu = np.zeros((self.state_dim, self.num_induced_points))
        q_sqrt = np.repeat(np.eye(self.num_induced_points)[None, ...], self.state_dim, axis=0)

        self.model = gpf.models.SVGP(
            kernel=kernel,
            likelihood=gpf.likelihoods.Gaussian(),
            inducing_variables=Z,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
        )

    def optimize(self, num_iter: int) -> np.ndarray:
        losses = np.zeros(num_iter)
        train_iter = iter(self.train_ds)
        train_loss = self.model.training_loss_closure(train_iter, compile=True)
        optimizer = tfk.optimizers.Adam()

        @tf.function
        def train_step(train_iter: tf.data.Iterator):
            data = next(train_iter)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.model.trainable_variables)
                loss = self.model.training_loss(data)

            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            return loss

        for step in range(num_iter):
            losses[step] = train_step(train_iter)

        return losses

    def add_data(self,
        states_tn: np.ndarray,
        controls_tm: np.ndarray,
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

        if self.model is None:
            self.build_model()

        if relearn_iter is not None:
            return self.optimize(relearn_iter)

        return None

