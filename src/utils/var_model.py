import numpy as np

class VAR:
    def fit(self, mouse_event_sequences):
        deltas = [mouse_event_sequence.vectorise() for mouse_event_sequence in mouse_event_sequences]

        self.__initial_deltas = np.array([delta[0] for delta in deltas])
        deltas = np.vstack(deltas) # Stack sequences into matrix

        current_deltas = deltas[1:]
        # (x_1, y_2, 1), (x_1, y_2, 1) etc.
        lagged_deltas = np.hstack([deltas[:-1], np.ones((len(current_deltas), 1))])

        # Find mapping between last and current delta
        x, _, _, _ = np.linalg.lstsq(lagged_deltas, current_deltas)

        self.__coefficients = x[:-1].T
        self.__intercept = x[-1]

        # Determine diff between desired and pred
        diff = current_deltas - lagged_deltas @ x
        self.__covariance_matrix = (diff.T @ diff) / (len(current_deltas) - lagged_deltas.shape[1]) + np.eye(2) * 1e-6

        return self

    def sample(self, delta_len):
        delta_0 = self.__initial_deltas[np.random.randint(len(self.__initial_deltas))]
        deltas = [delta_0] # Generate using this seed
        for _ in range(delta_len - 1):
            # Sample the distribution
            noise = np.random.multivariate_normal(np.zeros_like(self.__intercept), self.__covariance_matrix)
            new_delta = self.__coefficients @ deltas[-1] + self.__intercept + noise
            deltas.append(new_delta)
        return np.array(deltas)
