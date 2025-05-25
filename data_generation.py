import numpy as np

class Data_Generator():

    def __init__(self):

        pass

    def Lorenz_attractor(self,T, dt):
        """
        Integrates the Lorenz attractor equations using the forward Euler method.

        Args:
            T (float): Total simulation time.
            dt (float): Time step for integration.
            initial_state (array-like, optional): Initial conditions [x0, y0, z0].
                                                 If None, a random initial state is used.
            sigma (float, optional): Lorenz system parameter. Defaults to 10.
            rho (float, optional): Lorenz system parameter. Defaults to 28.
            beta (float, optional): Lorenz system parameter. Defaults to 8/3.

        Returns:
            numpy.ndarray: A NumPy array of shape (N, 3) containing the trajectory
                           of the Lorenz attractor, where N is the number of time steps.
        """

        sigma = 10
        rho = 28
        beta = 8 / 3
        N = int(T / dt)
        positions = np.zeros((N, 3))

        positions[0] = 2 * np.random.random(3) - 1  # Random in [-1, 1) for each dimension


        for i in range(N - 1):
            x = positions[i, 0]
            y = positions[i, 1]
            z = positions[i, 2]

            dxdt = sigma * (y - x)
            dydt = rho * x - y - x * z
            dzdt = x * y - beta * z

            positions[i + 1, 0] = x + dt * dxdt
            positions[i + 1, 1] = y + dt * dydt
            positions[i + 1, 2] = z + dt * dzdt

        return positions

    def sequences(self, x, input_len, pred_len):
        """
        Generates training feature vector and the output from a longer timeseries

        :param x: complete position timeseries
        :param input_len: length of the feature vector
        :param pred_len: length of the output(ground truth) vector
        :return: features and output arrays
        """

        input_seq = []
        pred_seq = []

        for i in range(len(x) - input_len - pred_len):

            input_seq.append(x[i:i+input_len])
            pred_seq.append(x[i+input_len : i+input_len+pred_len])

        return np.array(input_seq), np.array(pred_seq)

    def normalize(self, X, y):
        """
        Normalizes the feature and output vectors

        :param X: feature vector
        :param y: output vector
        :return: normalized arrays
        """
        X = (X - X.mean()) / (X.std())
        y = (y - y.mean()) / (y.std())

        return X, y








