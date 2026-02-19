import numpy as np

class EnsembleKalmanFilter:
    """
    Implements the Ensemble Kalman Filter (EnKF).
    """
    def __init__(self, model, R, N):
        """
        Initializes the EnKF.

        Args:
            model (callable): The forward model function.
            R (np.ndarray): The observation noise covariance matrix.
            N (int): The number of ensemble members.
        """
        self.model = model
        self.R = R
        self.N = N

    def forecast(self, X, t_span, t_eval, *model_args):
        """
        Propagates the ensemble forward in time using the model.

        Args:
            X (np.ndarray): The ensemble of state vectors (shape: dim_x, N).
            t_span (tuple): Time interval for integration.
            t_eval (np.ndarray): Time points for evaluation.
            *model_args: Additional arguments for the model (e.g., sigma, rho, beta).

        Returns:
            np.ndarray: The forecast ensemble at the last time step.
        """
        X_f = np.empty_like(X)
        for i in range(self.N):
            sol = self.model(X[:, i], t_span, t_eval, *model_args)
            X_f[:, i] = sol.y[:, -1]
        return X_f

    def analysis(self, X_f, y, H):
        """
        Updates the forecast ensemble with an observation.

        Args:
            X_f (np.ndarray): The forecast ensemble (shape: dim_x, N).
            y (np.ndarray): The observation vector.
            H (np.ndarray): The observation operator.

        Returns:
            np.ndarray: The analysis ensemble.
        """
        # Forecast mean and covariance
        x_f_mean = np.mean(X_f, axis=1)
        P_f = np.cov(X_f)

        # Kalman gain
        K = P_f @ H.T @ np.linalg.inv(H @ P_f @ H.T + self.R)

        # Analysis update
        X_a = X_f + K @ (y[:, np.newaxis] - H @ X_f + np.random.multivariate_normal(np.zeros(len(y)), self.R, self.N).T)
        
        return X_a