import numpy as np
import numpy.linalg as la

class EssentialRegression:
    
    @staticmethod
    def fit_model(y, x, sigma, A_hat, Gamma_hat, I_hat):
        """
        Perform prediction on the data using the Essential Regression (ER) predictor.

        Arguments:
        y -- Response vector of length n (n,)
        x -- Data matrix of dimensions n x p (n x p)
        sigma -- Correlation matrix of dimensions p x p (p x p)
        A_hat -- Matrix of dimensions p x K (p x K)
        Gamma_hat -- Matrix of dimensions p x ? (p x ?)
        I_hat -- Matrix of dimensions p x ? (index set for subset of predictors)

        Returns:
        A dictionary containing:
        'er_predictor' -- The estimated coefficients theta_hat
        'pred_vals' -- The predicted values based on the model
        'theta_hat' -- The matrix R used in prediction
        """
        
        # Number of clusters (K), predictors (p), and observations (n)
        K = A_hat.shape[1]  # Number of clusters (columns in A_hat)
        p = x.shape[1]      # Number of predictors (columns in x)
        n = x.shape[0]      # Number of observations (rows in x)

        # Initialize matrix R (K x p) with zeros
        R = np.zeros((K, p))

        # Convert I_hat to numpy array if it isn't already
        I_hat = np.asarray(I_hat)
        
        # Solve for BI (Equation for parameter estimation, step 1)
        # We solve the linear system: (A_hat[I_hat, :]^T * A_hat[I_hat, :]) * X = (A_hat[I_hat, :]^T)
        # The result BI will be used later in calculating R
        try:
            BI = la.solve(np.dot(A_hat[I_hat, :].T, A_hat[I_hat, :]), A_hat[I_hat, :].T)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            BI = np.dot(la.pinv(np.dot(A_hat[I_hat, :].T, A_hat[I_hat, :])), A_hat[I_hat, :].T)
        
        # Step 2: Compute R matrix (K x p)
        # - For indices in I_hat, use the Gamma_hat adjustment
        # - For other indices, use the original correlation matrix sigma
        R[:, I_hat] = np.dot(BI, (sigma[np.ix_(I_hat, I_hat)] - np.diag(Gamma_hat[I_hat])))
        R[:, np.setdiff1d(np.arange(p), I_hat)] = np.dot(BI, sigma[np.ix_(I_hat, np.setdiff1d(np.arange(p), I_hat))])
        
        # Step 3: Compute Q matrix (n x K), which is used in theta_hat calculation
        Q = np.dot(x, R.T)
        
        # Step 4: Solve for theta_hat (coefficients for regression)
        # theta_hat = (R^T * (Q^T * Q)^-1 * (Q^T * y))
        try:
            theta_hat = np.dot(R.T, la.solve(np.dot(Q.T, Q), np.dot(Q.T, y)))
        except np.linalg.LinAlgError:
            # Fallback: Use Moore-Penrose pseudo-inverse in case of singular matrix
            theta_hat = np.dot(R.T, np.dot(la.pinv(np.dot(Q.T, Q)), np.dot(Q.T, y)))

        # Step 5: Predict values (Y_hat = X * theta_hat)
        pred_vals = np.dot(x, theta_hat)
        
        # Return the results as a dictionary
        return {
            'er_predictor': theta_hat,
            'pred_vals': pred_vals,
            'theta_hat': R.T
        }
