import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, LogisticRegression


class Estimator():
    def __init__(self, model, scaler):
        if model == 'linear':
            self.model = LinearRegression()
        elif model == 'logistic':
            self.model = LogisticRegression()
        else:
            raise ValueError(f"Invalid model: {model}")
        
        self.scaler = scaler
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def train_test_split(self, X, y, test_size=0.2, seed=1334):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed)
        return X_train, X_test, y_train, y_test

    def score(self, yhat, y):
        yhat = [1 if i >= 0.5 else 0 for i in yhat]
        if len(np.unique(y)) == 1:
            return None
        auc = roc_auc_score(y, yhat)
        return auc

    @staticmethod
    def scale_features(X, scaler, feature_range=(-1, 1)):
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if scaler == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif scaler == 'standard':
            scaler = StandardScaler()
        else:
            return X
            
        scaler.fit(X)
        return scaler.transform(X)

    def evaluate(self, X, y, n_iters=10, test_size=0.2):
        scores = [] 
        X = X.copy()

        X = self.scale_features(X, scaler=self.scaler)

        for iter in range(n_iters):
            X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size, seed=iter)
            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)
            scores.append(self.score(y_pred, y_test))
            
        return np.array(scores)
    