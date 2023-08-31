import numpy as np
from itertools import product
from sklearn.model_selection import RepeatedKFold


class ModelTuner:
    def __init__(self, model_func, params, n_splits, n_repeats):
        self.model_func = model_func
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        keys, values = zip(*params.items())
        self.experiments = [dict(zip(keys, v)) for v in product(*values)]

    def cv_model(self, model_specs, X, y, epochs, random_state=42):
        losses = []
        cv = RepeatedKFold(n_splits=self.n_splits, 
                        n_repeats=self.n_repeats, 
                        random_state=random_state)
        splits = cv.split(X)
        for i, (train_ix, test_ix) in enumerate(splits):
            X_train, X_test = X[train_ix], X[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
            model = self.model_func(X.shape[1:], **model_specs)
            model.fit(X_train, y_train, epochs=epochs, verbose=0)
            loss = model.evaluate(X_test, y_test, verbose=0)
            losses.append(loss)
            print(f'Fold {i} loss = {loss}')
        print(f'\nMean loss = {np.mean(losses): .4f} STD = {np.std(losses): .4f}')
        return np.mean(losses), model

    def fit(self, X, y, epochs):
        print(f'Running {len(self.experiments)} experiments...')
        self.best_loss_ = float('INF')
        for i in range(len(self.experiments)):
            print(f'\nExperiment {i + 1}\n{self.experiments[i]}')
            loss, model = self.cv_model(self.experiments[i],
                                        X,
                                        y,
                                        epochs)
            if loss < self.best_loss_:
                self.best_loss_ = loss
                self.best_model_ = model
                self.best_params_ = self.experiments[i]
        print(self.best_model_.summary())
        return self.best_params_