from itertools import combinations
from sklearn.model_selection import cross_val_score
import numpy as np
from math import factorial


class DropColumnsCV:
    def __init__(self, model, df, y, scaler, scoring, max_k, cv=5):
        self.model = model
        self.df = df
        self.y = y
        self.scaler = scaler
        self.scoring = scoring
        self.max_k = max_k
        self.cv = cv

    def drop_columns(self):
        columns = self.df.columns
        for k in range(1, self.max_k + 1):
            combos = combinations(columns, k)
            for combo in combos:
                new_df = self.df.drop(columns=list(combo))
                yield new_df

    def eval_combinations(self):
        n = self.df.shape[1]
        n_cands = sum([factorial(n) / (factorial(k) * factorial(n - k)) for k in range(1, self.max_k +  1)])
        print(f'Fitting {self.cv} folds over {int(n_cands)} candidates, total {self.cv * int(n_cands)}\n')
        drop = self.drop_columns()
        experiment = 1
        self.best_score_ = -float('INF')
        while True:
            try:
                sub_df = next(drop)
            except StopIteration:
                break
            print(f'Candidate {experiment}\n{sub_df.columns}')
            experiment += 1

            if self.scaler is not None:
                X = self.scaler.fit_transform(sub_df)
            else:
                X = sub_df

            n_scores = cross_val_score(
                self.model, 
                X, 
                self.y, 
                cv=self.cv,
                scoring=self.scoring,
                verbose=0,
                n_jobs=-1
            )

            score = np.mean(n_scores)
            if score > self.best_score_:
                self.best_score_ = score
                self.best_features_ = sub_df.columns
            print(f'Mean score: {abs(score): .4f} STD: {np.std(np.absolute(n_scores)): .4f}\n')
        print(f'Best features: {self.best_features_}\nBest score: {self.best_score_: .4f}')

        