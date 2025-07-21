import re
import sys
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class TweetPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for tweet-like posts.
    For now, this only removes "@" mentions.
    """

    def __init__(self):
        self.mention_regex = re.compile(r'@[\w.-]+')

    def fit(self, X, y=None):
        return self

    def _remove_mention(self, text: str) -> str:
        return self.mention_regex.sub('', text).strip()

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return self.preprocess_dataframe(X)
        elif isinstance(X, pd.Series):
            return self.preprocess_series(X)
        elif isinstance(X, np.ndarray):
            return self.preprocess_np(X)
        elif isinstance(X, list):
            return self.preprocess_list(X)
        else:
            print("Type: ",type(X), "not supported", file=sys.stderr)
            raise NotImplementedError

    def preprocess_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.select_dtypes(include=object).columns:
            X[col] = self.preprocess_series(X[col])
        return X

    def preprocess_series(self, X: pd.Series) -> pd.Series:
        return X.apply(self._remove_mention)

    def preprocess_np(self, X: np.ndarray) -> np.ndarray:
        data = np.empty(X.shape, dtype=str)
        for i in range(data.size):
            idx = np.unravel_index(i, X.shape)
            data[idx] = self._remove_mention(X[idx])
        return data

    def preprocess_list(self, X: list[str]) -> list[str]:
        return [self._remove_mention(text) for text in X]
