from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Common interface for all models.
    """

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass