import torch
from torch.utils.data import TensorDataset
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
import pandas as pd


class Test:
    def __init__(self,task_type,*args, **kwargs) -> None:
        self.task_type = task_type
        if task_type == 'regression':
            self.clf = RegressionExperiment()
        elif task_type == 'classification':
            self.clf = ClassificationExperiment()
        else:
            raise ValueError(f"Unsupported task_type '{task_type}'. Use 'classification' or 'regression'.")

        self.clf.setup(data=self.df, target=self.target, normalize=True,
                       imputation_type='iterative', *args, **kwargs)

    def get_data(self):
        return self.df, self.target

    def train_numpy(self):
        return self.clf.X_train_transformed.values, self.clf.y_train_transformed.values

    def test_numpy(self):
        return self.clf.X_test_transformed.values, self.clf.y_test_transformed.values

    def train_tensor(self, device=None, dtype=torch.float32):
        return Test._tensor(self.train_numpy(), device, dtype)

    def test_tensor(self, device=None, dtype=torch.float32):
        return Test._tensor(self.test_numpy(), device, dtype)

    def train_dataset(self, device=None, dtype=torch.float32):
        return Test._dataset(self.train_numpy(), device, dtype)

    def test_dataset(self, device=None, dtype=torch.float32):
        return Test._dataset(self.test_numpy(), device, dtype)

    def evaluate_model(self, model):
        self.clf.evaluate_model(model)

    @staticmethod
    def _tensor(data, device=None, dtype=torch.float32):
        device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu')
        return list(map(lambda x: torch.from_numpy(x).to(device).to(dtype), data))

    @staticmethod
    def _dataset(data, device=None, dtype=torch.float32):
        data = Test._tensor(data, device, dtype)
        return TensorDataset(*data)

    @property
    def target_type(self):
        return self.task_type

