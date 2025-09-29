import torch
from torch import nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score 
import pandas as pd
 
class GRIFFIN(nn.Module):
    def __init__(self, in_features: int, rules: int, out_features: int, binary: bool, rank:int ,
                 regression: bool, zeta:float, Xi:float, eta:float, device=None, dtype=None):
        super().__init__()
        factory_kwargs = self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.rules_count = rules
        self.in_features = in_features
        self.out_features = out_features
        self.regression = regression
        self.binary = binary
        if binary:
            self.out_features = out_features = 1
        self.rank = rank
        self.device = device
        self.zeta = zeta
        self.Xi = Xi
        self.parameter_selector = nn.Parameter(torch.zeros((1, in_features), **factory_kwargs))
        self.s = nn.Parameter(torch.zeros((1, rules, rank), **factory_kwargs))
        self.V = nn.Parameter(torch.ones((1, rules, in_features, rank), **factory_kwargs))
        
        self.sigmoid = nn.Sigmoid()
        
        self.mean = nn.Parameter(torch.rand(
            (1, rules, in_features), **factory_kwargs))
        self.std = nn.Parameter(torch.rand(
            (1, rules, rank), **factory_kwargs))
        self.tsk_parameter = nn.Parameter(torch.rand(
            (1, rules, rank, out_features), **factory_kwargs))

        self.tsk_inner_bias = nn.Parameter(torch.rand((1, rules, out_features), **factory_kwargs))
        self.tsk_final_bias = nn.Parameter(torch.rand((1, out_features), **factory_kwargs))

    def encode(self, X):
        selector = torch.sigmoid(self.parameter_selector*self.Xi) # 1,f
        X = X * selector # b,f
        
        X = torch.unsqueeze(X, dim=1) #b, 1, f
        Z = X - self.mean #(b, 1, f) - (1, r, f) -> (b, r, f)
        Z = torch.unsqueeze(Z, dim=2) # (b, r, f) -> (b, r, 1, f) 
        Z = torch.matmul(Z, self.V) # (1, r, f, f) -> (b, r, 1, f)
        Z = torch.squeeze(Z, dim=2) # (b, r, f)

        y = torch.exp(-0.5 * torch.pow(Z / self.std, exponent=2)) # member function 

        epsilon = 1e-12  
        literal = self.sigmoid(self.eta * self.s)     
        y = (y * literal) + (1 - y) * (1 - literal)

        relaxer = self.sigmoid(self.s * self.zeta)    
        y = relaxer + (1 - relaxer) * y # (b, r, rank)
        
        y = torch.clamp(torch.prod(y,dim = 2) , min = epsilon)

        return Z, y

    def forward(self, X):
        Z, y = self.encode(X) # b, r
        
        if self.rules_count > 1 and not self.regression:
            y = F.normalize(y, p=1, dim=1)

        y = self.tsk(Z, y)

        return y

    def tsk(self, Z, y):
        # Z shape -> b, r, rank
        zeta = self.sigmoid(self.s / 4)  # 1, r, rank
        Z = (1 - zeta) * Z
        Z = torch.unsqueeze(Z, dim=2)  # (b, r, rank) -> (b, r, 1, rank) 
        Z = torch.matmul(Z, self.tsk_parameter)  # (b, r, 1, rank) @ (1, r, rank, o) -> (b, r, 1, o)
        Z = torch.squeeze(Z, dim=2)  # (b, r, o)
        Z = Z + self.tsk_inner_bias
        y = torch.unsqueeze(y, dim=2)  # b, r, 1
        Z = Z * y
        
        return Z.sum(dim=1) + self.tsk_final_bias

    def _init(self, X_train, y_train): 
        gmm = GaussianMixture(n_components=self.rules_count, covariance_type='full', random_state=24)
        X = np.concatenate([X_train, y_train[..., None]], axis=1)
        gmm.fit(X)  
        labels = gmm.predict(X)
        labels = torch.from_numpy(labels)
        X_train = torch.from_numpy(X_train)
        
        
        V = torch.ones(1, self.rules_count, self.in_features, self.rank)        
        mean = torch.rand((1, self.rules_count, self.in_features))
        std = torch.rand((1, self.rules_count, self.rank))

        for i in range(self.rules_count):
            cluster_points = X_train[labels == i]
            if cluster_points.shape[0] > 1:
                
                m = torch.mean(cluster_points, dim=0, keepdims=True)
                mean[:, i] = m
                cluster_points = cluster_points - m
                
                cov = cluster_points.T @ cluster_points
                # eigL, eigV = torch.linalg.eig(cov)
                # eigV = eigV.float()
                # V[0, i] = eigV
                # std[0, i, :len(eigL)] = eigL.float()
                eigL, eigV = torch.linalg.eig(cov)
                eigL = eigL.real
                eigV = eigV.real

                # Sort indices by eigenvalue magnitude descending
                topk = torch.topk(eigL, k=self.rank)
                idx = topk.indices

                # Select top-k eigenvectors and eigenvalues
                eigL = eigL[idx]
                eigV = eigV[:, idx]

                # Assign top components
                V[0, i, :, :self.rank] = eigV
                std[0, i, :self.rank] = eigL

                
        std[torch.abs(std) < 1e-9] = 1.0
        self.V = nn.Parameter(V)
        self.mean = nn.Parameter(mean)
        self.std = nn.Parameter(std)

        
    def _init_fexmax(self, X_train: np.ndarray, threshold: float = 1e-6, max_iter: int = 50):
        """
        Initialize fuzzy rules using FExMax clustering (Fuzzy Explanation Maximization).
        Inspired by Maximum Likelihood Estimation and Expectation Maximization.
        
        Args:
            X_train (np.ndarray): Training data (N x d).
            threshold (float): Variance threshold for pruning weak components.
            max_iter (int): Maximum number of iterations for convergence.
        """
        X = torch.from_numpy(X_train).float().to(self.device)

        N, d = X.shape
        R = self.rules_count

        # Random initialization
        M = torch.rand((R, d), device=self.device)   # centers
        Gamma = torch.eye(d, device=self.device).unsqueeze(0).repeat(R, 1, 1)  # eigvecs
        Delta = torch.eye(d, device=self.device).unsqueeze(0).repeat(R, 1, 1)  # eigvals diag

        for it in range(max_iter):
            # Compute membership μ (Eq. 46/53)
            mu = []
            for i in range(R):
                diff = X - M[i]
                Q_inv = torch.inverse(Gamma[i] @ Delta[i] @ Gamma[i].T + 1e-6*torch.eye(d, device=self.device))
                exponent = -0.5 * torch.sum(diff @ Q_inv * diff, dim=1)
                mu.append(torch.exp(exponent))
            mu = torch.stack(mu, dim=1)  # N x R

            # Normalize memberships
            mu = mu / (mu.sum(dim=1, keepdim=True) + 1e-12)

            # Update centers Mᶦ (Eq. 55)
            for i in range(R):
                weights = mu[:, i].unsqueeze(1)
                M[i] = (weights * X).sum(dim=0) / (weights.sum() + 1e-12)

                # Update covariance Qᶦ
                diff = X - M[i]
                cov = (weights * diff).T @ diff / (weights.sum() + 1e-12)

                eigvals, eigvecs = torch.linalg.eigh(cov)
                # Prune small variances
                mask = eigvals > threshold
                eigvals = eigvals[mask]
                eigvecs = eigvecs[:, mask]

                # Rebuild Γᶦ and Δᶦ
                Gamma[i] = eigvecs
                Delta[i] = torch.diag(eigvals)

        # Save parameters
        self.mean = nn.Parameter(M.unsqueeze(0))  # 1, R, d
        self.V = nn.Parameter(Gamma.unsqueeze(0)) # 1, R, d, r
        self.std = nn.Parameter(torch.stack([torch.diag(Delta[i]) for i in range(R)]).unsqueeze(0)) # 1,R,r
class SklearnGRIFFINrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, device=None, dtype=torch.float32):
        self.device = device if device else 'cpu'
        self.dtype = dtype
        # Initialize the model
        self.model = model.to(self.device)

    def fit(self, X, y):
        return self

    def predict(self, X):
        self._check_is_filiteraled()
        X = self._convert_to_tensor(X)

        # Use the model to get predictions
        with torch.no_grad():
            y_pred = self.model(X)

        if self.model.out_features == 1 and not self.model.regression:
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().numpy() > 0.5
        elif self.model.regression:
            return y_pred
        else:
            y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred.argmax(dim=1).cpu().numpy()
            
        return y_pred

    def predict_proba(self, X):
        self._check_is_filiteraled()
        X = self._convert_to_tensor(X)

        with torch.no_grad():
            y_pred = self.model(X)
        
        if self.model.out_features == 1:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = torch.softmax(y_pred, dim=1)
    
        return y_pred.cpu().numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def _convert_to_tensor(self, data):
        """ Helper function to convert numpy arrays to torch tensors and move to the correct device. """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)

        elif isinstance(data, torch.Tensor):
            data = data.to(self.device)

        elif isinstance(data, pd.DataFrame):
            data = torch.tensor(
                data.values, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(
                "Input data must be a NumPy array or a PyTorch tensor.")
        return data

    def _check_is_filiteraled(self):
        pass

    def get_params(self, deep=True):
        return {
            'model': self.model
        }

    def set_params(self, **parameters):
        return self

if __name__ == "__main__":
    griffin = GRIFFIN(3, 5, 2,Xi=0.5,eta=0.5,zeta=0.5,binary=True, rank=3,regression=True)
    X = np.random.randn(100, 3).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    
    griffin._init(X, y)
    X = torch.from_numpy(X)
    print(griffin(X).shape)
    
    