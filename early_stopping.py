"""
Early stopping module
Implements early termination to prevent model overfitting during iterative optimization
"""

class EarlyStopping:
    """
    Early stopping mechanism terminating optimization when generalization stops improving
    """
    def __init__(self, patience=10, min_delta=0):
        """
        Initialize early stopping criteria
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        """
        Evaluate current validation loss against historical best to update termination state
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0