"""
Learning rate scheduler module
Responsible for dynamic optimization step size adjustment
"""

from torch import optim

class LRSchedulerFactory:
    """
    Factory creating diverse learning rate decay policies
    """
    @staticmethod
    def create_scheduler(optimizer, args):
        """
        Instantiate concrete learning rate scheduler from parsed arguments
        """
        if args.lr_scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.lr_decay_step,
                gamma=args.lr_decay_rate
            )
        else:
            raise ValueError(f"Unknown scheduler type: {args.lr_scheduler}")