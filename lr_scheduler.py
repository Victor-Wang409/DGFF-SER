"""
学习率调度器模块
用于创建和管理各种学习率调度策略
"""

from torch import optim

class LRSchedulerFactory:
    """
    学习率调度器工厂，用于创建各种学习率调度策略
    """
    @staticmethod
    def create_scheduler(optimizer, args):
        """
        创建学习率调度器
        
        参数:
            optimizer: 优化器
            args: 参数配置
        
        返回:
            学习率调度器
        """
        if args.lr_scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.lr_decay_step,
                gamma=args.lr_decay_rate
            )
        else:
            raise ValueError(f"Unknown scheduler type: {args.lr_scheduler}")
