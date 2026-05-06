"""
Training execution module
Governs the overall model training workflow and resource allocation
"""

import os
import logging
import torch
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from early_stopping import EarlyStopping
from lr_scheduler import LRSchedulerFactory
from data_processor import DataProcessor
from loss_functions import LossFactory
from model import VADConfig, VADModelWithGating
from trainer import TrainingManager

class TrainerExecutor:
    """
    Trainer executor responsible for driving the full training loop
    """
    @staticmethod
    def train_model(args, fold, fold_dir, dataset, train_idx, eval_idx, test_idx, device):
        """
        Train the deep learning model on a single cross validation fold
        """
        # Create randomized data loaders
        train_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(train_idx),
            collate_fn=DataProcessor.collate_fn
        )
        eval_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(eval_idx),
            collate_fn=DataProcessor.collate_fn
        )
        test_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(test_idx),
            collate_fn=DataProcessor.collate_fn
        )
        
        # Configure model architecture
        config = VADConfig(
            emotion2vec_dim=1024,
            hubert_dim=1024,
            hidden_dim=1024,
            num_hidden_layers=4,
            num_groups=8,
            # Assign zero to disable supplementary feature dimensions
            wav2vec_dim=0,
            data2vec_dim=0
        )

        # Instantiate neural network
        model = VADModelWithGating(config).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        
        # Initialize learning rate scheduler and loss criteria
        scheduler = LRSchedulerFactory.create_scheduler(optimizer, args)
        vad_criterion = LossFactory.CCCLoss()
        contrast_criterion = LossFactory.SupervisedContrastiveLoss(temperature=0.1)
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        
        best_val_ccc = -float('inf')
        best_model = None
        
        # Establish performance tracking log
        metrics_file = os.path.join(fold_dir, 'metrics.csv')
        with open(metrics_file, 'w') as f:
            f.write('epoch,train_loss,val_ccc_v,val_ccc_a,val_ccc_d,val_ccc_avg\n')
        
        for epoch in range(args.epochs):
            # Execute one training epoch
            metrics = TrainingManager.train_one_epoch(
                model, 
                optimizer, 
                vad_criterion,
                contrast_criterion, 
                train_loader, 
                device
            )
            train_loss = metrics['loss']
            
            # Step learning rate scheduler
            scheduler.step()
                    
            # Record current learning rate
            current_lr = scheduler.get_last_lr()[0]
            logging.info(f"Epoch {epoch+1}")
            logging.info(f"Current learning rate: {current_lr:.2e}")
            
            # Evaluate on validation subset
            val_v, val_a, val_d = TrainingManager.validate_and_test(model, eval_loader, device)
            val_ccc_avg = (val_v + val_a + val_d) / 3
            
            # Prepare epoch specific output directory
            epoch_dir = os.path.join(fold_dir, f'epoch{epoch+1}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Serialize model state and configuration
            model.save_pretrained(epoch_dir, safe_serialization=False)
            
            # Serialize optimizer parameters
            torch.save(optimizer.state_dict(), os.path.join(epoch_dir, 'optimizer.pt'))
            
            # Append epoch metrics to tracking file
            with open(metrics_file, 'a') as f:
                f.write(f'{epoch+1},{train_loss:.4f},{val_v:.4f},{val_a:.4f},{val_d:.4f},{val_ccc_avg:.4f}\n')
            
            logging.info(
                f"Fold {fold+1}, Epoch {epoch+1:3d} | "
                f"Loss: {train_loss:.4f} | "
                f"Val CCC: V={val_v:.3f}, A={val_a:.3f}, D={val_d:.3f} | "
                f"Avg={val_ccc_avg:.3f}"
            )
            
            # Update best performing model state
            if val_ccc_avg > best_val_ccc:
                best_val_ccc = val_ccc_avg
                best_model = model.state_dict()
                # Serialize optimal model snapshot
                best_model_dir = os.path.join(fold_dir, 'best_model')
                os.makedirs(best_model_dir, exist_ok=True)
                model.save_pretrained(best_model_dir, safe_serialization=False)
                logging.info(f"Saved new best model with val_ccc={val_ccc_avg:.3f}")
            
            # Serialize training checkpoint for potential resumption
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_ccc': best_val_ccc,
                'config': config.to_dict()
            }
            torch.save(checkpoint, os.path.join(fold_dir, 'checkpoint.pt'))
            
            # Check early stopping criterion
            early_stopping(1 - val_ccc_avg)
            if early_stopping.early_stop:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load optimal model for final test evaluation
        model.load_state_dict(best_model)
        test_v, test_a, test_d = TrainingManager.validate_and_test(model, test_loader, device)
        test_ccc_avg = (test_v + test_a + test_d) / 3
        
        # Save absolute test results
        with open(os.path.join(fold_dir, 'test_results.txt'), 'w') as f:
            f.write(f"Test CCC:\nValence: {test_v:.3f}\nArousal: {test_a:.3f}\n"
                   f"Dominance: {test_d:.3f}\nAverage: {test_ccc_avg:.3f}")
        
        return test_v, test_a, test_d