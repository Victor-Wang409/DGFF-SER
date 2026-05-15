"""
Main program module
Serves as the entry point managing execution flow and hyperparameter parsing
"""

import os
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import EmotionDataset
from trainer_executor import TrainerExecutor
from util import split_iemocap, split_msppodcast

def main():
    """
    Main function managing configuration setup and cross-validation execution
    """
    parser = argparse.ArgumentParser(description='Training VAD prediction model')
    parser.add_argument('--emotion2vec_dir', type=str, required=True, help='Directory containing emo2vec features')
    parser.add_argument('--hubert_dir', type=str, required=True, help='Directory containing hubert features')
    parser.add_argument('--wav2vec_dir', type=str, default=None, help='Directory containing wav2vec features')
    parser.add_argument('--data2vec_dir', type=str, default=None, help='Directory containing data2vec features')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=0.01)

    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step'], help='Type of learning rate scheduler')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='Step size for StepLR scheduler')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='Decay rate for StepLR scheduler')
    
    args = parser.parse_args()
    
    import random
    
    # Enforce strict determinism for reproducible deep learning experiments
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create target directory for model artifacts
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Configure unified logging subsystem
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.save_dir, 'training.log'))
        ]
    )

    logging.info(f"Actual batch size: {args.batch_size}")
    
    # Identify optimal computing device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Instantiate emotion dataset abstraction
    dataset = EmotionDataset(
        args.emotion2vec_dir, 
        args.hubert_dir, 
        args.csv_path,
        wav2vec_dir=args.wav2vec_dir,
        data2vec_dir=args.data2vec_dir
    )
    
    # Dynamically select dataset splitting strategy based on metadata naming
    csv_filename = os.path.basename(args.csv_path).lower()
    if 'msp' in csv_filename:
        logging.info("Using MSP-Podcast split strategy...")
        folds = split_msppodcast(dataset.df)
    else:
        logging.info("Using IEMOCAP 5-fold split strategy...")
        folds = split_iemocap(dataset.df)
        
    fold_results = []
    
    # Execute training loop across all dataset folds
    for fold in range(len(folds)):
        logging.info(f"\n{'='*50}\nFold {fold+1}/{len(folds)}\n{'='*50}")
        
        # [修复] 交叉验证严谨性：在每个 Fold 开始前重置随机种子。
        # 加上 fold 偏移量，既保证各个 Fold 之间的参数初始化不同，又保证多次运行实验的完全可复现。
        current_seed = args.seed + fold
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(current_seed)
            torch.cuda.manual_seed_all(current_seed)
        
        # Prepare dedicated directory for current fold artifacts
        fold_dir = os.path.join(args.save_dir, f'fold{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Extract indices specific to current fold
        fold_data = folds[fold]
        
        # Train model and append evaluation metrics
        fold_results.append(
            TrainerExecutor.train_model(
                args=args,
                fold=fold,
                fold_dir=fold_dir,
                dataset=dataset,
                train_idx=fold_data['train_idx'],
                eval_idx=fold_data['eval_idx'],
                test_idx=fold_data['test_idx'],
                device=device
            )
        )
    
    # Aggregate and statistical evaluation of cross-validation metrics
    avg_v = np.mean([res[0] for res in fold_results])
    avg_a = np.mean([res[1] for res in fold_results])
    avg_d = np.mean([res[2] for res in fold_results])
    avg_all = (avg_v + avg_a + avg_d) / 3
    
    std_v = np.std([res[0] for res in fold_results])
    std_a = np.std([res[1] for res in fold_results])
    std_d = np.std([res[2] for res in fold_results])
    
    final_results = (
        f"Final Cross-Validation Results\n"
        f"Average CCC ± std:\n"
        f"Valence: {avg_v:.3f} ± {std_v:.3f}\n"
        f"Arousal: {avg_a:.3f} ± {std_a:.3f}\n"
        f"Dominance: {avg_d:.3f} ± {std_d:.3f}\n"
        f"Overall VAD: {avg_all:.3f}"
    )
    
    logging.info(f"\n{'='*50}\n{final_results}\n{'='*50}")
    
    # Persist final statistical performance
    with open(os.path.join(args.save_dir, 'final_results.txt'), 'w') as f:
        f.write(final_results)

if __name__ == '__main__':
    main()
