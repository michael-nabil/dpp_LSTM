import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold

# Import your classes and functions from the previous files
from dataset import VideoSummarizationDataset
from train import train_two_phase
from test import test_model

def run_5_fold_cv(h5_file_path, batch_size=1, n_splits=5, random_seed=42):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting {n_splits}-Fold Cross Validation on device: {device}")
    
    # 1. Find out how many videos are in the dataset
    with h5py.File(h5_file_path, 'r') as f:
        dataset_size = len(f.keys())
    
    # Create an array of indices [0, 1, 2, ..., N]
    indices = np.arange(dataset_size)
    
    # 2. Initialize the K-Fold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    # Instantiate the base datasets (using the optimized mode flags!)
    train_dataset = VideoSummarizationDataset(h5_file_path, mode='train')
    test_dataset = VideoSummarizationDataset(h5_file_path, mode='test')
    
    fold_results = []
    
    # 3. The Cross-Validation Loop
    for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
        fold_num = fold + 1
        print(f"\n{'='*40}")
        print(f"========== FOLD {fold_num}/{n_splits} ==========")
        print(f"{'='*40}")
        print(f"Train videos: {len(train_idx)} | Test videos: {len(test_idx)}")
        
        # Create DataLoaders specifically for this fold's indices
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx)
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx)
        )
        
        # Define fold-specific save paths so models don't overwrite each other
        phase1_save_path = f'./saved_models/vsLSTM_phase1_fold{fold_num}.pt'
        final_save_path = f'./saved_models/dppLSTM_final_fold{fold_num}.pt'
        
        # 4. Train a BRAND NEW model from scratch for this fold
        train_two_phase(
            train_loader, 
            device=device,
            phase1_save_path=phase1_save_path,
            final_save_path=final_save_path
        )
        
        # 5. Test the model on the unseen fold
        fold_f1_score = test_model(
            test_loader, 
            model_path=final_save_path, 
            device=device
        )
        
        fold_results.append(fold_f1_score)
        print(f"Fold {fold_num} Final F1-Score: {fold_f1_score * 100:.2f}%")
        
    # 6. Calculate the final, statistically rigorous average
    print(f"\n{'='*40}")
    print("CROSS-VALIDATION COMPLETE")
    for i, score in enumerate(fold_results):
        print(f"Fold {i+1}: {score * 100:.2f}%")
        
    final_average_f1 = np.mean(fold_results)
    print(f"FINAL OVERALL F1-SCORE: {final_average_f1 * 100:.2f}%")
    print(f"{'='*40}")

if __name__ == '__main__':
    kaggle_h5_path = '/kaggle/input/summe-video-summarization/eccv16_dataset_summe_google_pool5.h5'
    run_5_fold_cv(kaggle_h5_path)