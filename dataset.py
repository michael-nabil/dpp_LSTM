import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class VideoSummarizationDataset(Dataset):
    def __init__(self, h5_file_path):
        """
        Args:
            h5_file_path (str): Path to the Kaggle .h5 dataset file.
        """
        self.h5_file = h5py.File(h5_file_path, 'r')
        
        # The original dataset stores the video indices in an array called 'idx'
        # e.g., [1, 2, 3, ..., 25] for SumMe
        self.video_indices = self.h5_file['idx'][...]
        
    def __len__(self):
        return len(self.video_indices)

    def __getitem__(self, idx):
        # h5py keys are strings, and the video indices usually start at 1
        video_id = str(self.video_indices[idx])
        
        # Extract the arrays using the original author's naming convention
        features = self.h5_file[f'fea_{video_id}'][...]
        gt_score = self.h5_file[f'gt_1_{video_id}'][...]
        gt_summary = self.h5_file[f'gt_2_{video_id}'][...]
        
        # Convert to PyTorch tensors
        features = torch.tensor(features, dtype=torch.float32)
        gt_score = torch.tensor(gt_score, dtype=torch.float32)
        gt_summary = torch.tensor(gt_summary, dtype=torch.float32)
        
        return features, gt_score, gt_summary

def get_dataloaders(h5_file_path, batch_size=1, test_split=0.2, random_seed=42):
    """
    Creates the dynamic train/test splits from the single H5 file.
    """
    dataset = VideoSummarizationDataset(h5_file_path)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    
    # Shuffle indices to randomize the train/test split
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    train_indices, test_indices = indices[split:], indices[:split]
    
    # Create DataLoaders
    # Note: batch_size is usually 1 for video summarization because 
    # each video has a different sequence length (number of frames).
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=SubsetRandomSampler(train_indices)
    )
    
    test_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=SubsetRandomSampler(test_indices)
    )
    
    return train_loader, test_loader