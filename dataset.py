import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class VideoSummarizationDataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file = h5py.File(h5_file_path, 'r')
        # The Kaggle dataset uses 'video_1', 'video_2', etc. as main keys
        self.video_keys = list(self.h5_file.keys())
        
    def __len__(self):
        return len(self.video_keys)

    def __getitem__(self, idx):
        video_id = self.video_keys[idx]
        
        # Extract using the Kaggle dataset's specific naming convention
        features = self.h5_file[video_id]['features'][...]
        gt_score = self.h5_file[video_id]['gtscore'][...]
        gt_summary = self.h5_file[video_id]['gtsummary'][...]
        
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
    # each video has a different sequence length (number of frames)
    train_loader = DataLoader(dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices))
    
    test_loader = DataLoader(dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices))
    
    return train_loader, test_loader