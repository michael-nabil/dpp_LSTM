import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# added "mode" parameter in the VideoSummarizationDataset class, to load
# this aditional data: (change_points, n_frames, user_summary), during testing only

class VideoSummarizationDataset(Dataset):
    def __init__(self, h5_file_path, mode='train'):
        """
        Args:
            h5_file_path: Path to the .h5 dataset.
            mode: 'train' or 'test'. Dictates what arrays are loaded from disk.
        """
        self.h5_file = h5py.File(h5_file_path, 'r')
        # The Kaggle dataset uses 'video_1', 'video_2', etc. as main keys
        self.video_keys = list(self.h5_file.keys())
        self.mode = mode

    def __len__(self):
        return len(self.video_keys)

    def __getitem__(self, idx):
        video_id = self.video_keys[idx]
        
        # 1. Always load the core arrays needed for a forward pass
        # Extract using the Kaggle dataset's specific naming convention, and convert them into Pytorch tensors
        features = torch.tensor(self.h5_file[video_id]['features'][...], dtype=torch.float32)
        gt_score = torch.tensor(self.h5_file[video_id]['gtscore'][...], dtype=torch.float32)
        gt_summary = torch.tensor(self.h5_file[video_id]['gtsummary'][...], dtype=torch.float32)
        
       # 2. If in test mode, incur the extra I/O cost to load evaluation metadata
        if self.mode == 'test':
            change_points = self.h5_file[video_id]['change_points'][...]
            n_frames = self.h5_file[video_id]['n_frames'][()]
            user_summary = self.h5_file[video_id]['user_summary'][...]
            
            return features, gt_score, gt_summary, change_points, n_frames, user_summary, video_id
        
        # 3. If in train mode, return early and save time!, don't return video_id
        return features, gt_score, gt_summary

def get_dataloaders(h5_file_path, batch_size=1, test_split=0.2, random_seed=42):
    """
    Creates the dynamic train/test splits from the single H5 file.
    """
    # Briefly open the file just to get the total number of videos
    with h5py.File(h5_file_path, 'r') as f:
            dataset_size = len(f.keys())

    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))

    # Shuffle indices to randomize the train/test split
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    train_indices, test_indices = indices[split:], indices[:split]
    
    # Instantiate TWO separate dataset objects with different modes
    train_dataset = VideoSummarizationDataset(h5_file_path, mode='train')
    test_dataset = VideoSummarizationDataset(h5_file_path, mode='test')

    # Create DataLoaders
    # Note: batch_size is usually 1 for video summarization because 
    # each video has a different sequence length (number of frames)
    train_loader = DataLoader(train_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices))
    
    test_loader = DataLoader(test_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices))
    
    return train_loader, test_loader