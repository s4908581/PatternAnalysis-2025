""" 
This module handles the loading and data augmentation of ADNI dataset 
With reference to GavinSaiun's work: https://github.com/GavinSaiun/GFNet-Alzheimer-Detection 
"""

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
import os
import torchvision.transforms as transforms

# Paths to the ADNI dataset
BASE_DIR = Path(__file__).parent
ADNI_ROOT_PATH = BASE_DIR / 'ADNI'

# Data augmentation and normalization for training
# Including random rotations, resized crops, and color jittering
# This helps improve model generalization
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(size=224),
    transforms.ColorJitter(brightness=(0.8, 1.2)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0062], std=[0.0083])
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0062], std=[0.0083])
])

class ADNIDataset(Dataset):
    """ Custom Dataset for loading ADNI images and labels. """


    def __init__(self, root_dir, train=True, transform=None):
        """ Initializes the dataset by loading image paths and labels."""
        """
        Args:
            root_dir (str or Path): Directory with all the images.
            train (bool): If True, loads training data; otherwise, loads test data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.root_dir = Path(root_dir, 'train' if train else 'test')
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load AD (Alzheimer's Disease) class images and labels
        ad_path = self.root_dir / 'AD'
        ad_files = os.listdir(ad_path)
        self.image_paths.extend([ad_path / file for file in ad_files])
        self.labels.extend([1] * len(ad_files))  
      
        # Load NC (Normal Control) class images and labels
        nc_path = self.root_dir / 'NC'
        nc_files = os.listdir(nc_path)
        self.image_paths.extend([nc_path / file for file in nc_files])
        self.labels.extend([0] * len(nc_files))  

        print(f"Checking dataset paths...")
        print(f"AD path exists: {ad_path.exists()}, files: {len(ad_files)}")
        print(f"NC path exists: {nc_path.exists()}, files: {len(nc_files)}")
        if len(ad_files) == 0 or len(nc_files) == 0:
            raise RuntimeError("No images found in dataset directories")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Total number of samples.
        """

        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label by index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (image, label) where image is the transformed image tensor and label is its class label.
        """

        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open image in grayscale
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)

        return image, label

def get_adni_dataloader(batch_size, train=True, val_split=0.2, num_workers=4):
    """
    Creates DataLoader for the ADNI dataset with training/validation split or test set.

    Args:
        batch_size (int): Number of samples per batch.
        train (bool): If True, returns training and validation DataLoaders; otherwise, returns test DataLoader.
        val_split (float): Proportion of training data to use for validation.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        If train is True:
            tuple: (train_loader, val_loader) DataLoaders for training and validation sets.
        If train is False:
            DataLoader: DataLoader for the test set.
    """
    
    if train:
        # Create full training dataset
        full_dataset = ADNIDataset(root_dir=ADNI_ROOT_PATH, train=True, transform=TRAIN_TRANSFORM)
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        # Split dataset into training and validation sets
        # The validation set helps monitor model performance on unseen data during training
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        # Create DataLoaders for training and validation sets
        # DataLoaders handle batching, shuffling, and parallel loading of data
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, # Batch size stands for number of samples per batch
            shuffle=True, # Shuffle means shuffle training data for better generalization (reduce overfitting)
            num_workers=num_workers, # Number of subprocesses to use for data loading, helps speed up data loading
            pin_memory=True,  # Pin memory enables faster data transfer to GPU
            persistent_workers=True # Keeps workers alive between epochs for efficiency
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        return train_loader, val_loader
    
    else:
        # Create test dataset and DataLoader
        test_dataset = ADNIDataset(root_dir=ADNI_ROOT_PATH, train=False, transform=TEST_TRANSFORM)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        return test_loader
