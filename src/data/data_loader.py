"""
Dynamic data loader implementation for handling changing batch sizes.
"""

import torch

class DynamicBatchSizeDataLoader:
    """
    Custom DataLoader class for dynamic batch size.
    Allows changing batch size during training/evaluation.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.dataloader = self._create_dataloader()
        self.iterator = None
        
    def _create_dataloader(self):
        """Create internal PyTorch DataLoader"""
        return torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last
        )
    
    def set_batch_size(self, new_batch_size):
        """Change batch size"""
        if new_batch_size == self.batch_size:
            return False  # No change

        print(f"[DynamicDataLoader] Batch size changed: {self.batch_size} -> {new_batch_size}")
        self.batch_size = new_batch_size
        
        # If existing iterator, close it
        if self.iterator is not None:
            try:
                self.dataloader._iterator.close()
            except:
                pass
            self.iterator = None
        
        # Create new DataLoader
        self.dataloader = self._create_dataloader()
        return True  # Batch size changed
        
    def __iter__(self):
        """Return DataLoader iterator"""
        self.iterator = iter(self.dataloader)
        return self.iterator
    
    def __len__(self):
        """Return DataLoader length"""
        return len(self.dataloader)