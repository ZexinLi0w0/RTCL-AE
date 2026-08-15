"""
Model scheduler for sharing model states between processes.
"""

import torch

class ModelScheduler:
    """
    Memory-based lock-free double-buffer scheduler.
    Uses PyTorch's shared memory features to efficiently share model states between processes.
    """
    def __init__(self, shared_data, use_double_buffer=False):
        self.shared_data = shared_data
        self.use_double_buffer = use_double_buffer
        
        # Flag to track if we're using shared memory tensors
        if "using_shared_memory" not in self.shared_data:
            self.shared_data["using_shared_memory"] = use_double_buffer
            
        if use_double_buffer:
            # Double-buffer with shared memory approach
            if "latest" not in self.shared_data:
                self.shared_data["latest"] = "A"
            
            # Initialize the buffer indices as empty
            if "bufA_keys" not in self.shared_data:
                self.shared_data["bufA_keys"] = []
            if "bufB_keys" not in self.shared_data:
                self.shared_data["bufB_keys"] = []
        else:
            # For single buffer approach
            if "single_buf" not in self.shared_data:
                self.shared_data["single_buf"] = None

    def write_state_dict(self, model_state_dict):
        """
        Writes the model state to the shared memory.
        
        Args:
            model_state_dict: Model state dictionary to be shared
        """
        if self.use_double_buffer:
            # Select the inactive buffer
            current_latest = self.shared_data["latest"]  # "A" or "B"
            next_buf = "B" if current_latest == "A" else "A"
            
            # Get all keys from the inactive buffer
            old_keys = list(self.shared_data.get(f"buf{next_buf}_keys", []))
            
            # Clean up existing tensors in the inactive buffer
            for key in old_keys:
                buffer_key = f"buf{next_buf}_{key}"
                if buffer_key in self.shared_data:
                    del self.shared_data[buffer_key]
            
            # New key list
            new_keys = []
            
            # Copy each tensor of the model state to the shared memory
            for key, tensor in model_state_dict.items():
                # If the key is too long, replace it with a hash value to save memory
                short_key = str(hash(key))
                new_keys.append(short_key)
                
                # Move the tensor to CPU and save it to the shared memory
                cpu_tensor = tensor.cpu()
                buffer_key = f"buf{next_buf}_{short_key}"
                
                # Large tensors are compressed (convert float32 to float16 to save memory)
                if cpu_tensor.element_size() * cpu_tensor.nelement() > 1024*1024:  # if larger than 1MB
                    if cpu_tensor.dtype == torch.float32:
                        cpu_tensor = cpu_tensor.half()  # compress to float16 to save memory
                
                # Save the tensor to the shared memory
                self.shared_data[buffer_key] = cpu_tensor
                
            # Save the new key list
            self.shared_data[f"buf{next_buf}_keys"] = new_keys
            
            # Save the key-original name mapping
            key_mapping = {str(hash(k)): k for k in model_state_dict.keys()}
            self.shared_data[f"buf{next_buf}_mapping"] = key_mapping
            
            # Update the active buffer (perform last)
            self.shared_data["latest"] = next_buf
            print(f"[ModelScheduler] Model state updated to memory buffer {next_buf}")
        else:
            # Use single buffer as usual
            self.shared_data["single_buf"] = {"model_state_dict": model_state_dict}

    def read_state_dict(self):
        """
        Reads the model state from the shared memory.
        
        Returns:
            Model state dictionary if available, None otherwise
        """
        if self.use_double_buffer:
            # Read from the active buffer
            current_buf = self.shared_data["latest"]
            keys = self.shared_data.get(f"buf{current_buf}_keys", [])
            
            # If there are no keys, the model hasn't been saved yet
            if not keys:
                return None
                
            # Get the key-original name mapping
            key_mapping = self.shared_data.get(f"buf{current_buf}_mapping", {})
            
            # Collect each tensor to restore the state_dict
            state_dict = {}
            for short_key in keys:
                buffer_key = f"buf{current_buf}_{short_key}"
                if buffer_key in self.shared_data:
                    tensor = self.shared_data[buffer_key]
                    
                    # If needed, decompress (half -> float32)
                    if tensor.dtype == torch.float16:
                        tensor = tensor.float()  # restore to float32
                        
                    # Restore the original key name
                    original_key = key_mapping.get(short_key, short_key)
                    state_dict[original_key] = tensor
            
            # Return the state_dict in the usual format
            return {"model_state_dict": state_dict}
        else:
            # Use single buffer as usual
            return self.shared_data.get("single_buf", None)