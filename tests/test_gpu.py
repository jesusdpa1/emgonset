#%%
import torch
#%%
# Check if CUDA is available
print(torch.cuda.is_available())

# If available, you can also check additional GPU info
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
# %%
