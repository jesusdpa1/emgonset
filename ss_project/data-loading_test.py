# %%
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dotenv
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy import signal
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from emgonset.processing.rectifiers import create_abs_rectifier
from emgonset.processing.tkeo import create_tkeo2
from emgonset.processing.transforms import EMGTransformCompose
from emgonset.utils.io import (
    EMGData,
    create_emg_dataloader,  # Corrected import
)
from emgonset.visualization.general_plots import plot_multi_channel_data

dotenv.load_dotenv()
# %%
data_dir = Path(os.getenv("DATA_DIR"))
data_path = data_dir.joinpath(
    r"becca\drv_00_baseline_25-02-26_9881-2_testSubject_topoMapping"
)

reference_channel_path = data_path.joinpath(r"referenceChannel.ant")
emg_path = data_path.joinpath(r"RawG.ant")
# %%

ref_ = EMGData(reference_channel_path)
emg_ = EMGData(emg_path)
# %%
fs_ref = ref_.fs
fs_emg = emg_.fs
print(f"ref_fs: {fs_ref}, emg_fs: {fs_emg}")
ref_data = ref_.load_data()
emg_data = emg_.load_data()
# %%
print(f"{ref_data.shape}")
print(f"{emg_data.shape}")
# %%
img_ref = plot_multi_channel_data(ref_data, time_window=[0, 1], fs=fs_emg)
# %%
img_emg = img = plot_multi_channel_data(emg_data, time_window=[0, 1], fs=fs_emg)
# %%
