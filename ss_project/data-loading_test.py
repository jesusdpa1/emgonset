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

from emgonset.processing.filters import create_bandpass_filter, create_notch_filter
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
ref_fs = ref_.fs
emg_fs = emg_.fs
print(f"ref_fs: {ref_fs}, emg_fs: {emg_fs}")
ref_data = ref_.load_data()
emg_data = emg_.load_data()
# %%
print(f"{ref_data.shape}")
print(f"{emg_data.shape}")
# %%
img_ref = plot_multi_channel_data(ref_data, time_window=[0, 1], fs=ref_fs)

# %%
img_emg = img = plot_multi_channel_data(emg_data, time_window=[0, 1], fs=emg_fs)

# %%
notch_filter = create_notch_filter(60)
notch_filter.initialize(emg_fs)
bp_filter = create_bandpass_filter(20, 2000)
bp_filter.initialize(emg_fs)
# %%
# filter notch
notch_ref = notch_filter(ref_data)
bp_ref = bp_filter(notch_ref)

# filter bandpass
notch_emg = notch_filter(emg_data)
bp_emg = bp_filter(notch_emg)

# %%
img_ref_filtered = plot_multi_channel_data(bp_ref, time_window=[0, 1], fs=ref_fs)

img_emg_filtered = plot_multi_channel_data(bp_emg, time_window=[0, 1], fs=emg_fs)

# %%
