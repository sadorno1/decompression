from scipy.io import loadmat, savemat
import numpy as np
import glob

all_in, all_out = [], []
for f in sorted(glob.glob("data/eager1_log_delay_*.mat")):
    d = loadmat(f)
    all_in.append(d["train_input_real"].ravel())
    all_out.append(d["train_output_real"].ravel())

combined_in  = np.concatenate(all_in)
combined_out = np.concatenate(all_out)

savemat("data/compression_data_logdelay.mat",
        {"train_input_real":  combined_in,
         "train_output_real": combined_out})
print(f"Saved: {len(combined_in):,} samples")