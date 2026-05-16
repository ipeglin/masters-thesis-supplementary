import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lib.fs.get_script_name import get_name
from plotting import plot_config
from plotting.plot_config import get_figs_output_dir

FILE_DIR = Path("/Users/ipeglin/Documents/masters_thesis/bids_processed_consolidated_data")
# FILE_DIR = Path("/Volumes/work/bids_processed_consolidated_data")  # IDUN

SUBJECT_ID = "sub-NDARINVAG388HJL"
FILE_SUFFIX_REST = "_task-restAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.h5"
FILE_PATH_REST = FILE_DIR / SUBJECT_ID / f"{SUBJECT_ID}{FILE_SUFFIX_REST}"

DATASET_PATH = "01fmri_parcellation/full_run_std"
TR = 0.8  # Repetition time (seconds)

def plot_multichannel_heatmap(file_path: Path):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    with h5py.File(file_path, "r") as f:
        if DATASET_PATH not in f:
            print(f"Dataset {DATASET_PATH} not found.")
            return
        
        # Shape usually (channels, timepoints)
        ts = np.asarray(f[DATASET_PATH]) 

    n_channels, n_time = ts.shape

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    time_axis = np.arange(n_time) * TR

    # Center colormap around 0
    vmax = np.max(np.abs(ts))
    vmin = -vmax

    im = ax.imshow(
        ts,
        aspect="auto",
        origin="lower", # Channel 0 at bottom
        cmap="RdBu_r",
        extent=[time_axis[0], time_axis[-1], 0, n_channels],
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(f"Multichannel fMRI Timeseries\n{SUBJECT_ID} (resting state)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Amplitude (Z-score)")

    out_dir = get_figs_output_dir() / get_name()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / f"multichannel_heatmap_{SUBJECT_ID}.pdf"
    fig.savefig(out_file)
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    plot_multichannel_heatmap(FILE_PATH_REST)
