import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

from lib.fs.get_script_name import get_name
from plotting import plot_config
from plotting.plot_config import get_figs_output_dir

# FILE_DIR = Path("/Users/ipeglin/Documents/masters_thesis/bids_processed_consolidated_data")
FILE_DIR = Path("/Volumes/work/bids_processed_consolidated_data")  # IDUN network mount

out_dir = get_figs_output_dir()
SUBJECT_ID = "sub-NDARINVAG388HJL"
FILE_SUFFIX_REST = "_task-restAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.h5"
FILE_SUFFIX_HAMMER = "_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.h5"
FILE_PATH_REST = FILE_DIR / SUBJECT_ID / f"{SUBJECT_ID}{FILE_SUFFIX_REST}"
FILE_PATH_HAMMER = FILE_DIR / SUBJECT_ID / f"{SUBJECT_ID}{FILE_SUFFIX_HAMMER}"
FS = 1.25 # fs for TR = 0.8s

SLOW_BANDS = [
    ("Slow 5*", 0.005, 0.010, "#e8f4f8"),
    ("Slow 4", 0.027, 0.073, "#f8f0e8"),
    ("Slow 3", 0.073, 0.198, "#e8f8ec"),
    ("Slow 2*", 0.198, 0.250, "#f8e8f4"),
]

def plot_all_channels(FILE_PATH, group_name, out_filename, random_subset=False, n_channels=10, seed=42, filter_mode="all"):
    if not FILE_PATH.exists():
        print(f"File not found: {FILE_PATH}")
        return
        
    with h5py.File(FILE_PATH, 'r') as f:
        if group_name not in f:
            print(f"Group {group_name} not found")
            return
        group = f[group_name]
        modes = np.asarray(group['modes'])           # [K, C, T]
        center_freqs = np.asarray(group['center_frequencies'])  # [K]
    
    K, C, T = modes.shape

    valid_k = []
    band_infos = []
    for k in range(K):
        band_name, band_color = None, None
        for name, low, high, color in SLOW_BANDS:
            if low <= center_freqs[k] <= high:
                band_name, band_color = name, color
                break
        band_infos.append((band_name, band_color))
        
        if filter_mode == "all":
            valid_k.append(k)
        elif filter_mode == "slow" and band_name is not None:
            valid_k.append(k)
        elif filter_mode == "fast" and band_name is None:
            valid_k.append(k)
            
    if not valid_k:
        print(f"No valid modes for filter {filter_mode} in {group_name}")
        return
    
    if random_subset and C > n_channels:
        rng = np.random.default_rng(seed)
        channel_indices = np.sort(rng.choice(C, size=n_channels, replace=False))
        modes = modes[:, channel_indices, :]
        C = n_channels
    else:
        channel_indices = np.arange(C)

    t = np.arange(T) / FS
    
    fig, axes = plt.subplots(len(valid_k), C, figsize=(max(C * 0.5, 15), len(valid_k) * 1.2), sharex=True, sharey='row', squeeze=False)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    for i, k in enumerate(valid_k):
        band_name, band_color = band_infos[k]
                
        for c in range(C):
            ax = axes[i, c]
            if band_color:
                ax.set_facecolor(band_color)
                
            ax.plot(t, modes[k, c, :], color='k', linewidth=0.3, alpha=0.6)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if c > 0:
                ax.spines['left'].set_visible(False)
                
            if i == 0:
                ax.set_title(f"Ch {channel_indices[c]}", rotation=45, ha='left', fontsize=8)
                
            if c == 0:
                ax.set_ylabel(f"IMF$_{{{k}}}$\n{center_freqs[k]:.3f} Hz", rotation=0, labelpad=40, va='center')
                ax.spines['left'].set_visible(True)
                ax.tick_params(left=True, labelleft=True)
                
            if c == C - 1 and band_name:
                ax.text(1.02, 0.5, band_name, transform=ax.transAxes, va='center', ha='left', fontsize=10, color='#555555', fontweight='bold')
                
    # Shared x-label
    fig.text(0.5, -0.02, 'Time (s)', ha='center')
    
    out_path = out_dir / f"{out_filename}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")

def plot_roi_channels(FILE_PATH, group_name, out_filename, random_subset=False, n_channels=10, seed=42, filter_mode="all"):
    if not FILE_PATH.exists():
        print(f"File not found: {FILE_PATH}")
        return
        
    with h5py.File(FILE_PATH, 'r') as f:
        if group_name not in f:
            print(f"Group {group_name} not found")
            return
        group = f[group_name]
        modes = np.asarray(group['modes'])           # [K, C, T]
        center_freqs = np.asarray(group['center_frequencies'])  # [K]
        roi_labels = group.attrs['roi_labels'].split(',')
        
    K, C, T = modes.shape

    valid_k = []
    band_infos = []
    for k in range(K):
        band_name, band_color = None, None
        for name, low, high, color in SLOW_BANDS:
            if low <= center_freqs[k] <= high:
                band_name, band_color = name, color
                break
        band_infos.append((band_name, band_color))
        
        if filter_mode == "all":
            valid_k.append(k)
        elif filter_mode == "slow" and band_name is not None:
            valid_k.append(k)
        elif filter_mode == "fast" and band_name is None:
            valid_k.append(k)
            
    if not valid_k:
        print(f"No valid modes for filter {filter_mode} in {group_name}")
        return
    
    if random_subset and C > n_channels:
        rng = np.random.default_rng(seed)
        channel_indices = np.sort(rng.choice(C, size=n_channels, replace=False))
        modes = modes[:, channel_indices, :]
        roi_labels = [roi_labels[i] for i in channel_indices]
        C = n_channels
    else:
        channel_indices = np.arange(C)

    t = np.arange(T) / FS
    
    fig, axes = plt.subplots(len(valid_k), C, figsize=(max(C * 3.0, 15), len(valid_k) * 1.5), sharex=True, sharey='row', squeeze=False)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    for i, k in enumerate(valid_k):
        band_name, band_color = band_infos[k]
                
        for c in range(C):
            ax = axes[i, c]
            if band_color:
                ax.set_facecolor(band_color)
                
            ax.plot(t, modes[k, c, :], color='k', linewidth=0.8, alpha=0.8)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
                
            if i == 0:
                ax.set_title(f"ROI {channel_indices[c]}\n{roi_labels[c].replace("_", " ").replace("17networks ", "")}", rotation=45, ha='left', fontsize=9)
                
            if c == 0:
                ax.set_ylabel(f"IMF$_{{{k}}}$\n{center_freqs[k]:.3f} Hz", rotation=0, labelpad=40, va='center')
                ax.tick_params(left=True, labelleft=True)
                
            if c == C - 1 and band_name:
                ax.text(1.02, 0.5, band_name, transform=ax.transAxes, va='center', ha='left', fontsize=10, color='#555555', fontweight='bold')
                
            if i == len(valid_k) - 1:
                ax.tick_params(bottom=True, labelbottom=True)
                
    fig.text(0.5, -0.02, 'Time (s)', ha='center')

    out_path = out_dir / f"{out_filename}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

if __name__ == '__main__':
    RANDOM_SUBSET = True
    N_CHANNELS = 3
    SEED = 42

    for suffix in ["", "_na"]:
        for filter_mode in ["all", "slow", "fast"]:
            plot_all_channels(FILE_PATH_REST, f'04hht/full_run_std{suffix}', f"rest/{filter_mode}/all_channels_{filter_mode}{suffix}", random_subset=RANDOM_SUBSET, n_channels=N_CHANNELS, seed=SEED, filter_mode=filter_mode)
            plot_roi_channels(FILE_PATH_REST, f'04hht/full_run_std_roi{suffix}', f"rest/{filter_mode}/roi_{filter_mode}{suffix}", random_subset=RANDOM_SUBSET, n_channels=N_CHANNELS, seed=SEED, filter_mode=filter_mode)
            
            if FILE_PATH_HAMMER.exists():
                with h5py.File(FILE_PATH_HAMMER, 'r') as f:
                    base_group = f'04hht/blocks_std{suffix}'
                    if base_group in f:
                        for condition in f[base_group].keys():
                            blocks = sorted(list(f[base_group][condition].keys()), key=natural_sort_key)
                            for block in blocks:
                                plot_all_channels(FILE_PATH_HAMMER, f'{base_group}/{condition}/{block}', f"hammer_{condition}{suffix}/{filter_mode}/all_channels_{block}_{filter_mode}", random_subset=RANDOM_SUBSET, n_channels=N_CHANNELS, seed=SEED, filter_mode=filter_mode)
                    
                    base_group_roi = f'04hht/blocks_std_roi{suffix}'
                    if base_group_roi in f:
                        for condition in f[base_group_roi].keys():
                            blocks = sorted(list(f[base_group_roi][condition].keys()), key=natural_sort_key)
                            for block in blocks:
                                plot_roi_channels(FILE_PATH_HAMMER, f'{base_group_roi}/{condition}/{block}', f"hammer_{condition}{suffix}/{filter_mode}/roi_{block}_{filter_mode}", random_subset=RANDOM_SUBSET, n_channels=N_CHANNELS, seed=SEED, filter_mode=filter_mode)
