import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import re

from plotting.plot_config import get_figs_output_dir

FILE_DIR = Path("/Users/ipeglin/Documents/masters_thesis/bids_processed_consolidated_data")
FILE_DIR = Path("/Volumes/work/bids_processed_consolidated_data")

def extract_subject_mode_data(subject_dir, subject_id, file_suffix, exp_name):
    file_path = subject_dir / f"{subject_id}{file_suffix}"
    if not file_path.exists():
        return None
    
    records = []
    try:
        with h5py.File(file_path, 'r') as f:
            if '04hht/full_run_std' in f:
                group = f['04hht/full_run_std']
                modes = np.asarray(group['modes']) # [K, C, T]
                center_freqs = np.asarray(group['center_frequencies']) # [K]
                
                K, C, T = modes.shape
                energy = np.sum(modes**2, axis=(1, 2))
                rel_energy = energy / np.sum(energy) * 100
                
                for k in range(K):
                    records.append({
                        'Subject': subject_id,
                        'fMRI run': exp_name,
                        'Mode': k + 1,
                        'CenterFreq': center_freqs[k],
                        'Energy': rel_energy[k]
                    })
            elif '04hht/blocks_std' in f:
                blocks_std = f['04hht/blocks_std']
                all_modes_energy = []
                all_center_freqs = []
                K = None
                
                for trial_type in blocks_std.keys():
                    trial_group = blocks_std[trial_type]
                    for block_name in trial_group.keys():
                        group = trial_group[block_name]
                        modes = np.asarray(group['modes'])
                        center_freqs = np.asarray(group['center_frequencies'])
                        K = modes.shape[0]
                        energy = np.sum(modes**2, axis=(1, 2))
                        rel_energy = energy / np.sum(energy) * 100
                        
                        all_modes_energy.append(rel_energy)
                        all_center_freqs.append(center_freqs)
                
                if K is not None:
                    # Average across blocks
                    avg_energy = np.mean(all_modes_energy, axis=0)
                    avg_freqs = np.mean(all_center_freqs, axis=0)
                    
                    for k in range(K):
                        records.append({
                            'Subject': subject_id,
                            'fMRI run': exp_name,
                            'Mode': k + 1,
                            'CenterFreq': avg_freqs[k],
                            'Energy': avg_energy[k]
                        })
            else:
                return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    return records

def collect_data(include_hammer=False):
    all_data = []
    
    if not FILE_DIR.exists():
        print(f"Directory not found: {FILE_DIR}")
        return pd.DataFrame()
        
    for subject_dir in FILE_DIR.glob("sub-*"):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name
        
        # Rest run-01
        res1 = extract_subject_mode_data(
            subject_dir, subject_id,
            "_task-restAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.h5",
            "RS run 1 (AP)"
        )
        if res1: all_data.extend(res1)
        
        # Rest run-02
        res2 = extract_subject_mode_data(
            subject_dir, subject_id,
            "_task-restAP_run-02_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.h5",
            "RS run 2 (AP)"
        )
        if res2: all_data.extend(res2)
        
        # Hammer task
        if include_hammer:
            res_hammer = extract_subject_mode_data(
                subject_dir, subject_id,
                "_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.h5",
                "Hammer task"
            )
            if res_hammer: all_data.extend(res_hammer)
            
    return pd.DataFrame(all_data)

def plot_modes_distribution():
    INCLUDE_HAMMER = True # easy toggle
    df = collect_data(include_hammer=INCLUDE_HAMMER)
    
    if df.empty:
        print("No data extracted.")
        return
        
    # frequencies to mHz
    df['CenterFreq'] *= 1000

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Neurophysiological band 10 - 200 mHz
    neuro_band_min = 10
    neuro_band_max = 200
    
    # For shaded area logic, find IM limits whose average frequency falls in the band
    # Group by mode and check if its average freq is between 10 and 200
    avg_freqs = df.groupby('Mode')['CenterFreq'].mean()
    within_band_modes = avg_freqs[(avg_freqs >= neuro_band_min) & (avg_freqs <= neuro_band_max)].index.tolist()
    
    if within_band_modes:
        shade_start = min(within_band_modes) - 0.5
        shade_end = max(within_band_modes) + 0.5
    else:
        shade_start, shade_end = None, None

    # (a) Frequency Distribution
    ax_freq = axes[0]
    sns.boxplot(data=df, x='Mode', y='CenterFreq', hue='fMRI run', ax=ax_freq, fliersize=3)
    ax_freq.set_title('(a)')
    ax_freq.set_ylabel('Frequency (mHz)')
    ax_freq.set_xlabel('Intrinsic Mode Function')
    ax_freq.grid(axis='y', linestyle='-', alpha=0.7)
    
    if shade_start is not None:
        # Subtract 1 because x-axis is categorical 0-indexed in seaborn boxplot but labels are 1-10
        ax_freq.axvspan(shade_start - 1, shade_end - 1, color='#e5f0f3', alpha=0.5, zorder=-1)
        
    # (b) Energy Distribution
    ax_energy = axes[1]
    sns.boxplot(data=df, x='Mode', y='Energy', hue='fMRI run', ax=ax_energy, fliersize=3)
    ax_energy.set_title('(b)')
    ax_energy.set_ylabel('Energy (%)')
    ax_energy.set_xlabel('Intrinsic Mode Functions')
    ax_energy.grid(axis='y', linestyle='-', alpha=0.7)
    
    if shade_start is not None:
        ax_energy.axvspan(shade_start - 1, shade_end - 1, color='#e5f0f3', alpha=0.5, zorder=-1)
        
    plt.tight_layout()
    
    out_dir = get_figs_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "modes_distribution.pdf"
    
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")

if __name__ == '__main__':
    plot_modes_distribution()
