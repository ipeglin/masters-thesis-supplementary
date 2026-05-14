"""
Illustrates the effect of underbinning (too few modes $M$) in MVMD.
Shows:
  - Underbinned decomposition (M too small) with high alpha. 
    Modes are wide-band and absorb multiple signal components.
  - Proper decomposition (sufficient M) isolating components cleanly.
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from lib.fs.get_script_name import get_name
from plotting import plot_config
from plotting.plot_config import FIG_WIDTH_INCHES, get_figs_output_dir
from lib.mvmd.mvmd import mvmd

out_dir = get_figs_output_dir()

# Signal parameters
FS = 1000          
T = 2.0            
MODE_FREQS = [20, 50, 80, 110, 140]
MODE_AMPS = [1.0, 0.8, 1.2, 0.9, 1.1]

ALPHA_UNDER = 5000  
M_UNDER = 2

ALPHA_PROPER = 5000
M_PROPER = 5

MODE_COLORS = ['#d6604d', '#4dac26', '#ac26a8', '#3498db', '#f4a582']
ALPHA_FILL  = 0.18

def generate_underbinning_illustration():
    t = np.linspace(0, T, int(T * FS), endpoint=False)
    N = len(t)
    
    # Composite signal
    modes_true = [a * np.cos(2 * np.pi * w * t) for a, w in zip(MODE_AMPS, MODE_FREQS)]
    signal = np.sum(modes_true, axis=0)
    signal_mvmd = signal.reshape(1, N)
    
    # ── MVMD ──────────────────────────────────────────────────────────── #
    modes_under, _, _ = mvmd(signal_mvmd, M_UNDER, ALPHA_UNDER, init=1, tau=0)
    modes_proper, _, _ = mvmd(signal_mvmd, M_PROPER, ALPHA_PROPER, init=1, tau=0)
    
    freqs = np.fft.rfftfreq(N, 1.0 / FS)
    
    # ── Plot ──────────────────────────────────────────────────────────── #
    fig = plt.figure(figsize=(FIG_WIDTH_INCHES * 1.8, FIG_WIDTH_INCHES * 1.2))
    # gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    xlim_sp = 180
    
    # Underbinned Spectra
    ax_us = fig.add_subplot(gs[0, 0])
    for m in range(M_UNDER):
        spec = np.abs(np.fft.rfft(modes_under[m, 0, :])) / N
        ax_us.plot(freqs, spec, color=MODE_COLORS[m], label=f'IMF$_{{\\mathrm{{{m}}}}}$')
        ax_us.fill_between(freqs, spec, alpha=ALPHA_FILL, color=MODE_COLORS[m])
    ax_us.set_title(r'Underbinned Spectra ($M=%d$, $\alpha=%d$)' % (M_UNDER, ALPHA_UNDER))
    ax_us.set_xlabel('Frequency (Hz)')
    ax_us.set_ylabel('Magnitude')
    ax_us.set_xlim(0, xlim_sp)
    ax_us.legend(loc='upper right')
    
    # Proper Spectra
    ax_ps = fig.add_subplot(gs[0, 1])
    for m in range(M_PROPER):
        spec = np.abs(np.fft.rfft(modes_proper[m, 0, :])) / N
        c = MODE_COLORS[m % len(MODE_COLORS)]
        ax_ps.plot(freqs, spec, color=c, label=f'IMF$_{{\\mathrm{{{m}}}}}$')
        ax_ps.fill_between(freqs, spec, alpha=ALPHA_FILL, color=c)
    ax_ps.set_title(r'Proper Spectra ($M=%d$, $\alpha=%d$)' % (M_PROPER, ALPHA_PROPER))
    ax_ps.set_xlabel('Frequency (Hz)')
    ax_ps.set_xlim(0, xlim_sp)
    ax_ps.legend(loc='upper right')
    
    OFFSET = 3.0

    # Underbinned Time
    ax_ut = fig.add_subplot(gs[1, 0])
    t_show = slice(0, int(0.4 * FS))
    for m in range(M_UNDER):
        y_shift = -m * OFFSET
        ax_ut.plot(t[t_show], modes_under[m, 0, t_show] + y_shift, color=MODE_COLORS[m])
        ax_ut.axhline(y_shift, color='k', ls='--', alpha=0.1, lw=0.8)
    ax_ut.set_title('Underbinned Modes in Time')
    ax_ut.set_xlabel('Time (s)')
    ax_ut.set_yticks([])
    
    # Proper Time
    ax_pt = fig.add_subplot(gs[1, 1])
    for m in range(M_PROPER):
        c = MODE_COLORS[m % len(MODE_COLORS)]
        y_shift = -m * OFFSET
        ax_pt.plot(t[t_show], modes_proper[m, 0, t_show] + y_shift, color=c, lw=0.8)
        ax_pt.axhline(y_shift, color='k', ls='--', alpha=0.1, lw=0.8)
    ax_pt.set_title('Proper Modes in Time')
    ax_pt.set_xlabel('Time (s)')
    ax_pt.set_yticks([])
    
    out_path = out_dir / (get_name() + '.pdf')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)

if __name__ == '__main__':
    generate_underbinning_illustration()
    print("VMD underbinning illustration generated.")
