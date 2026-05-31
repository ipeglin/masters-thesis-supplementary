import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from math import gamma
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plotting import plot_config

# ---------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------

out_dir = plot_config.get_figs_output_dir()
pdf_path = out_dir / 'bold_composite_hrf_mvmd_concept.pdf'

# ---------------------------------------------------------------------
# Global layout settings
# ---------------------------------------------------------------------

fig_width = plot_config.FIG_WIDTH_INCHES
fig_height = fig_width * 1.3
fig_dpi = 220


def text_arrow(from_pos, to_pos, text=None):
  x1, y1 = from_pos
  x2, y2 = to_pos
  arr = FancyArrowPatch(
    (x1, y1),
    (x2, y2),
    arrowstyle='-|>',
    mutation_scale=18,
    linewidth=1.4,
    color='0.20',
    connectionstyle='arc3,rad=0.0'
  )
  ax_bg.add_patch(arr)
  if text is not None:
      fig.text(x1 + 0.015, (y1 + y2) / 2, text, ha='left', va='center', fontsize=12)


def text_arrow_vertical(length, direction, from_pos=None, to_pos=None, text=None):
  if direction not in {'up', 'down'}:
    raise ValueError(f'Unsupported vertical direction: {direction}')

  if to_pos is not None:
    x, y2 = to_pos
    y1 = y2 + length if direction == 'down' else y2 - length
    from_pos = (x, y1)
  elif from_pos is not None:
    x, y1 = from_pos
    y2 = y1 - length if direction == 'down' else y1 + length
    to_pos = (x, y2)
  else:
    raise ValueError('text_arrow_vertical requires from_pos or to_pos')

  text_arrow(from_pos, to_pos, text=text)


def text_arrow_horizontal(length, direction, from_pos=None, to_pos=None, text=None):
  if direction not in {'left', 'right'}:
    raise ValueError(f'Unsupported horizontal direction: {direction}')
  # horizontal lengths are in figure-relative fractions; because the figure
  # height and width differ, a fixed fraction in x does not match the visual
  # physical length of the same fraction in y. Scale x-length so that a
  # horizontal arrow with `length` matches the on-page size of a vertical
  # arrow of the same `length`.
  length_x = length * (fig_height / fig_width)

  if to_pos is not None:
    x2, y = to_pos
    x1 = x2 - length_x if direction == 'right' else x2 + length_x
    from_pos = (x1, y)
  elif from_pos is not None:
    x1, y = from_pos
    x2 = x1 + length_x if direction == 'right' else x1 - length_x
    to_pos = (x2, y)
  else:
    raise ValueError('text_arrow_horizontal requires from_pos or to_pos')

  text_arrow(from_pos, to_pos, text=text)

arrow_length = 0.05

# ---------------------------------------------------------------------
# Synthetic conceptual signal generation
# ---------------------------------------------------------------------

np.random.seed(7)

t = np.linspace(0, 120, 900)
dt = t[1] - t[0]

def gamma_hrf(t, a1=6, b1=1, a2=16, b2=1, c=1 / 6):
  h = (t ** (a1 - 1) * np.exp(-t / b1)) / (gamma(a1) * b1 ** a1)
  h -= c * (t ** (a2 - 1) * np.exp(-t / b2)) / (gamma(a2) * b2 ** a2)
  h /= np.max(h)
  return h

def norm(x):
  return x / np.max(np.abs(x))

neural = (
  0.65 * np.sin(2 * np.pi * 0.075 * t + 0.4)
  + 0.28 * np.sin(2 * np.pi * 0.17 * t)
) * (0.75 + 0.25 * np.sin(2 * np.pi * 0.012 * t))

vascular = (
  0.85 * np.sin(2 * np.pi * 0.032 * t - 0.8)
  + 0.12 * np.sin(2 * np.pi * 0.09 * t + 1.5)
)

metabolic = (
  0.55 * np.sin(2 * np.pi * 0.018 * t + 1.1)
  + 0.14 * np.sin(2 * np.pi * 0.13 * t + 2.2)
)

compartment = (
  0.22 * np.sin(2 * np.pi * 0.21 * t + 0.7)
  + 0.10 * np.sin(2 * np.pi * 0.28 * t)
)

components = [
  ('Deoxyhaemoglobin', neural),
  ('Venous volume', vascular),
  ('Oxygen extraction', metabolic),
  ('Compartment effects', compartment)
]

weights = np.array([0.44, 0.34, 0.26, 0.16])

composite = sum(w * x for w, (_, x) in zip(weights, components))
composite += 0.10 * np.sin(2 * np.pi * 0.005 * t)

hrf_t = np.arange(0, 32, dt)
hrf = gamma_hrf(hrf_t)
hrf /= np.sum(hrf)

bold_lpf = np.convolve(composite, hrf, mode='same')
bold_lpf = bold_lpf / np.std(bold_lpf) * 0.75

components = [(label, norm(sig)) for label, sig in components]
composite_n = norm(composite)
bold_n = norm(bold_lpf)


# ---------------------------------------------------------------------
# Figure canvas
# ---------------------------------------------------------------------

fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
fig.set_layout_engine(None)
ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
ax_bg.set_axis_off()

# Footer note
fig.text(
  0.5,
  -0.01,
  'Conceptual illustration only: traces are synthetic and intended to show signal mixing.',
  ha='center',
  va='center',
  fontsize=10
)

# ---------------------------------------------------------------------
# Bottom Row: HRF-smoothed BOLD
# ---------------------------------------------------------------------

bot_pad = 0.05
ax3 = fig.add_axes([0.05 + bot_pad, 0, 0.90 - 2 * bot_pad, 0.15])
ax3.patch.set_alpha(0.0)

ax3.axhline(0, color='0.75')

ax3.plot(
  t,
  composite_n * 0.33,
  alpha=0.35,
  linestyle='--',
  color='k',
  label='Composite before HRF'
)

ax3.plot(
  t,
  bold_n,
  color=plot_config.CONCEPT_PALETTE["accent_a"],
  label='Observed LPF BOLD'
)

ax3.set_xlim(t.min(), t.max())
ax3.set_ylim(-1.30, 1.30)
ax3.set_xticks([])
ax3.set_yticks([])

for spine in ax3.spines.values():
  spine.set_visible(False)
  
# Title for Bottom Row
fig.text(0.5, 0.15, 'Observed LPF BOLD signal', ha='center', va='center', fontsize=12, fontweight='bold')

# HRF inset
hrf_ax = fig.add_axes([0.63, 0.19, 0.10, 0.08])
hrf_ax.patch.set_alpha(0.0)
hrf_ax.plot(hrf_t, hrf / np.max(hrf), color=plot_config.CONCEPT_PALETTE["accent_b"])
hrf_ax.set_xticks([])
hrf_ax.set_yticks([])

for spine in hrf_ax.spines.values():
  spine.set_visible(False)

hrf_ax.text(
  0.5,
  1.1,
  'HRF',
  transform=hrf_ax.transAxes,
  fontsize=10,
  ha='center',
  va='top'
)

# Arrow down to row 3
arrow_r2_3_target_pos = (0.5, 0.17)
text_arrow_vertical(arrow_length, 'down', to_pos=arrow_r2_3_target_pos)

# Convolution symbol
conv_sym_pos = (0.5, arrow_r2_3_target_pos[1] + 1.3*arrow_length)
fig.text(conv_sym_pos[0], conv_sym_pos[1], r'$\circledast$', ha='center', va='center', fontsize=22)

# Arrow pointing into convolution from HRF inset
arrow_hrf_to_conv_pos = (conv_sym_pos[0] + 1.5*1.3*arrow_length, conv_sym_pos[1])
text_arrow_horizontal(arrow_length, 'left', from_pos=arrow_hrf_to_conv_pos)
fig.text(0.4, conv_sym_pos[1]-0.007, 'HRF conv', ha='center', va='bottom', fontsize=11)

# Convolution graphics between row 2 and row 3
# Arrow down from row 2
arrow_r2_conv_pos = (conv_sym_pos[0], conv_sym_pos[1] + 1.3*arrow_length)
text_arrow_vertical(arrow_length, 'down', from_pos=arrow_r2_conv_pos)

# ---------------------------------------------------------------------
# Middle Row: Composite BOLD
# ---------------------------------------------------------------------

mid_pad = 0.05
ax2 = fig.add_axes([0.05 + mid_pad, 0.32, 0.90 - 2 * mid_pad, 0.12])
ax2.patch.set_alpha(0.0)
ax2.axhline(0, color='0.75')
ax2.plot(t, composite_n, color='k')

ax2.set_xlim(t.min(), t.max())
ax2.set_ylim(-1.15, 1.15)
ax2.set_xticks([])
ax2.set_yticks([])

for spine in ax2.spines.values():
  spine.set_visible(False)

# fig.text(
#   0.5,
#   0.465,
#   (
#     r'$S=(1-V)S_e+VS_i$'
#     + '\n'
#     + r'$\Delta S/S \approx V_0[k_1(1-q)+k_2(1-q/v)+k_3(1-v)]$'
#   ),
#   ha='center',
#   va='center',
#   fontsize=12
# )

# Title for Middle Row
fig.text(0.5, 0.47, 'Volume-weighted composite BOLD', ha='center', va='center', fontsize=12, fontweight='bold')


# Upper arrow from row 1 to row 2
text_arrow_vertical(arrow_length*3, 'down', to_pos=(0.5, 0.49), text='weighted sum')

# ---------------------------------------------------------------------
# Top Row: 3D Latent Components
# ---------------------------------------------------------------------

# Add a 3D projection, then remove axes to make it clean.
# The axis is made slightly wider than the figure canvas because the 3D
# projection compresses the apparent horizontal extent of the traces.
# [left, bottom, width, height]
ax1 = fig.add_axes([-0.08, 0.47, 1.16, 0.55], projection='3d')
ax1.set_axis_off()
ax1.patch.set_alpha(0.0)
ax1.view_init(elev=18, azim=-61)
ax1.set_box_aspect((9.0, 2.7, 0.85))  # x (horizontal), y (depth), z (height)

for i, (label, sig) in enumerate(components):
    z_spacing = 4
    y_pos = (len(components) - i) * z_spacing
    line, = ax1.plot(t, np.full_like(t, y_pos), sig, label=label)
    
    # Baseline for floating effect
    ax1.plot(t, np.full_like(t, y_pos), np.zeros_like(t), color='gray', alpha=0.3, linestyle='--')

# ax1.legend(bbox_to_anchor=(-0.1, 0.4), frameon=False, fontsize=11, reverse=True)
ax1.legend(
    loc='center left',             
    bbox_to_anchor=(0.05, 0.65),   # X, Y relative to the whole figure (0 to 1)
    bbox_transform=fig.transFigure, # <--- This is the correct keyword!
    frameon=False, 
    fontsize=11, 
    reverse=True
)
ax1.margins(x=0)
ax1.set_xlim(t.min(), t.max())

# Title for Top Row
fig.text(0.5, 0.88, 'Latent dynamic components', ha='center', va='center', fontsize=12, fontweight='bold')


# # MVMD motivation box
# mvmd_box = FancyBboxPatch(
#   (0.15, 0.015),
#   0.70,
#   0.06,
#   boxstyle='round,pad=0.012,rounding_size=0.018',
#   linewidth=1.2,
#   edgecolor='0.25',
#   facecolor='0.94'
# )
# ax_bg.add_patch(mvmd_box)

# fig.text(
#   0.5,
#   0.060,
#   'MVMD motivation',
#   ha='center',
#   va='center',
#   fontsize=12,
#   fontweight='bold'
# )

# fig.text(
#   0.5,
#   0.035,
#   (
#     'Decompose multivariate BOLD signals to inspect hidden frequency-resolved\n'
#     'dynamics rather than relying only on the smoothed waveform.'
#   ),
#   ha='center',
#   va='center',
#   fontsize=11
# )


# ---------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------

plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.05)
print(f"Saved: {pdf_path}")
