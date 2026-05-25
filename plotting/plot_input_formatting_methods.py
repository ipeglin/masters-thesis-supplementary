from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from matplotlib import patches
from matplotlib.patches import ConnectionPatch

try:
  import plot_config
  FIG_WIDTH_INCHES = plot_config.FIG_WIDTH_INCHES
  FIG_HEIGHT_INCHES = plot_config.FIG_HEIGHT_INCHES
except ImportError:
  FIG_WIDTH_INCHES = 6.4
  FIG_HEIGHT_INCHES = 4.8

try:
  from lib.fs.project_config import REPO_ROOT
  from lib.fs.get_script_name import get_name
except ImportError:
  REPO_ROOT = Path(__file__).resolve().parent

  def get_name():
    return Path(__file__).stem

OUT_DIR = REPO_ROOT / 'figures' / get_name()
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
TARGET_SHAPE = (224, 224)
BLUE = '#4d89a8'
DARK_BLUE = '#2f5f78'
ORANGE = '#e8952f'
DARK = '#2b2b2b'
GREY = '#f3f6f8'
LIGHT_BLUE = '#eaf3f7'
PAD_COLOUR = '#071b43'


def generate_synthetic_spectrum(height, width, seed=None, kind='rest'):
  rng = np.random.default_rng(seed)
  y = np.linspace(0, 1, height)[:, None]
  x = np.linspace(0, 1, width)[None, :]

  if kind == 'task':
    base = (
      0.45 * np.sin(2 * np.pi * (3.0 * y + 0.30 * np.sin(2 * np.pi * x)))
      + 0.25 * np.cos(2 * np.pi * (8.0 * y - 1.4 * x))
      + 0.20 * np.sin(2 * np.pi * (1.3 * x + 2.2 * y))
    )
    noise = rng.normal(0, 0.35, size=(height, width))
    img = base + scipy.ndimage.gaussian_filter(noise, sigma=(2.0, 0.8))
  else:
    blobs = rng.normal(0, 1, size=(height // 5, width // 8))
    blobs = scipy.ndimage.zoom(
      blobs,
      (height / blobs.shape[0], width / blobs.shape[1]),
      order=3,
    )[:height, :width]
    trend = 0.35 * np.sin(2 * np.pi * (2.2 * y + 0.9 * x))
    img = blobs + trend

  img = scipy.ndimage.gaussian_filter(img, sigma=(1.1, 1.1))
  img = (img - img.min()) / (img.max() - img.min())
  return img


def resize_image(img, target_shape=TARGET_SHAPE):
  factors = (target_shape[0] / img.shape[0], target_shape[1] / img.shape[1])
  out = scipy.ndimage.zoom(img, factors, order=3)
  return out[:target_shape[0], :target_shape[1]]


def pad_image_width(img, target_width=224, pad_value=0.0):
  if img.shape[1] >= target_width:
    return img[:, :target_width]
  pad_width = target_width - img.shape[1]
  return np.pad(
    img,
    ((0, 0), (0, pad_width)),
    mode='constant',
    constant_values=pad_value,
  )


def hide_axes(ax):
  ax.set_xticks([])
  ax.set_yticks([])
  for spine in ax.spines.values():
    spine.set_visible(False)


def add_card(fig, rect, title=None, subtitle=None, facecolor='white'):
  x, y, w, h = rect
  card = patches.FancyBboxPatch(
    (x, y),
    w,
    h,
    boxstyle='round,pad=0.010,rounding_size=0.018',
    transform=fig.transFigure,
    linewidth=1.0,
    edgecolor='#d3dce3',
    facecolor=facecolor,
    zorder=-5,
  )
  fig.patches.append(card)

  if title is not None:
    fig.text(
      x + 0.018,
      y + h - 0.035,
      title,
      ha='left',
      va='top',
      fontsize=12,
      fontweight='bold',
      color=DARK,
    )
  if subtitle is not None:
    fig.text(
      x + 0.018,
      y + h - 0.062,
      subtitle,
      ha='left',
      va='top',
      fontsize=11,
      color='#5f6b73',
    )


def add_image_panel(fig, rect, img, title, subtitle=None, cmap='viridis', vmin=0, vmax=1):
  add_card(fig, rect, title, subtitle)
  x, y, w, h = rect
  title_space = 0.108 if subtitle is not None else 0.073
  if subtitle and '\n' in subtitle:
    title_space += 0.030
  ax = fig.add_axes([x + 0.018, y + 0.025, w - 0.036, h - title_space])
  ax.imshow(img, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
  hide_axes(ax)
  return ax


def add_stacked_image_panel(fig, base_rect, imgs, title, subtitle=None, cmap='viridis', vmin=0, vmax=1):
  n = len(imgs)
  x_base, y_base, w, h = base_rect
  
  # 1. Draw a single background card
  add_card(fig, base_rect, title, subtitle)
  
  # 2. Compute inner bounds so we don't overlap the label text
  title_space = 0.108 if subtitle is not None else 0.073
  if subtitle and '\n' in subtitle:
    title_space += 0.030
  x_inner = x_base + 0.018
  y_inner = y_base + 0.025
  w_inner = w - 0.036
  h_inner = h - title_space
  
  # 3. Stack offsets (visual depth)
  dx, dy = 0.018, 0.018 # Shift per layer inside the card
  w_img = w_inner - (n - 1) * dx
  h_img = h_inner - (n - 1) * dy
  
  axes = []
  # Draw back (layer=n-1) to front (layer=0)
  for layer in range(n - 1, -1, -1):
    img = imgs[layer]
    ax_x = x_inner + layer * dx
    ax_y = y_inner + layer * dy
    ax = fig.add_axes([ax_x, ax_y, w_img, h_img])
    
    # Back layers are slightly transparent (appears washed out / less saturated over white)
    layer_alpha = 1.0 - 0.25 * layer
    ax.imshow(img, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, alpha=layer_alpha)
    
    # Add a thin subtle border to distinguish overlapping layers
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#acb8c2')
        spine.set_linewidth(0.8)
    
    ax.set_xticks([])
    ax.set_yticks([])
    axes.append((layer, ax))
  
  # Return sorted so axes[0] is the front image, axes[1] middle, etc.
  axes.sort(key=lambda item: item[0])
  return [item[1] for item in axes]


def add_process_box(fig, rect, title, lines, facecolor=LIGHT_BLUE):
  add_card(fig, rect, title, facecolor=facecolor)
  x, y, w, h = rect
  for i, line in enumerate(lines):
    fig.text(
      x + 0.030,
      y + h - 0.080 - i * 0.034,
      line,
      ha='left',
      va='top',
      fontsize=8.2,
      color=DARK_BLUE,
    )


def add_cnn_box(fig, rect, title='CNN feature extractor', subtitle='224$\\times$224 input'):
  x, y, w, h = rect
  box = patches.FancyBboxPatch(
    (x, y),
    w,
    h,
    boxstyle='round,pad=0.012,rounding_size=0.020',
    transform=fig.transFigure,
    linewidth=0,
    facecolor=BLUE,
    zorder=-5,
  )
  fig.patches.append(box)
  fig.text(
    x + w / 2,
    y + h * 0.60,
    title,
    ha='center',
    va='center',
    fontsize=12,
    fontweight='bold',
    color='white',
  )
  fig.text(
    x + w / 2,
    y + h * 0.35,
    subtitle,
    ha='center',
    va='center',
    fontsize=11.0,
    color='white',
  )


def connect(fig, start, end, color=BLUE, rad=0.0, lw=2.2):
  arrow = ConnectionPatch(
    start,
    end,
    coordsA=fig.transFigure,
    coordsB=fig.transFigure,
    arrowstyle='-|>',
    mutation_scale=16,
    linewidth=lw,
    color=color,
    connectionstyle=f'arc3,rad={rad}',
    shrinkA=4,
    shrinkB=4,
    zorder=10,
  )
  fig.add_artist(arrow)


def rect_mid_left(rect):
  x, y, w, h = rect
  return x - 0.01, y + h / 2


def rect_mid_right(rect):
  x, y, w, h = rect
  return x + w + 0.02, y + h / 2


def rect_mid_bottom(rect):
  x, y, w, h = rect
  return x + w / 2, y


def rect_mid_top(rect):
  x, y, w, h = rect
  return x + w / 2, y + h


def add_pad_overlay(ax, start_col, total_cols=224, alpha_mult=1.0):
  ax.axvspan(start_col - 0.5, total_cols - 0.5, color=PAD_COLOUR, alpha=0.95 * alpha_mult, lw=0)
  ax.axvline(start_col - 0.5, color='white', lw=0.8, alpha=0.70 * alpha_mult)


def add_block_boundaries(ax, block_width=23, n_blocks=9):
  for i in range(1, n_blocks):
    ax.axvline(i * block_width - 0.5, color='white', lw=0.7, ls='--', alpha=0.75)


def add_block_labels(ax, labels, block_width=23):
  for i, label in enumerate(labels):
    x = i * block_width + block_width / 2
    ax.text(x, 195, str(label), color='white', ha='center', va='center',
            fontsize=7.5, fontweight='bold', 
            bbox=dict(facecolor='black', alpha=0.45, edgecolor='none', boxstyle='round,pad=0.2'))


def plot_resting_state_methods():
  fig = plt.figure(figsize=(FIG_WIDTH_INCHES * 1.85, FIG_HEIGHT_INCHES * 1.75), dpi=DPI)
  fig.patch.set_facecolor('white')

  fig.text(
    0.04,
    0.955,
    'Resting-state spectrum formatting',
    ha='left',
    va='top',
    fontsize=16,
    fontweight='bold',
    color=DARK,
  )
  # fig.text(
  #   0.04,
  #   0.915,
  #   'Conceptual transformation from a 224$\\times$488 time-frequency map to CNN-compatible 224$\\times$224 inputs',
  #   ha='left',
  #   va='top',
  #   fontsize=9,
  #   color='#5f6b73',
  # )

  original = generate_synthetic_spectrum(224, 488, seed=42, kind='rest')
  resized = resize_image(original)
  chunks = np.array_split(original, 3, axis=1)
  padded_chunks = [pad_image_width(chunk, pad_value=0.0) for chunk in chunks]
  avg_chunks = np.mean(padded_chunks, axis=0)

  # 
  original_rect = [0.040, 0.360, 0.220, 0.280]  # Centered vertically at 0.5, same size as cnn_rect
  resize_rect = [0.380, 0.690, 0.240, 0.220]
  chunk_rect = [0.380, 0.390, 0.240, 0.220]     # Centered vertically at 0.5
  avg_rect = [0.380, 0.090, 0.240, 0.220]
  cnn_rect = [0.740, 0.360, 0.220, 0.280]       # Centered vertically at 0.5, same size as original_rect

  add_image_panel(fig, original_rect, original, 'Original spectrum', '224$\\times$488 volumes', cmap='viridis')
  add_image_panel(fig, resize_rect, resized, 'Resize', 'Bicubic interpolation to 224$\\times$224', cmap='viridis')
  
  chunk_axes = add_stacked_image_panel(fig, chunk_rect, padded_chunks, 'Chunk + pad', '3 separate chunks with padding', cmap='viridis')
  for i, (ax_c, source_chunk) in enumerate(zip(chunk_axes, chunks)):
    add_pad_overlay(ax_c, source_chunk.shape[1], alpha_mult=1.0 - 0.25 * i)
    
  ax_avg = add_image_panel(fig, avg_rect, avg_chunks, 'Average chunks', 'Mean of padded temporal chunks', cmap='viridis')
  add_pad_overlay(ax_avg, max(chunk.shape[1] for chunk in chunks))
  add_cnn_box(fig, cnn_rect)

  connect(fig, rect_mid_right(original_rect), rect_mid_left(resize_rect), color=ORANGE, rad=-0.10)
  connect(fig, rect_mid_right(original_rect), rect_mid_left(chunk_rect), color=ORANGE)
  connect(fig, rect_mid_right(original_rect), rect_mid_left(avg_rect), color=ORANGE, rad=0.10)

  connect(fig, rect_mid_right(resize_rect), rect_mid_left(cnn_rect), color=BLUE, rad=0.10)
  connect(fig, rect_mid_right(chunk_rect), rect_mid_left(cnn_rect), color=BLUE)
  connect(fig, rect_mid_right(avg_rect), rect_mid_left(cnn_rect), color=BLUE, rad=-0.10)

  fig.text(0.680, 0.515, 'feature\nextraction', ha='center', va='center', fontsize=8.5, color=DARK_BLUE)
  fig.text(0.310, 0.515, 'formatting\nstrategy', ha='center', va='center', fontsize=8.5, color='#9a5b14')

  fig.savefig(OUT_DIR / 'resting_state_methods.pdf', bbox_inches='tight')
  fig.savefig(OUT_DIR / 'resting_state_methods.png', bbox_inches='tight')
  plt.close(fig)


def make_task_blocks():
  blocks = []
  for i in range(9):
    width = 23 if i < 4 else 24
    block = generate_synthetic_spectrum(224, width, seed=100 + i, kind='task')
    blocks.append(block[:, :23])
  return blocks


def plot_task_methods():
  fig = plt.figure(figsize=(FIG_WIDTH_INCHES * 1.85, FIG_HEIGHT_INCHES * 1.75), dpi=DPI)
  fig.patch.set_facecolor('white')

  fig.text(
    0.04,
    0.955,
    'Task-block spectrum formatting',
    ha='left',
    va='top',
    fontsize=16,
    fontweight='bold',
    color=DARK,
  )
  # fig.text(
  #   0.04,
  #   0.915,
  #   'Conceptual alternatives for mapping nine short task blocks to 224$\\times$224 CNN inputs',
  #   ha='left',
  #   va='top',
  #   fontsize=9,
  #   color='#5f6b73',
  # )

  blocks = make_task_blocks()
  resized_blocks = [resize_image(block) for block in blocks]
  avg_resized = np.mean(resized_blocks, axis=0)
  padded_blocks = [pad_image_width(block, pad_value=0.0) for block in blocks]
  avg_short = pad_image_width(np.mean(blocks, axis=0), pad_value=0.0)

  rng = np.random.default_rng(42)
  permutation = rng.permutation(len(blocks))
  permuted_blocks = [blocks[i] for i in permutation]
  concatenated = np.concatenate(permuted_blocks, axis=1)
  concat_padded = pad_image_width(concatenated, pad_value=0.0)

  block_strip = np.concatenate(blocks, axis=1)

  source_rect = [0.040, 0.380, 0.220, 0.240]  # Smaller height, centered vertically at 0.5
  m4_rect = [0.380, 0.690, 0.240, 0.220]
  m5_rect = [0.380, 0.390, 0.240, 0.220]      # Centered vertically at 0.5
  m6_rect = [0.380, 0.090, 0.240, 0.220]
  cnn_rect = [0.740, 0.380, 0.220, 0.240]      # Smaller height, centered vertically at 0.5

  ax_source = add_image_panel(fig, source_rect, block_strip, 'Task blocks', '9 blocks $\\times$ 23 volumes\nafter alignment', cmap='plasma')
  add_block_boundaries(ax_source)
  add_block_labels(ax_source, range(1, 10))

  m4_axes = add_stacked_image_panel(fig, m4_rect, padded_blocks[:3], 'Pad each short block', '3 of 9 blocks visually stacked', cmap='plasma')
  for i, ax_m4 in enumerate(m4_axes):
    add_pad_overlay(ax_m4, 23, alpha_mult=1.0 - 0.25 * i)

  ax_m5 = add_image_panel(fig, m5_rect, avg_short, 'Average then pad', 'Mean of the 9 blocks', cmap='plasma')
  add_pad_overlay(ax_m5, 23)

  ax_m6 = add_image_panel(fig, m6_rect, concat_padded, 'Concatenate blocks + pad', 'All blocks shuffled and\nplaced contiguously', cmap='plasma')
  add_block_boundaries(ax_m6)
  add_block_labels(ax_m6, permutation + 1)
  add_pad_overlay(ax_m6, concatenated.shape[1])

  add_cnn_box(fig, cnn_rect)

  method_rects = [m4_rect, m5_rect, m6_rect]
  for i, rect in enumerate(method_rects):
    connect(fig, rect_mid_right(source_rect), rect_mid_left(rect), color=ORANGE, lw=1.9, rad=-(1 - i) * 0.10)
    connect(fig, rect_mid_right(rect), rect_mid_left(cnn_rect), color=BLUE, lw=1.9, rad=-(i - 1) * 0.10)

  fig.text(0.310, 0.515, 'block\nformatting', ha='center', va='center', fontsize=8.5, color='#9a5b14')

  fig.savefig(OUT_DIR / 'task_blocks_methods.pdf', bbox_inches='tight')
  fig.savefig(OUT_DIR / 'task_blocks_methods.png', bbox_inches='tight')
  plt.close(fig)


if __name__ == '__main__':
  plot_resting_state_methods()
  plot_task_methods()
  print(f'Saved pretty formatting figures to {OUT_DIR}')
