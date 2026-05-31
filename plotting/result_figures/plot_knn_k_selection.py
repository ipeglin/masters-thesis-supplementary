import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from plotting import plot_config
from plotting.plot_config import get_figs_output_dir

def plot_tie_breaker():
    """Plot simple concept of tie-breaker with odd K."""
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Points randomly scattered. 2 red, 2 blue inside k=4. 1 blue just outside.
    X = np.array([[-0.7, 0.6], [0.4, 0.8], [0.8, -0.3], [-0.5, -0.9], [1.15, 0.4]])
    y = np.array([0, 0, 1, 1, 1])
    query_point = np.array([[0, 0]])
    
    _cp = plot_config.CONCEPT_PALETTE
    cmap = ListedColormap([_cp["pale"], _cp["accent_a"]])
    colors = [_cp["mid"] if label == 0 else _cp["accent_a"] for label in y]
    
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=100, edgecolors='k', zorder=3)
    ax.scatter(query_point[:, 0], query_point[:, 1], c='gray', s=50, marker='h', edgecolors='k', label=r'Query Point $\mathbf{p}_d$', zorder=3)
    
    # Draw circles for k=4 and k=5
    circle4 = plt.Circle((0, 0), 1.08, color=_cp["accent_a"], fill=False, linestyle='--', label='k=4')
    circle5 = plt.Circle((0, 0), 1.26, color=_cp["accent_b"], fill=False, linestyle=':', label='k=5')
    
    ax.add_patch(circle4)
    ax.add_patch(circle5)
    
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title("Neighbourhood Tie-breaking")
    ax.axis('off')

    out_dir = get_figs_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "knn_tie_breaker.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def plot_k_decision_boundaries():
    """Plot effect of K magnitude on decision boundary."""
    X, y = make_moons(n_samples=200, noise=0.3, random_state=42)
    
    # k=1 (low, overfit/complex), k=15 (good), k=100 (high, underfit/smooth)
    # Note: ML theory says low k = overfit/noise-sensitive, high k = smooth/underfit
    ks = [1, 15, 99]
    titles = ['k=1 (Complex/Noisy)', 'k=15 (Generalizable balanced)', 'k=99 (Smooth/Underfit)']
    
    fig = plt.figure(figsize=(8, 7))
    gs = fig.add_gridspec(2, 4)
    ax1 = fig.add_subplot(gs[0, 0:2]) # Top left
    ax2 = fig.add_subplot(gs[1, 1:3]) # Bottom middle
    ax3 = fig.add_subplot(gs[0, 2:4]) # Top right
    axes = [ax1, ax2, ax3]
    
    _cp = plot_config.CONCEPT_PALETTE
    cmap_light = ListedColormap([_cp["wash"], _cp["accent_a"] + "66"])
    cmap_bold = ListedColormap([_cp["mid"], _cp["accent_a"]])
    
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    for ax, k, title in zip(axes, ks, titles):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X, y)
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
        
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    plt.tight_layout()
    out_dir = get_figs_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "knn_decision_boundaries.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

if __name__ == "__main__":
    plot_tie_breaker()
    plot_k_decision_boundaries()
