import os
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter, MaxNLocator

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath}
\usepackage{sfmath}
\renewcommand{\rmdefault}{cmss}
'''
params = {
    'text.usetex': True,
    'font.size': 9,
    'font.family': 'sans-serif',
    'axes.unicode_minus': False
}
plt.rcParams.update(params)

# ################################################# Load data baseline #################################################
z_vis_n0_c0_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_vis_z_n0_c0.npy'))
z_vis_n0_c0_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_vis_z_n0_c0.npy'))
z_vis_n0_c0_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_vis_z_n0_c0.npy'))
z_vis_n0_c0_cm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_vis_z_n0_c0.npy'))

target_vis_pose_outputs = [z_vis_n0_c0_baseline, z_vis_n0_c0_joint, z_vis_n0_c0_wocm, z_vis_n0_c0_cm]
methods_label = [r'Baseline', r'Joint', r'wo-CM', r'w-CM']

methods_color = [
    "#D81B60",  # Pink
    "#4DAF4A",  # Green
    "#CC5308",  # Orange
    "#377EB8"  # Blue
]

vis_pose_properties = [r'$\textnormal{Trans}_{x} (mm)$', r'$\textnormal{Trans}_{y} (mm) $',
                       r'$\textnormal{Trans}_{z} (mm)$',
                       r'$\textnormal{Rot}_{x} (cRad)$', r'$\textnormal{Rot}_{y} (cRad)$',
                       r'$\textnormal{Rot}_{z} (cRad)$']

prop_style = ['solid', 'solid', 'solid']

time_s = np.linspace(0, 30, 98)


def custom_y_formatter(x, pos):
    if x == 0:
        return "0"
    elif abs(x) < 1:
        return f"{x:.1f}"
    else:
        return f"{x:.0f}"


fig, ax = plt.subplots(1, 6, figsize=[8.3, 1.5], gridspec_kw={'wspace': 0.35})  # Increased vertical spacing with hspace

# Visual Pose Properties
for i in range(0, len(target_vis_pose_outputs)):
    method = target_vis_pose_outputs[i]
    mean_error = np.mean(method, axis=0)
    mean_error[:, 0:3] *= 1000
    mean_error[:, 3:] *= 100
    std_error = np.std(method, axis=0)
    for prop in range(0, len(vis_pose_properties)):
        ax[prop].plot(time_s, mean_error[:, prop], color=methods_color[i], label=methods_label[i], linewidth=1.0)
        ax[prop].fill_between(time_s, mean_error[:, prop] - std_error[:, prop] / 10,
                              mean_error[:, prop] + std_error[:, prop] / 10, color=methods_color[i], alpha=0.1)
        ax[prop].set_title(vis_pose_properties[prop], fontsize=10)
        ax[prop].grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        ax[prop].set_xlabel('Time (s)', fontsize=10)

# Y-axis labels
ax[0].set_ylabel('MSE', fontsize=10)

formatter = FuncFormatter(custom_y_formatter)

for axis in ax:
    axis.yaxis.set_major_formatter(formatter)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))  # Ensure ~4 ticks

# Legend
handles = [plt.Line2D([0], [0], color=methods_color[i], lw=4, alpha=1.0, linestyle=(0, ())) for i in
           range(len(methods_label))]
legend = fig.legend(handles, methods_label, loc="upper center", bbox_to_anchor=(0.5, 1.25),
                    ncol=len(methods_label), frameon=False,
                    columnspacing=0.9, handlelength=1.25, handletextpad=0.5, fontsize=10)

# Save
plt.savefig(os.path.join('results', 'appendix_Figure_1.pdf'), dpi=500, bbox_inches='tight')
plt.savefig(os.path.join('results', 'appendix_Figure_1.svg'), dpi=500, bbox_inches='tight')
plt.close()
