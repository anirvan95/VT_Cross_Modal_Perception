"""
This code plots the figure 3 using the npy values
"""
import os
import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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


def p_to_stars(p):
    """helper: convert p-value to stars"""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


def add_sig(ax, x1, x2, y, text, line_h=0.005):
    """Draw a bracket between x1 and x2 at height y and write text above it."""
    ax.plot([x1, x1, x2, x2], [y, y + line_h, y + line_h, y], lw=0.5, color='k')
    ax.text((x1 + x2) * 0.5, y+0.001, text, ha='center', va='bottom', fontsize=9)


# ################################################ Define the panel here ###############################################
textwidth_in_inches = 8  # Or fetch from LaTeX if you know the document class

# 3 Panels: 30-70 in first row and complete in bottom row
fig = plt.figure(figsize=(textwidth_in_inches, 6.5))
# 2 rows, 12 columns grid
gs = fig.add_gridspec(
    nrows=2,
    ncols=13,
    height_ratios=[1, 0.6],  # 5) Made second row shorter
    hspace=0.45,
    wspace=0.5,
    left=0.08,
    right=0.98,
    top=0.92,
    bottom=0.08
)

# ---------- Row 1 ----------
# Panel A (30% width)
ax_a = fig.add_subplot(gs[0, :5])

# Panel B (70% width)
ax_b = fig.add_subplot(gs[0, 6:])

# ---------- Row 2 ----------
gs_c_left = gs[1, :6].subgridspec(nrows=1, ncols=3, wspace=0.125)
gs_c_right = gs[1, 6:].subgridspec(nrows=1, ncols=3, wspace=0.125)

ax_c = []
for i in range(3):
    ax_c.append(fig.add_subplot(gs_c_left[0, i]))
for i in range(3):
    ax_c.append(fig.add_subplot(gs_c_right[0, i]))

# ############################################## Panel A ###############################################################
logreg_class_accuracies = np.load(os.path.join('results', 'classification', 'logreg_accuracies.npy'))
logreg_class_stds = np.load(os.path.join('results', 'classification', 'logreg_stds.npy'))

legend_methods_label_A = [r'Baseline', r'Joint', r'wo-CM', r'w-CM']
legend_methods_color_A = [
    "#D81B60",  # Pink
    "#4DAF4A",  # Green
    "#CC5308",  # Orange
    "#377EB8"  # Blue
]

methods_label_A = [r'$\mathbf{y}^{VT}$', r'$\mathbf{y}^{VT}$', r'$\mathbf{y}^{V}$', r'$\mathbf{y}^{T}$',
                   r'$\mathbf{y}^{V,T}$',
                   r'$\mathbf{y}^{V}$', r'$\mathbf{y}^{T}$', r'$\mathbf{y}^{V,T}$']
methods_color_A = ["#D81B60",
                   "#4DAF4A",
                   "#CC5308",
                   "#CC5308",
                   "#CC5308",
                   "#377EB8",
                   "#377EB8",
                   "#377EB8", ]

# Re-arrange based on label
logreg_class_results = np.array(np.concatenate(
    [logreg_class_accuracies[0:1], logreg_class_accuracies[1:2], logreg_class_accuracies[5:],
     logreg_class_accuracies[2:5]]))
logreg_class_stds_rearranged = np.array(
    np.concatenate([logreg_class_stds[0:1], logreg_class_stds[1:2], logreg_class_stds[5:], logreg_class_stds[2:5]]))

bar_width = 0.4
group_spacing = 0.70

x_positions = [0, 1, 2, 2.5, 3, 4, 4.5, 5]
bars = ax_a.bar(x_positions, logreg_class_results, width=bar_width, color=methods_color_A, alpha=1.0,
                yerr=logreg_class_stds_rearranged, capsize=2,  error_kw={
             'elinewidth': 0.75,
             'capthick': 0.75,
             'linestyle': 'dashed',
             'dash_capstyle': 'butt'
         })

ax_a.set_ylabel('Classification Accuracy', fontsize=10)
ax_a.set_ylim(0, 1.1)
ax_a.set_xticks(x_positions)
ax_a.set_xticklabels(methods_label_A, rotation=0, ha='center', fontsize=10)

# Legend for panel a
handles = [plt.Line2D([0], [0], color=legend_methods_color_A[i], lw=4, alpha=1.0, linestyle=(0, ())) for i in
           range(len(legend_methods_label_A))]
legend = ax_a.legend(handles, legend_methods_label_A, loc="upper center", bbox_to_anchor=(0.5, 1.1),
                     ncol=len(legend_methods_label_A), frameon=False,
                     columnspacing=0.9, handlelength=1.25, handletextpad=0.5, fontsize=10)


# ############################################## Panel B ###############################################################
y_tac_n0_c0_baseline = np.load(
    os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_tac_y_n0_c0.npy'))
y_tac_n0_c0_joint = np.load(
    os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_tac_y_n0_c0.npy'))
y_tac_n0_c0_wocm = np.load(
    os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_tac_y_n0_c0.npy'))
y_tac_n0_c0_cm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_tac_y_n0_c0.npy'))

y_vis_n0_c0_baseline = np.load(
    os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_vis_y_n0_c0.npy'))
y_vis_n0_c0_joint = np.load(
    os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_vis_y_n0_c0.npy'))
y_vis_n0_c0_wocm = np.load(
    os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_vis_y_n0_c0.npy'))
y_vis_n0_c0_cm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_vis_y_n0_c0.npy'))

results = {
    'Baseline': np.concatenate([np.mean(y_vis_n0_c0_baseline, axis=1), np.mean(y_tac_n0_c0_baseline, axis=1)], axis=-1),
    'Joint': np.concatenate([np.mean(y_vis_n0_c0_joint, axis=1), np.mean(y_tac_n0_c0_joint, axis=1)], axis=-1),
    'wo-CM': np.concatenate([np.mean(y_vis_n0_c0_wocm, axis=1), np.mean(y_tac_n0_c0_wocm, axis=1)], axis=-1),
    'w-CM': np.concatenate([np.mean(y_vis_n0_c0_cm, axis=1), np.mean(y_tac_n0_c0_cm, axis=1)], axis=-1)
}
alpha = 0.05
methods = list(results.keys())
ref_method = 'w-CM'
n_props = results[ref_method].shape[1]
n_methods = len(methods)

test_results = {}
for m in methods:
    if m == ref_method:
        continue
    test_results[m] = {'pvals': np.ones(n_props), 'symbol': ['ns'] * n_props}
    for prop_idx in range(n_props):
        x = results[ref_method][:, prop_idx]
        y = results[m][:, prop_idx]
        assert x.shape[0] == y.shape[0], "Paired tests require same number of runs."
        stat, p = stats.ttest_rel(x, y, nan_policy='omit')
        test_results[m]['pvals'][prop_idx] = p
        test_results[m]['symbol'][prop_idx] = p_to_stars(p)

# Print summary
print("Significance (w-CM vs others) per property:")
for m in methods:
    if m == ref_method: continue
    print(f"\nComparing w-CM vs {m}:")
    for prop_idx in range(n_props):
        p = test_results[m]['pvals'][prop_idx]
        star = test_results[m]['symbol'][prop_idx]
        mean_ref = results[ref_method][:, prop_idx].mean()
        mean_m = results[m][:, prop_idx].mean()
        print(f"  Property {prop_idx}: p={p:.4g}, {star}, means: {ref_method}={mean_ref:.4f}, {m}={mean_m:.4f}")

# ######################################################## Basic Comparison ############################################
target_tac_outputs = [y_tac_n0_c0_baseline, y_tac_n0_c0_joint, y_tac_n0_c0_wocm, y_tac_n0_c0_cm]
target_vis_outputs = [y_vis_n0_c0_baseline, y_vis_n0_c0_joint, y_vis_n0_c0_wocm, y_vis_n0_c0_cm]

methods_label = [r'Baseline', r'Joint', r'wo-CM', r'w-CM']
methods_color = [
    "#D81B60",  # Pink
    "#4DAF4A",  # Green
    "#CC5308",  # Orange
    "#377EB8"  # Blue
]
vis_properties = ['Shape', 'Size', 'Vis.\nTexture']  # 3) Two-line label
tac_properties = ['Stiffness', 'Mass', 'Surf.\nFriction']  # 3) Two-line label

bar_width = 0.4
group_spacing = 0.80
num_subgroups, num_groups = 4, 6
group_labels = vis_properties + tac_properties
subgroup_labels = methods_label

object_error = np.zeros((len(methods_label), len(vis_properties) + len(tac_properties)))
object_error_std = np.zeros((len(methods_label), len(vis_properties) + len(tac_properties)))

# Extrinsic Properties
for i in range(len(target_vis_outputs)):
    method = target_vis_outputs[i]
    mean_error = np.mean(method, axis=0)
    for prop in range(len(vis_properties)):
        object_error[i, prop] = np.mean(mean_error[:, prop])
        object_error_std[i, prop] = np.std(mean_error[:, prop])

# Intrinsic Properties
for i in range(len(target_tac_outputs)):
    method = target_tac_outputs[i]
    mean_error = np.mean(method, axis=0)
    for prop in range(len(tac_properties)):
        object_error[i, prop + len(vis_properties)] = np.mean(mean_error[:, prop])
        object_error_std[i, prop + len(vis_properties)] = np.std(mean_error[:, prop])

# Compute positions
indices = np.arange(num_groups) * (num_subgroups * bar_width + group_spacing)
bar_centers = np.zeros((num_subgroups, num_groups))
for i in range(num_subgroups):
    bar_centers[i, :] = indices + i * bar_width
ref_idx = methods_label.index(ref_method)

# Plot each subgroup
for i in range(num_subgroups):
    offset = i * bar_width
    ax_b.bar(indices + offset, object_error[i],
           yerr=object_error_std[i],
           width=bar_width,
           label=subgroup_labels[i],
           color=methods_color[i],
           alpha=1.0,
           capsize=2,
           error_kw={
               'elinewidth': 0.75,
               'capthick': 0.75,
               'linestyle': 'dashed',
               'dash_capstyle': 'butt'
           })

# X-axis ticks centered on each group
ax_b.set_xticks(indices + bar_width + 0.2)
ax_b.set_xticklabels(group_labels)

ax_b.set_ylabel('NMSE', fontsize=10)
ax_b.set_xticklabels(group_labels, fontsize=10)

handles = [plt.Line2D([0], [0], color=methods_color[i], lw=4, alpha=1.0, linestyle=(0, ())) for i in
           range(len(methods_label))]
legend = ax_b.legend(handles, methods_label, loc="upper center", bbox_to_anchor=(0.5, 1.1),
                    ncol=len(methods_label), frameon=False,
                    columnspacing=0.9, handlelength=1.25, handletextpad=0.5, fontsize=10)




# Add significance annotations
for prop_idx in range(num_groups):
    tops = object_error[:, prop_idx] + object_error_std[:, prop_idx]
    y_base = np.nanmax(tops)
    y_step = 0.025
    current_y = y_base + y_step

    for m in methods_label:
        if m == ref_method:
            continue
        p = test_results[m]['pvals'][prop_idx]
        symbol = test_results[m]['symbol'][prop_idx]
        if p < alpha:
            m_idx = methods_label.index(m)
            x_ref = bar_centers[ref_idx, prop_idx]
            x_m = bar_centers[m_idx, prop_idx]
            add_sig(ax_b, x_ref, x_m, current_y, symbol)
            current_y += y_step

ax_b.yaxis.set_major_locator(plt.MaxNLocator(5))

# ############################################# Panel C ############################################################
target_tac_outputs = [y_tac_n0_c0_joint, y_tac_n0_c0_wocm, y_tac_n0_c0_cm]
target_vis_outputs = [y_vis_n0_c0_joint, y_vis_n0_c0_wocm, y_vis_n0_c0_cm]

methods_label = [r'Joint', r'wo-CM', r'w-CM']
methods_color = [
    "#4DAF4A",  # Green
    "#CC5308",  # Orange
    "#377EB8"   # Blue
]

vis_properties = ['Shape', 'Size', 'Vis. Texture']
tac_properties = ['Stiffness', 'Mass', 'Surf. Friction']
prop_style = ['solid', 'solid', 'solid']
time_s = np.linspace(0, 30, 98)

# Extrinsic Properties
for i in range(len(target_vis_outputs)):
    method = target_vis_outputs[i]
    mean_error = np.mean(method, axis=0)
    std_error = np.std(method, axis=0)
    for prop in range(len(vis_properties)):
        ax_c[prop].plot(time_s, mean_error[:, prop], color=methods_color[i], label=methods_label[i], linewidth=1.0, linestyle=prop_style[prop])
        ax_c[prop].fill_between(time_s, mean_error[:, prop] - std_error[:, prop]/10, mean_error[:, prop] + std_error[:, prop]/10, color=methods_color[i], alpha=0.1)
        ax_c[prop].set_title(vis_properties[prop], fontsize=10)
        ax_c[prop].set_xlabel('Time (s)', fontsize=10)
        ax_c[prop].grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
        if prop != 0:
            ax_c[prop].tick_params(labelleft=False)

# Intrinsic Properties
for i in range(len(target_tac_outputs)):
    method = target_tac_outputs[i]
    mean_error = np.mean(method, axis=0)
    std_error = np.std(method, axis=0)
    for prop in range(len(tac_properties)):
        ax_c[prop+3].plot(time_s, mean_error[:, prop], color=methods_color[i], label=methods_label[i], linewidth=1.0, linestyle=prop_style[prop])
        ax_c[prop+3].fill_between(time_s, mean_error[:, prop] - std_error[:, prop]/10, mean_error[:, prop] + std_error[:, prop]/10, color=methods_color[i], alpha=0.1)
        ax_c[prop+3].set_title(tac_properties[prop], fontsize=10)
        ax_c[prop+3].set_xlabel('Time (s)', fontsize=10)
        ax_c[prop+3].grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
        ax_c[prop+3].tick_params(labelleft=False)

# Compute global min and max
vymin, vymax = float('inf'), -float('inf')
tymin, tymax = float('inf'), -float('inf')

for i in range(len(target_vis_outputs)):
    method = target_vis_outputs[i]
    mean_error = np.mean(method, axis=0)
    std_error = np.std(method, axis=0)
    lower = np.min(mean_error - std_error/50)
    upper = np.max(mean_error + std_error/50)
    vymin = min(vymin, lower)
    vymax = max(vymax, upper)

for i in range(len(target_tac_outputs)):
    method = target_tac_outputs[i]
    mean_error = np.mean(method, axis=0)
    std_error = np.std(method, axis=0)
    lower = np.min(mean_error - std_error/50)
    upper = np.max(mean_error + std_error/50)
    tymin = min(tymin, lower)
    tymax = max(tymax, upper)

vymin = np.floor(vymin * 100) / 100
vymax = np.ceil(vymax * 100) / 100
tymin = np.floor(tymin * 100) / 100
tymax = np.ceil(tymax * 100) / 100

vis_yticks = np.linspace(tymin, tymax, 4)
tac_yticks = np.linspace(tymin, tymax, 4)

two_decimal_formatter = ticker.FormatStrFormatter('%.2f')

for i in range(3):
    ax_c[i].set_yticks(vis_yticks)
    ax_c[i].set_ylim(tymin, tymax)
    ax_c[i].yaxis.set_major_formatter(two_decimal_formatter)

for i in range(3, 6):
    ax_c[i].set_yticks(tac_yticks)
    ax_c[i].set_ylim(tymin, tymax)
    ax_c[i].yaxis.set_major_formatter(two_decimal_formatter)

ax_c[0].set_ylabel('NMSE', fontsize=10)
handles = [plt.Line2D([0], [0], color=methods_color[i], lw=4, alpha=1.0, linestyle=(0, ())) for i in range(len(methods_label))]
legend = fig.legend(handles, methods_label, loc="upper center", bbox_to_anchor=(0.5, 0.42),
                    ncol=len(methods_label), frameon=False,
                    columnspacing=0.9, handlelength=1.25, handletextpad=0.5, fontsize=10)


# Panel label (a), aligned with the legend row
ax_a.text(-0.15, 1.1, r'{a}',
          transform=ax_a.transAxes,
          ha='right', va='center', fontsize=12)

ax_b.text(-0.1, 1.1, r'{b}',
          transform=ax_b.transAxes,
          ha='right', va='center', fontsize=12)

fig.text(0.025, 0.42, r'{c}',
         ha='right', va='center', fontsize=12)

plt.savefig(os.path.join('results', 'Figure_3.pdf'), dpi=500)
plt.savefig(os.path.join('results', 'Figure_3.svg'), dpi=500)



