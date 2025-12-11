"""
This code plots the figure 4 using the npy values
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
    ax.text((x1 + x2) * 0.5, y-0.0025, text, ha='center', va='bottom', fontsize=7)


# ################################################ Define the panel here ###############################################
textwidth_in_inches = 8  # Or fetch from LaTeX if you know the document class
fig = plt.figure(figsize=(textwidth_in_inches, 9.25))

# 5 rows, 13 columns grid
gs = fig.add_gridspec(
    nrows=5,
    ncols=24,
    height_ratios=[1.25, 1.15, 0.8, 0.8, 0.8],  # one ratio per row
    hspace=0.8,
    wspace=0.5,
    left=0.08,
    right=0.98,
    top=0.95,
    bottom=0.06
)

# ---------- Row 1 ----------
# Panel A
ax_a = fig.add_subplot(gs[0, :11])

# Panel B 3 coluns
gs_b = gs[0, 13:].subgridspec(nrows=1, ncols=3, wspace=0.125)
ax_b = [fig.add_subplot(gs_b[0, i]) for i in range(3)]

# ---------- Row 2 ----------
gs_c_left = gs[1, :12].subgridspec(nrows=1, ncols=3, wspace=0.125)
gs_c_right = gs[1, 12:].subgridspec(nrows=1, ncols=3, wspace=0.125)

ax_c = []
for i in range(3):
    ax_c.append(fig.add_subplot(gs_c_left[0, i]))
for i in range(3):
    ax_c.append(fig.add_subplot(gs_c_right[0, i]))

# ---------- Rows 3,4,5: 3 x 6 (2D ax) ----------
# Subgrid over rows 3–5 (indices 2,3,4) and all columns
gs_bottom = gs[2:, :].subgridspec(nrows=3, ncols=6, wspace=0.125, hspace=0.1)

ax_d = np.empty((3, 6), dtype=object)  # 3 rows × 6 plots (3 main + 3 surprise)

for i in range(3):      # illusions -> rows 2,3,4 in the big figure
    for j in range(6):  # left block
        ax_d[i, j] = fig.add_subplot(gs_bottom[i, j])

# ###################################################### Panel A #######################################################

y_tac_n0_c0_baseline = np.load(
    os.path.join('results', 'regression', 'baseline', 'surprise_set', 'global_error_tac_y.npy'))
y_tac_n0_c0_joint = np.load(
    os.path.join('results', 'regression', 'joint', 'surprise_set', 'global_error_tac_y.npy'))
y_tac_n0_c0_wocm = np.load(
    os.path.join('results', 'regression', 'wo-cm', 'surprise_set', 'global_error_tac_y.npy'))
y_tac_n0_c0_cm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'surprise_set', 'global_error_tac_y.npy'))

y_vis_n0_c0_baseline = np.load(
    os.path.join('results', 'regression', 'baseline', 'surprise_set', 'global_error_vis_y.npy'))
y_vis_n0_c0_joint = np.load(
    os.path.join('results', 'regression', 'joint', 'surprise_set', 'global_error_vis_y.npy'))
y_vis_n0_c0_wocm = np.load(
    os.path.join('results', 'regression', 'wo-cm', 'surprise_set', 'global_error_vis_y.npy'))
y_vis_n0_c0_cm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'surprise_set', 'global_error_vis_y.npy'))

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
    ax_a.bar(indices + offset, object_error[i],
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
ax_a.set_xticks(indices + bar_width + 0.2)
ax_a.set_xticklabels(group_labels)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

ax_a.set_ylabel('NMSE', fontsize=10)
ax_a.set_xticklabels(group_labels, fontsize=10)

handles = [plt.Line2D([0], [0], color=methods_color[i], lw=4, alpha=1.0, linestyle=(0, ())) for i in
           range(len(methods_label))]
legend = ax_a.legend(handles, methods_label, loc="upper center", bbox_to_anchor=(0.5, 1.3),
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
            add_sig(ax_a, x_ref, x_m, current_y, symbol)
            current_y += y_step

ax_a.yaxis.set_major_locator(plt.MaxNLocator(5))

# ############################################### Panel B #############################################################
y_tac_n0_c0_wcm_late = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_tac_y_n0_c0.npy'))
y_tac_n0_c0_wcm_early = np.load(os.path.join('results', 'regression', 'w-cm', 'early', 'global_error_tac_y_n_n0_c0.npy'))

target_tac_outputs = [y_tac_n0_c0_wcm_early, y_tac_n0_c0_wcm_late]

methods_label = [r'w-CM (Early)', r'w-CM (Late)']
methods_color = [
    "#9A169A",  # Orange
    "#377EB8"   # Blue
]

tac_properties = ['Stiffness', 'Mass', 'Surf. Friction']

time_s = np.linspace(0, 30, 98)
# Intrinsic Properties
for i in range(0, len(target_tac_outputs)):
    method = target_tac_outputs[i]
    mean_error = np.mean(method, axis=0)
    std_error = np.std(method, axis=0)
    for prop in range(0, len(tac_properties)):
        ax_b[prop].plot(time_s, mean_error[:, prop], color=methods_color[i], label=methods_label[i], linewidth=1.0, linestyle='solid')
        ax_b[prop].fill_between(time_s, mean_error[:, prop] - std_error[:, prop]/10,  mean_error[:, prop] + std_error[:, prop]/10, color=methods_color[i], alpha=0.1)
        ax_b[prop].set_title(tac_properties[prop], fontsize=10)
        ax_b[prop].set_xlabel('Time (s)', fontsize=10)
        ax_b[prop].grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
        if prop != 0:
            ax_b[prop].tick_params(labelleft=False)

# Compute global min and max
tymin, tymax = float('inf'), -float('inf')

for i in range(len(target_tac_outputs)):
    method = target_tac_outputs[i]
    mean_error = np.mean(method, axis=0)
    std_error = np.std(method, axis=0)
    lower = np.min(mean_error - std_error/10)
    upper = np.max(mean_error + std_error/10)
    tymin = min(tymin, lower)
    tymax = max(tymax, upper)

# Optional rounding
tymin = np.floor(tymin * 100) / 100
tymax = np.ceil(tymax * 100) / 100

# Set uniform ticks
tac_yticks = np.linspace(tymin, tymax, 4)
import matplotlib.ticker as ticker

# Formatter for 2 decimal places
two_decimal_formatter = ticker.FormatStrFormatter('%.2f')

import matplotlib.ticker as ticker
for i in range(0, 3):
    ax_b[i].set_yticks(tac_yticks)
    ax_b[i].set_ylim(tymin, tymax)
    ax_b[i].yaxis.set_major_formatter(two_decimal_formatter)

ax_b[0].set_ylabel('NMSE', fontsize=10)
handles = [plt.Line2D([0], [0], color=methods_color[i], lw=4, alpha=1.0, linestyle=(0, ())) for i in range(len(methods_label))]
legend = ax_b[1].legend(handles, methods_label, loc="upper center", bbox_to_anchor=(0.5, 1.35),
                    ncol=len(methods_label), frameon=False,
                    columnspacing=0.9, handlelength=1.25, handletextpad=0.5, fontsize=10)


# ####################################################### Panel C #######################################################
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
legend = ax_c[2].legend(handles, methods_label, loc="upper center", bbox_to_anchor=(0.75, 1.375),
                    ncol=len(methods_label), frameon=False,
                    columnspacing=0.9, handlelength=1.25, handletextpad=0.5, fontsize=10)

# ######################################################## Panel D ####################################################
time_s = np.linspace(0, 30, 98)
filter_color = '#377EB8'
filter_label = 'Filtered'
cm_label = 'CM-V2T'
cm_color = '#A65628'
gt_color = 'k'
methods_color = [filter_color, cm_color, gt_color]
methods_label = [r'Filtered $\mathbf{y}^{T}$', r'CM-V2T', r'GT']
lf_data_aligned = np.load(os.path.join('results', 'latent_filtering_aligned_set.npz'))
filtered_y_al = lf_data_aligned['trans_tac_y']
cm_y_al = lf_data_aligned['trans_vis2tac_y']
object_label_al = lf_data_aligned['object_label']
lf_data_surprise = np.load(os.path.join('results', 'latent_filtering_surprise_set.npz'))
filtered_y_sp = lf_data_surprise['trans_tac_y']
cm_y_sp = lf_data_surprise['trans_vis2tac_y']
object_label_sp = lf_data_surprise['object_label']

for illusion in range(0, 3):
    for prop in range(0, 3):
        # Filtered Output
        ax_d[illusion, prop].plot(time_s, filtered_y_al[illusion, :, prop, 0], color=filter_color, linewidth=1.0,
                                linestyle='solid', label='Filtered')
        ax_d[illusion, prop].errorbar(time_s, filtered_y_al[illusion, :, prop, 0],
                                    yerr=filtered_y_al[illusion, :, prop, 1] / 10,
                                    fmt='none',
                                    capsize=1,
                                    label=filter_label,
                                    ecolor=filter_color,
                                    alpha=0.25)
        # CM Prediction
        ax_d[illusion, prop].plot(time_s, cm_y_al[illusion, :, prop, 0], color=cm_color, linewidth=1.0, linestyle='solid', label='CM-V2T')
        ax_d[illusion, prop].errorbar(time_s, cm_y_al[illusion, :, prop, 0],
                                    yerr=cm_y_al[illusion, :, prop, 1] / 10,
                                    fmt='none',
                                    capsize=1,
                                    label=cm_label,
                                    ecolor=cm_color,
                                    alpha=0.2)
        # GT Label
        ax_d[illusion, prop].plot(time_s, object_label_al[illusion, :98, prop + 3], color='k', linewidth=2.0, linestyle='--', label='GT')
        ax_d[illusion, prop].set_ylim([-0.1, 1.1])
        ax_d[illusion, prop].grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
    for prop in range(0, 3):
        ax_d[illusion, prop + 3].plot(time_s, filtered_y_sp[illusion, :, prop, 0], color=filter_color, linewidth=1.0,
                                linestyle='solid', label='Filtered')
        ax_d[illusion, prop + 3].errorbar(time_s, filtered_y_sp[illusion, :, prop, 0],
                                        yerr=filtered_y_sp[illusion, :, prop, 1] / 10,
                                        fmt='none',
                                        capsize=2,
                                        label=filter_label,
                                        ecolor=filter_color,
                                        alpha=0.25)
        # CM Prediction
        ax_d[illusion, prop + 3].plot(time_s, cm_y_sp[illusion, :, prop, 0], color=cm_color, linewidth=1.0,
                                linestyle='solid', label='CM-V2T')
        ax_d[illusion, prop + 3].errorbar(time_s, cm_y_sp[illusion, :, prop, 0],
                                        yerr=cm_y_sp[illusion, :, prop, 1] / 10,
                                        fmt='none',
                                        capsize=1,
                                        label=cm_label,
                                        ecolor=cm_color,
                                        alpha=0.2)

        ax_d[illusion, prop + 3].plot(time_s, object_label_sp[illusion, :98, prop + 3], color='k',
                                    linewidth=2.0,
                                    linestyle='--', label='GT')

        ax_d[illusion, prop + 3].set_ylim([-0.1, 1.1])
        ax_d[illusion, prop + 3].grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

for i in range(3):
    for j in range(6):
        if j != 0:
            ax_d[i, j].tick_params(labelleft=False)  # Hide y-axis labels
        if i != 2:
            ax_d[i, j].tick_params(labelbottom=False)  # Hide x-axis labels

for prop in range(0, 3):
    ax_d[0, prop].set_title(tac_properties[prop], fontsize=10)
    ax_d[0, prop + 3].set_title(tac_properties[prop], fontsize=10)

for i in range(3):
    ax_d[i, 0].set_ylabel('Norm. Value', fontsize=10)

for j in range(6):
    ax_d[2, j].set_xlabel('Time (s)', fontsize=10)

handles = [plt.Line2D([0], [0], color=methods_color[i], lw=4, linestyle=(0, ())) for i in range(len(methods_label))]
legend = ax_d[0, 2].legend(handles, methods_label, loc="upper center", bbox_to_anchor=(0.75, 1.35), ncol=len(methods_label),
                    columnspacing=0.9, frameon=False, handlelength=1.25, handletextpad=0.5, fontsize=10)


# #####################################################################################################################
ax_a.text(-0.15, 1.275, r'{a}',
          transform=ax_a.transAxes,
          ha='right', va='center', fontsize=12)

ax_b[0].text(-0.45, 1.275, r'{b}',
          transform=ax_b[0].transAxes,
          ha='right', va='center', fontsize=12)

ax_c[0].text(-0.45, 1.275, r'{c}',
          transform=ax_c[0].transAxes,
          ha='right', va='center', fontsize=12)

ax_d[0, 0].text(-0.45, 1.275, r'{d}',
          transform=ax_d[0, 0].transAxes,
          ha='right', va='center', fontsize=12)

# Reference axis: bottom row, column 1
ax_ref = ax_d[2, 0]

# Arrow spanning columns 1–3 (in axis coordinates)
ax_ref.annotate(
    '',
    xy=(3.25, -0.35),      # arrow end (to the right)
    xytext=(0.0, -0.35),  # arrow start
    xycoords='axes fraction',
    textcoords='axes fraction',
    arrowprops=dict(
        arrowstyle='<->',
        lw=0.5,
        color='k'
    ),
    clip_on=False
)

# Centered text on the arrow with white background
ax_ref.text(
    3.25/2, -0.35,  # Changed y-coordinate to match arrow position
    r'Aligned set',
    transform=ax_ref.transAxes,
    ha='center',
    va='center',
    fontsize=10,
    bbox=dict(facecolor='white', edgecolor='none', pad=2)
)

# Reference axis: bottom row, column 4
ax_ref = ax_d[2, 3]

# Arrow spanning columns 1–3 (in axis coordinates)
ax_ref.annotate(
    '',
    xy=(3.25, -0.35),      # arrow end (to the right)
    xytext=(0.0, -0.35),  # arrow start
    xycoords='axes fraction',
    textcoords='axes fraction',
    arrowprops=dict(
        arrowstyle='<->',
        lw=0.5,
        color='k'
    ),
    clip_on=False
)

# Centered text on the arrow with white background
ax_ref.text(
    3.25/2, -0.35,  # Changed y-coordinate to match arrow position
    r'Surprise set',
    transform=ax_ref.transAxes,
    ha='center',
    va='center',
    fontsize=10,
    bbox=dict(facecolor='white', edgecolor='none', pad=2)
)
# plt.show()
plt.savefig(os.path.join('results', 'Figure_4.pdf'), dpi=500)
plt.savefig(os.path.join('results', 'Figure_4.svg'), dpi=500)