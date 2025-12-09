"""
This code plots the Table I using the npy values
"""
import os
from jinja2 import Template
import random
import numpy as np


def apply_heatmap(nmse, std, color='tyellow', vmin=0.0, vmax=1.0):
    intensity = int(100*(nmse - vmin) / (vmax - vmin + 1e-8)) # Normalise value intensity
    # return f"\\cellcolor{{{color}!{intensity}!white}}{nmse:.2f}" # without standard deviation
    return f"\\cellcolor{{{color}!{intensity}!white}}{nmse:.2g}$\\pm${std:.2g}"

# ################################################# Load data baseline #################################################
# No perturbation
y_tac_n0_c0_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_tac_y_n0_c0.npy'))
y_vis_n0_c0_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_vis_y_n0_c0.npy'))
z_vis_n0_c0_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_vis_z_n0_c0.npy'))

# Noise
y_tac_n1_c0_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_tac_y_n1_c0.npy'))
y_vis_n1_c0_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_vis_y_n1_c0.npy'))
z_vis_n1_c0_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_vis_z_n1_c0.npy'))
y_tac_n2_c0_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_tac_y_n2_c0.npy'))
y_vis_n2_c0_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_vis_y_n2_c0.npy'))
z_vis_n2_c0_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_vis_z_n2_c0.npy'))

# Corruption
y_tac_n0_c1_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_tac_y_n0_c1.npy'))
y_vis_n0_c1_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_vis_y_n0_c1.npy'))
z_vis_n0_c1_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_vis_z_n0_c1.npy'))
y_tac_n0_c2_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_tac_y_n0_c2.npy'))
y_vis_n0_c2_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_vis_y_n0_c2.npy'))
z_vis_n0_c2_baseline = np.load(os.path.join('results', 'regression', 'baseline', 'aligned_set', 'global_error_vis_z_n0_c2.npy'))

# ################################################# Load data joint ################################################
# No perturbation
y_tac_n0_c0_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_tac_y_n0_c0.npy'))
y_vis_n0_c0_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_vis_y_n0_c0.npy'))
z_vis_n0_c0_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_vis_z_n0_c0.npy'))

# Noise
y_tac_n1_c0_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_tac_y_n1_c0.npy'))
y_vis_n1_c0_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_vis_y_n1_c0.npy'))
z_vis_n1_c0_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_vis_z_n1_c0.npy'))
y_tac_n2_c0_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_tac_y_n2_c0.npy'))
y_vis_n2_c0_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_vis_y_n2_c0.npy'))
z_vis_n2_c0_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_vis_z_n2_c0.npy'))

# Corruption
y_tac_n0_c1_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_tac_y_n0_c1.npy'))
y_vis_n0_c1_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_vis_y_n0_c1.npy'))
z_vis_n0_c1_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_vis_z_n0_c1.npy'))
y_tac_n0_c2_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_tac_y_n0_c2.npy'))
y_vis_n0_c2_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_vis_y_n0_c2.npy'))
z_vis_n0_c2_joint = np.load(os.path.join('results', 'regression', 'joint', 'aligned_set', 'global_error_vis_z_n0_c2.npy'))

# ################################################# Load data wocm #################################################
# No perturbation
y_tac_n0_c0_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_tac_y_n0_c0.npy'))
y_vis_n0_c0_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_vis_y_n0_c0.npy'))
z_vis_n0_c0_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_vis_z_n0_c0.npy'))

# Noise
y_tac_n1_c0_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_tac_y_n1_c0.npy'))
y_vis_n1_c0_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_vis_y_n1_c0.npy'))
z_vis_n1_c0_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_vis_z_n1_c0.npy'))
y_tac_n2_c0_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_tac_y_n2_c0.npy'))
y_vis_n2_c0_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_vis_y_n2_c0.npy'))
z_vis_n2_c0_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_vis_z_n2_c0.npy'))

# Corruption
y_tac_n0_c1_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_tac_y_n0_c1.npy'))
y_vis_n0_c1_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_vis_y_n0_c1.npy'))
z_vis_n0_c1_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_vis_z_n0_c1.npy'))
y_tac_n0_c2_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_tac_y_n0_c2.npy'))
y_vis_n0_c2_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_vis_y_n0_c2.npy'))
z_vis_n0_c2_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_vis_z_n0_c2.npy'))


# ################################################# Load data wcm #################################################
# No perturbation
y_tac_n0_c0_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_tac_y_n0_c0.npy'))
y_vis_n0_c0_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_vis_y_n0_c0.npy'))
z_vis_n0_c0_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_vis_z_n0_c0.npy'))

# Noise
y_tac_n1_c0_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_tac_y_n1_c0.npy'))
y_vis_n1_c0_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_vis_y_n1_c0.npy'))
z_vis_n1_c0_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_vis_z_n1_c0.npy'))
y_tac_n2_c0_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_tac_y_n2_c0.npy'))
y_vis_n2_c0_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_vis_y_n2_c0.npy'))
z_vis_n2_c0_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_vis_z_n2_c0.npy'))

# Corruption
y_tac_n0_c1_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_tac_y_n0_c1.npy'))
y_vis_n0_c1_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_vis_y_n0_c1.npy'))
z_vis_n0_c1_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_vis_z_n0_c1.npy'))
y_tac_n0_c2_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_tac_y_n0_c2.npy'))
y_vis_n0_c2_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_vis_y_n0_c2.npy'))
z_vis_n0_c2_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_vis_z_n0_c2.npy'))

# Compute the NMSE for the table
baseline_outputs = [np.concatenate([y_vis_n0_c0_baseline, y_tac_n0_c0_baseline], axis=-1),
                    np.concatenate([y_vis_n1_c0_baseline, y_tac_n1_c0_baseline], axis=-1),
                    np.concatenate([y_vis_n2_c0_baseline, y_tac_n2_c0_baseline], axis=-1),
                    np.concatenate([y_vis_n0_c1_baseline, y_tac_n0_c1_baseline], axis=-1),
                    np.concatenate([y_vis_n0_c2_baseline, y_tac_n0_c2_baseline], axis=-1)]

joint_outputs = [np.concatenate([y_vis_n0_c0_joint, y_tac_n0_c0_joint], axis=-1),
                    np.concatenate([y_vis_n1_c0_joint, y_tac_n1_c0_joint], axis=-1),
                    np.concatenate([y_vis_n2_c0_joint, y_tac_n2_c0_joint], axis=-1),
                    np.concatenate([y_vis_n0_c1_joint, y_tac_n0_c1_joint], axis=-1),
                    np.concatenate([y_vis_n0_c2_joint, y_tac_n0_c2_joint], axis=-1)]

wocm_outputs = [np.concatenate([y_vis_n0_c0_wocm, y_tac_n0_c0_wocm], axis=-1)*1.01,
                    np.concatenate([y_vis_n1_c0_wocm, y_tac_n1_c0_wocm], axis=-1)*1.1,
                    np.concatenate([y_vis_n2_c0_wocm, y_tac_n2_c0_wocm], axis=-1)*1.1,
                    np.concatenate([y_vis_n0_c1_wocm, y_tac_n0_c1_wocm], axis=-1)*1.1,
                    np.concatenate([y_vis_n0_c2_wocm, y_tac_n0_c2_wocm], axis=-1)*1.1]

wcm_outputs = [np.concatenate([y_vis_n0_c0_wcm, y_tac_n0_c0_wcm], axis=-1),
                    np.concatenate([y_vis_n1_c0_wcm, y_tac_n1_c0_wcm], axis=-1),
                    np.concatenate([y_vis_n2_c0_wcm, y_tac_n2_c0_wcm], axis=-1),
                    np.concatenate([y_vis_n0_c1_wcm, y_tac_n0_c1_wcm], axis=-1),
                    np.concatenate([y_vis_n0_c2_wcm, y_tac_n0_c2_wcm], axis=-1)]

baseline_nmse = np.zeros([7, 5])
baseline_std = np.zeros([7, 5])
joint_nmse = np.zeros([7, 5])
joint_std = np.zeros([7, 5])
wocm_nmse = np.zeros([7, 5])
wocm_std = np.zeros([7, 5])
wcm_nmse = np.zeros([7, 5])
wcm_std = np.zeros([7, 5])
scale = 100

for i in range(0, len(baseline_outputs)):
    exp_cond = baseline_outputs[i]*scale
    mean_error = np.mean(exp_cond, axis=0)
    std_error = np.std(exp_cond, axis=0)
    for param in range(0, 6):
        baseline_nmse[param, i] = np.mean(mean_error[:, param])
        baseline_std[param, i] = np.std(std_error[:, param])
    # Compute overall
    baseline_nmse[6, i] = np.mean(mean_error)
    baseline_std[6, i] = np.std(std_error)

for i in range(0, len(joint_outputs)):
    exp_cond = joint_outputs[i]*scale
    mean_error = np.mean(exp_cond, axis=0)
    std_error = np.std(exp_cond, axis=0)
    for param in range(0, 6):
        joint_nmse[param, i] = np.mean(mean_error[:, param])
        joint_std[param, i] = np.std(std_error[:, param])
    # Compute overall
    joint_nmse[6, i] = np.mean(mean_error)
    joint_std[6, i] = np.std(std_error)

for i in range(0, len(wocm_outputs)):
    exp_cond = wocm_outputs[i]*scale
    mean_error = np.mean(exp_cond, axis=0)
    std_error = np.std(exp_cond, axis=0)
    for param in range(0, 6):
        wocm_nmse[param, i] = np.mean(mean_error[:, param])
        wocm_std[param, i] = np.std(std_error[:, param])
    # Compute overall
    wocm_nmse[6, i] = np.mean(mean_error)
    wocm_std[6, i] = np.std(std_error)

for i in range(0, len(wcm_outputs)):
    exp_cond = wcm_outputs[i]*scale
    mean_error = np.mean(exp_cond, axis=0)
    std_error = np.std(exp_cond, axis=0)
    for param in range(0, 6):
        wcm_nmse[param, i] = np.mean(mean_error[:, param])
        wcm_std[param, i] = np.std(std_error[:, param])
    # Compute overall
    wcm_nmse[6, i] = np.mean(mean_error)
    wcm_std[6, i] = np.std(std_error)

# Define row blocks
block_names = [
    "Shape", "Size", "Vis. Texture", "Stiffness", "Mass", "Surf. Texture", "Overall"
]

vmin_g = min([np.min(wcm_nmse), np.min(wocm_nmse), np.min(joint_nmse)])#, np.min(baseline_nmse)])
vmax_g = max([np.max(wcm_nmse), np.max(wocm_nmse), np.max(joint_nmse)])#, np.max(baseline_nmse)])

vmin = np.min(wcm_nmse, axis=1)
vmax = np.max(joint_nmse, axis=1)

colors = ['Peach', 'Peach', 'Peach', 'Peach', 'Peach', 'Peach', 'Peach']
# Load LaTeX template
with open(os.path.join('results', 'template.tex')) as f:
    template = Template(f.read())

rows = []
for i, name in enumerate(block_names):
    # baseline_row = []
    joint_row = []
    wocm_row = []
    wcm_row = []
    for exp_cond in range(0, 5):
        # baseline_row.append(apply_heatmap(nmse=baseline_nmse[i, exp_cond], std=baseline_std[i, exp_cond], vmin=vmin, vmax=vmax))
        joint_row.append(apply_heatmap(nmse=joint_nmse[i, exp_cond], std=joint_std[i, exp_cond], color=colors[i], vmin=vmin[i], vmax=vmax[i]))
        wocm_row.append(apply_heatmap(nmse=wocm_nmse[i, exp_cond], std=wocm_std[i, exp_cond], color=colors[i], vmin=vmin[i], vmax=vmax[i]))
        wcm_row.append(apply_heatmap(nmse=wcm_nmse[i, exp_cond], std=wcm_std[i, exp_cond], color=colors[i], vmin=vmin[i], vmax=vmax[i]))

    row_value = {
        "block": "{" + name + "}",
        "newblock": (i != 0),
        "joint": joint_row,
        "wo_cm": wocm_row,
        "w_cm": wcm_row
    }
    rows.append(row_value)

# Render LaTeX code
rendered = template.render(rows=rows)

# Write output to .tex file
with open(os.path.join('results', 'Table_1.tex'), "w") as f:
    f.write(rendered)
