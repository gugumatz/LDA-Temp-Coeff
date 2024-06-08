import re
import sys
import openpyxl
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import colorcet as cc
from sklearn.base import clone
import matplotlib.pyplot as plt
from itertools import compress
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ================ LDA method for residue mapping in IDPs ================= #

# Read NMR data
try:
    NMR_data = pd.read_excel(sys.argv[1], engine="openpyxl")
except 'FileNotFoundError':
    NMR_data = pd.read_excel(sys.argv[1], engine="odf")

# Read fasta file
with open(sys.argv[2], 'r') as f:
    fasta = [line.rstrip('\n') for line in f]
fasta = ''.join(fasta)

# ============================ Pre-processing ============================= #

# Get residue code names to separate training and testing sets
code_name = NMR_data.iloc[:, 0]
NMR_data = NMR_data.drop(['Residue'], axis=1)
header = NMR_data.columns.tolist()

# Separate AAT and SSN from residues code names
classes = []
SSN_testing = []
idxs_bool = np.zeros(code_name.shape, dtype=bool)
for idx, i in enumerate(code_name):
    if isinstance(i, str):
        match = re.match(r"(\w)(\d+)", i, re.I)
        items = match.groups()
        classes.append(items[0].upper())
    else:
        SSN_testing.append(i)
        idxs_bool[idx] = ~idxs_bool[idx]

# Create training and testing sets
test_set = NMR_data.loc[idxs_bool, :]
train_set = NMR_data.loc[~idxs_bool, :]
nuclei = test_set.columns.tolist()

# Get AATs of training set
AATs_unique = list(set(classes))
AATs_unique.sort()

# Count occurrences of each amino acid type in training set and fasta file
a = dict(Counter(classes))
d = collections.defaultdict(int)
fasta_list = list(fasta)
fasta_list.sort()
for i in fasta_list:
    d[i] += 1

# Check which AATs have already been fully classified and which are missing
inds_delete = []
for i in d:
    if i in a:
        if a[i] == d[i]:
            print(f"All residues of AAT < {i} > already classified. This AA type will be removed from the training"
                  f"set to avoid misclassifications.")
            inds = [j for j, x in enumerate(classes) if x == i]
            inds_delete.extend(inds)
            AATs_unique.remove(i)
        elif a[i] > d[i]:
            print(f"WARNING! There are misclassifications in the training set regarding AAT < {i} >")
            print(f"Number of residues of AAT < {i} > in the training set = {a[i]}")
            print(f"Number of residues of AAT < {i} > in the protein = {d[i]}")
    else:
        print(f"There are no residues of AAT < {i} > in the training set!\n"
              f"It will not be possible to classify residues of this type.")

if inds_delete:
    classes = np.delete(classes, inds_delete)
    train_set = train_set.drop(inds_delete, axis=0)
train_set.insert(len(nuclei), "amino", classes, True)
classes = train_set["amino"].copy()

print(f"\nThere are {len(train_set)} residues for training and {len(test_set)} residues for testing.")

# ============================ Classification ============================= #

# Get column positions of GLY and PRO
if 'G' in AATs_unique:
    G_idx = AATs_unique.index('G')
if 'P' in AATs_unique:
    P_idx = AATs_unique.index('P')

# Initialization
probs = np.ndarray(shape=(len(test_set), len(AATs_unique)))
LDA_clf = LinearDiscriminantAnalysis()
test_nan = np.array(test_set.isnull())
combs_nan = np.unique(test_nan, axis=0)
nuclei_ix = list(range(0, len(nuclei)))

# Loop over combinations
for i in combs_nan:
    idxs_nan = np.all(test_nan == i, axis=1)
    nuclei_meas = list(compress(nuclei, ~i))
    test_missing_comb = test_set[idxs_nan][nuclei_meas]

    # Classify residues with all NaN values
    if np.all(i):
        a = np.empty((idxs_nan.sum(), probs.shape[1]))
        a[:] = np.nan
        probs[idxs_nan, :] = a
        continue

    # Classify residues that could be GLY or PRO
    elif {"HB", "CB", "HN"}.isdisjoint(nuclei_meas) and all(x in AATs_unique for x in {'G', 'P'}):
        idxs = train_set[nuclei_meas].isnull().any(axis=1)
        train_set_aux = train_set[~idxs][nuclei_meas].copy()
        labels = classes[~idxs].copy()

        clf_clone = clone(LDA_clf)
        clf_clone.fit(train_set_aux, labels)
        probs[idxs_nan, :] = clf_clone.predict_proba(test_missing_comb)

    # Classify residues that could be GLY
    elif {"HB", "CB"}.isdisjoint(set(nuclei_meas)) and 'G' in AATs_unique:
        train_set_aux = train_set[train_set["amino"] != 'P'][nuclei_meas].copy()
        labels = classes[classes != 'P'].copy()
        idxs = train_set_aux.isnull().any(axis=1)
        train_set_aux = train_set_aux[~idxs]
        labels = labels[~idxs]

        clf_clone = clone(LDA_clf)
        clf_clone.fit(train_set_aux, labels)
        probs_aux = clf_clone.predict_proba(test_missing_comb)
        if 'P' in AATs_unique:
            probs[idxs_nan, :] = np.c_[probs_aux[:, :P_idx], np.zeros((idxs_nan.sum(), 1)), probs_aux[:, P_idx:]]
        else:
            probs[idxs_nan, :] = probs_aux

    # Classify residues that could be PRO
    elif {"HN"}.isdisjoint(set(nuclei_meas)) and 'P' in AATs_unique:
        train_set_aux = train_set[train_set["amino"] != 'G'][nuclei_meas].copy()
        labels = classes[classes != 'G'].copy()
        idxs = train_set_aux.isnull().any(axis=1)
        train_set_aux = train_set_aux[~idxs]
        labels = labels[~idxs]

        clf_clone = clone(LDA_clf)
        clf_clone.fit(train_set_aux, labels)
        probs_aux = clf_clone.predict_proba(test_missing_comb)
        if 'G' in AATs_unique:
            probs[idxs_nan, :] = np.c_[probs_aux[:, :G_idx], np.zeros((idxs_nan.sum(), 1)), probs_aux[:, G_idx:]]
        else:
            probs[idxs_nan, :] = probs_aux

    # Classify other residues
    else:
        train_set_aux = train_set.query('amino!="P" or amino!="G"')[nuclei_meas].copy()
        labels = classes[np.logical_or(classes != 'P', classes != 'G')].copy()
        idxs = train_set_aux.isnull().any(axis=1)
        train_set_aux = train_set_aux[~idxs]
        labels = labels[~idxs]

        clf_clone = clone(LDA_clf)
        clf_clone.fit(train_set_aux, labels)
        probs_aux = clf_clone.predict_proba(test_missing_comb)
        if all(x in AATs_unique for x in {'G', 'P'}):
            probs[idxs_nan, :] = np.c_[probs_aux[:, :G_idx], np.zeros((idxs_nan.sum(), 1)),
            probs_aux[:, G_idx:P_idx - 1], np.zeros((idxs_nan.sum(), 1)), probs_aux[:, P_idx - 1:]]
        elif 'P' in AATs_unique:
            probs[idxs_nan, :] = np.c_[probs_aux[:, :P_idx], np.zeros((idxs_nan.sum(), 1)), probs_aux[:, P_idx:]]
        elif 'G' in AATs_unique:
            probs[idxs_nan, :] = np.c_[probs_aux[:, :G_idx], np.zeros((idxs_nan.sum(), 1)), probs_aux[:, G_idx:]]
        else:
            probs[idxs_nan, :] = probs_aux

    # Set probabilities threshold (for more clear results)
    if {'CB'}.issubset(test_missing_comb):
        probs_aux = probs[idxs_nan]
        probs_aux[probs_aux < 0.05] = 0
        probs[idxs_nan] = probs_aux
    else:
        probs_aux = probs[idxs_nan]
        probs_aux[probs_aux < 0.1] = 0
        probs[idxs_nan] = probs_aux

# Write probabilities matrix to excel file
df = pd.DataFrame(probs, index=SSN_testing, columns=AATs_unique)
df.to_excel('Probabilities.xlsx')

# ================================= Plot ================================== #

# Legends and labels
x_labels = list(np.unique(AATs_unique))
x_pos = list(range(0, len(x_labels)))
y_pos = list(range(0, len(test_set)))
legend_labels = x_labels.copy()

# Remove extra legends
for i in reversed(range(0, probs.shape[1])):
    if (probs[:, i] == 0).all():
        legend_labels.pop(i)

# Create lists for plotting
x_vals = []
y_vals = []
p_vals = []
for i in y_pos:
    for j in x_pos:
        if probs[i, j] != 0:
            x_vals.append(x_pos[j])
            y_vals.append(y_pos[i])
            p_vals.append(probs[i, j])

# Create Data Frame using lists
plot_data = pd.DataFrame(columns=['label', 'residue', 'probability'])
plot_data['Label'] = x_vals
plot_data['Residue'] = y_vals
plot_data['Probability'] = p_vals

# Create stable color palette (always same colors for each AAT)
dict_col = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6,
            'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13,
            'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
palette = sns.color_palette(cc.glasbey, n_colors=20)
col_nums = [dict_col[x] for x in legend_labels]
col_list = []
for i in col_nums:
    col_list.append(palette[i])

# Create figure
h = sns.relplot(x="Label", y="Residue", hue="Label", size="Probability", markers=True, edgecolor=None,
                sizes=(40, 300), alpha=.5, palette=col_list, data=plot_data, aspect=0.4, legend='brief')

# Formatting and save
h.ax.margins(x=0.05, y=0.02)
h.despine(top=False, right=False)
plt.ylabel("Spin system", size=20)
plt.yticks(y_pos, SSN_testing, fontsize=8)
plt.xlabel("LDA classification", size=20)
plt.xticks(x_pos, x_labels, fontsize=10, rotation=60)
plt.grid(axis='x', color='k', linestyle='-', linewidth=0.2)
plt.grid(axis='y', color='k', linestyle=':', linewidth=0.2)
sns.move_legend(h, "upper right", bbox_to_anchor=(0.74, 0.8))
for t, l in zip(h._legend.texts, ['Labels'] + legend_labels):
    t.set_text(l)
h.fig.set_dpi(100)
h.fig.set_figheight(8)
h.fig.set_figwidth(10)
h.savefig('Probabilities.png', dpi=300)
plt.show()
