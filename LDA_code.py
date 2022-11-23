import re
import sys
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt
from itertools import compress
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ================ LDA method for residue mapping in IDPs ================= #

# Read NMR data
try:
    NMR_data = pd.read_excel(sys.argv[1])
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
AATs_training = []
SSN_testing = []
idxs_bool = np.zeros(code_name.shape, dtype=bool)
for idx, i in enumerate(code_name):
    if isinstance(i, str):
        match = re.match(r"(\w)(\d+)", i, re.I)
        items = match.groups()
        AATs_training.append(items[0].upper())
    else:
        SSN_testing.append(i)
        idxs_bool[idx] = ~idxs_bool[idx]

# Create training and testing sets
test_set = NMR_data.loc[idxs_bool, :]
test_set = test_set.to_numpy()
train_set = NMR_data.loc[~idxs_bool, :]
train_set = train_set.reset_index()

# Get AATs of training set
AATs = list(set(AATs_training))
AATs.sort()

# Count occurrences of each amino acid type in training set and fasta file
a = dict(Counter(AATs_training))
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
            print('All residues of AA type <', i, '> already classified.'
                  ' This AA type will be removed from the training set to avoid misclassifications.')
            inds = [j for j, x in enumerate(AATs_training) if x == i]
            inds_delete.extend(inds)
            AATs.remove(i)
        elif a[i] > d[i]:
            print('WARNING! There are misclassifications in the training set regarding AA type <', i, '>')
            print('Number of AAs of type <', i, '> in the training set = ', a[i])
            print('Number of AAs of type <', i, '> in the protein = ', d[i])
    else:
        print('No residues of AAT <', i, '> in training set!'
              ' It will not be possible to classify residues of this type.')
AATs_training = np.delete(AATs_training, inds_delete)
train_set.drop(inds_delete, axis=0, inplace=True)
train_set = train_set.drop(['index'], axis=1)

# Get column numbers of GLY and PRO
if 'G' in AATs:
    Gidx = AATs.index('G')
if 'P' in AATs:
    Pidx = AATs.index('P')

# ============================= Organize Data ============================= #

miss_res = 0

# Separate GLY residues in the training set, if there are GLY in the test protein and HB or CB are input CSs
if np.logical_and('G' in AATs, np.logical_or('HB' in header, 'CB' in header)):
    idxs_gly = (AATs_training == 'G')
    data_gly = train_set.loc[idxs_gly, [x for x in header if x not in ['HB', 'CB']]]
    gly_classes = AATs_training[idxs_gly]
    train_set = train_set.loc[np.invert(idxs_gly), :]
    AATs_training = AATs_training[np.invert(idxs_gly)]

    # Discard entries missing CSs
    idxs = np.invert(data_gly.isnull().any(axis=1))
    gly_classes = gly_classes[idxs]
    train_gly = data_gly.loc[idxs, :]
    train_gly = train_gly.to_numpy()
    miss_res = miss_res + 1

# Separate PRO residues in the training set, if there are PRO in the test protein and H or N are input CSs
if np.logical_and('P' in AATs, np.logical_or('H' in header, 'N' in header)):
    idxs_pro = (AATs_training == 'P')
    data_pro = train_set.loc[idxs_pro, [x for x in header if x not in ['H', 'N']]]
    pro_classes = AATs_training[idxs_pro]
    train_set = train_set.loc[np.invert(idxs_pro), :]
    AATs_training = AATs_training[np.invert(idxs_pro)]

    # Discard entries missing CSs
    idxs = np.invert(data_pro.isnull().any(axis=1))
    pro_classes = pro_classes[idxs]
    train_pro = data_pro.loc[idxs, :]
    train_pro = train_pro.to_numpy()
    miss_res = miss_res + 1

# Discard entries missing CSs in the main training set
idxs = np.invert(train_set.isnull().any(axis=1))
train_classes_all = AATs_training[idxs]
train_set_all = train_set.loc[idxs, :]
train_set_all = train_set_all.to_numpy()

# Separate test set entries with all CSs from those missing CSs
idxs_missing = np.isnan(test_set).any(axis=1)
test_set_all = test_set[~idxs_missing, :]

# ==================== Classify test set with all CSs ===================== #

Probabilities = np.ndarray(shape=(len(test_set), len(set(train_classes_all))))  # Matrix of AAT probabilities
Mdl = LinearDiscriminantAnalysis()  # Classification model
Mdl.fit(train_set_all, train_classes_all)  # Train the model
if any(~idxs_missing):
    Probabilities[~idxs_missing] = Mdl.predict_proba(test_set_all)

if miss_res == 2:
    Probabilities = np.c_[Probabilities[:, :Gidx], np.zeros((len(test_set), 1)), Probabilities[:, Gidx:Pidx - 1],
                          np.zeros((len(test_set), 1)), Probabilities[:, Pidx - 1:]]
elif miss_res == 1:
    if np.logical_and('G' in AATs, np.logical_or('HB' in header, 'CB' in header)):
        Probabilities = np.c_[Probabilities[:, :Gidx], np.zeros((len(test_set), 1)), Probabilities[:, Gidx:]]
    elif np.logical_and('P' in AATs, np.logical_or('H' in header, 'N' in header)):
        Probabilities = np.c_[Probabilities[:, :Pidx], np.zeros((len(test_set), 1)), Probabilities[:, Pidx:]]

# ================== Classify test set with missing CSs =================== #

# Find index of HB and CB columns
ord_gly = []
if 'HB' in header:
    ord_gly.append(header.index('HB'))
if 'CB' in header:
    ord_gly.append(header.index('CB'))

# Find index of H and N columns
ord_pro = []
if 'H' in header:
    ord_pro.append(header.index('H'))
if 'N' in header:
    ord_pro.append(header.index('N'))

HB_CB_set = {"HB", "CB"}
H_N_set = {"H", "N"}

header_cols = list(range(0, len(header)))
num_missing = [i for i, x in enumerate(idxs_missing) if x]

# Loop across residues with missing CHs
for i in num_missing:
    comb = np.isnan(test_set[i, :])
    CSs = set(compress(header, comb))  # CSs missing in the current residue

    # Classify empty entries
    if np.all(comb):
        Probabilities[i, :] = np.zeros(Probabilities.shape[1])
        continue

    # Classify residues missing all HB, CB, H and N
    elif (HB_CB_set | H_N_set).issubset(CSs):
        # If the test protein has both GLY and PRO in its primary sequence
        if all(x in AATs for x in {'G', 'P'}):
            cols1 = [x for x in header_cols if x not in ord_gly]
            cols2 = [x for x in header_cols if x not in ord_pro]
            comb_aux1 = comb[cols1]
            comb_aux2 = comb[cols2]
            train_set = np.concatenate((train_set_all[:, ~comb], train_gly[:, ~comb_aux1], train_pro[:, ~comb_aux2]))
            train_classes = np.concatenate((train_classes_all, gly_classes, pro_classes))

            Mdl_both = LinearDiscriminantAnalysis()
            Mdl_both.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl_both.predict_proba(observation)
            Probabilities[i, :] = Probs_aux

        # If the test protein only has GLY in its primary sequence
        elif 'G' in AATs:
            cols1 = [x for x in header_cols if x not in ord_gly]
            comb_aux1 = comb[cols1]
            train_set = np.concatenate((train_set_all[:, ~comb], train_gly[:, ~comb_aux1]))
            train_classes = np.concatenate((train_classes_all, gly_classes))

            Mdl = LinearDiscriminantAnalysis()
            Mdl.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl.predict_proba(observation)
            Probs_aux = np.concatenate((Probs_aux[0, :Gidx], np.array([0]), Probs_aux[0, Gidx:]))
            Probabilities[i, :] = Probs_aux

        # If the test protein only has PRO in its primary sequence
        elif 'P' in AATs:
            cols2 = [x for x in header_cols if x not in ord_pro]
            comb_aux2 = comb[cols2]
            train_set = np.concatenate((train_set_all[:, ~comb], train_pro[:, ~comb_aux2]))
            train_classes = np.concatenate((train_classes_all, pro_classes))

            Mdl = LinearDiscriminantAnalysis()
            Mdl.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl.predict_proba(observation)
            Probs_aux = np.concatenate((Probs_aux[0, :Pidx], np.array([0]), Probs_aux[0, Pidx:]))
            Probabilities[i, :] = Probs_aux

        # If the residue is missing all HB, CB, H and N, but the test protein doesn't have GLY nor PRO
        else:
            train_set = train_set_all[:, ~comb]
            train_classes = train_classes_all

            Mdl = LinearDiscriminantAnalysis()
            Mdl.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl.predict_proba(observation)
            Probabilities[i, :] = Probs_aux

    # Classify residues missing HB and CB
    # elif HB_CB_set.issubset(CSs):
    elif ({"HB"}.issubset(CSs) or {"CB"}.issubset(CSs)):
        # If the test protein has GLY in its primary sequence
        if 'G' in AATs:
            cols = [x for x in header_cols if x not in ord_gly]
            comb_aux = comb[cols]
            train_set = np.concatenate((train_set_all[:, ~comb], train_gly[:, ~comb_aux]))
            train_classes = np.concatenate((train_classes_all, gly_classes))

            Mdl_gly = LinearDiscriminantAnalysis()
            Mdl_gly.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl_gly.predict_proba(observation)
            # If the test protein also has PRO in its primary sequence
            if np.logical_and('P' in AATs, np.logical_or('H' in header, 'N' in header)):
                Probs_aux = np.concatenate((Probs_aux[0, :Pidx], np.array([0]), Probs_aux[0, Pidx:]))
            Probabilities[i, :] = Probs_aux

        # If the residue is missing HB and CB, but the test protein doesn't have GLY in its primary sequence
        else:
            train_set = train_set_all[:, ~comb]
            train_classes = train_classes_all

            Mdl = LinearDiscriminantAnalysis()
            Mdl.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl.predict_proba(observation)
            # If the test protein has PRO in its primary sequence
            if np.logical_and('G' in AATs, np.logical_or('HB' in header, 'CB' in header)):
                Probs_aux = np.concatenate((Probs_aux[0, :Pidx], np.array([0]), Probs_aux[0, Pidx:]))
            Probabilities[i, :] = Probs_aux

    # Classify residues missing H and N
    # elif H_N_set.issubset(CSs):
    elif ({"H"}.issubset(CSs) or {"N"}.issubset(CSs)):
        # If the test protein has PRO in its primary sequence
        if 'P' in AATs:
            cols = [x for x in header_cols if x not in ord_pro]
            comb_aux = comb[cols]
            train_set = np.concatenate((train_set_all[:, ~comb], train_pro[:, ~comb_aux]))
            train_classes = np.concatenate((train_classes_all, pro_classes))

            Mdl_pro = LinearDiscriminantAnalysis()
            Mdl_pro.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl_pro.predict_proba(observation)
            # If the test protein also has GLY in its primary sequence
            if np.logical_and('G' in AATs, np.logical_or('HB' in header, 'CB' in header)):
                Probs_aux = np.concatenate((Probs_aux[0, :Gidx], np.array([0]), Probs_aux[0, Gidx:]))
            Probabilities[i, :] = Probs_aux

        # If the residue is missing H and N, but the test protein doesn't have PRO in its primary sequence
        else:
            train_set = train_set_all[:, ~comb]
            train_classes = train_classes_all

            Mdl = LinearDiscriminantAnalysis()
            Mdl.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl.predict_proba(observation)
            # If it has GLY in its primary sequence
            if np.logical_and('P' in AATs, np.logical_or('H' in header, 'N' in header)):
                Probs_aux = np.concatenate((Probs_aux[0, :Gidx], np.array([0]), Probs_aux[0, Gidx:]))
            Probabilities[i, :] = Probs_aux

    # Classify residues missing any other combination of chemical shifts
    else:
        train_set = train_set_all[:, ~comb]
        Mdl_miss = LinearDiscriminantAnalysis()
        Mdl_miss.fit(train_set, train_classes_all)
        observation = test_set[i, ~comb].reshape(1, -1)
        Probs_aux = Mdl_miss.predict_proba(observation)
        if miss_res == 1:
            if 'G' in AATs:
                Probs_aux = np.concatenate([Probs_aux[0, :Gidx], np.array([0]), Probs_aux[0, Gidx:]])
            else:
                Probs_aux = np.concatenate([Probs_aux[0, :Pidx], np.array([0]), Probs_aux[0, Pidx:]])
        elif miss_res == 2:
            Probs_aux = np.concatenate([Probs_aux[0, :Gidx], np.array([0]), Probs_aux[0, Gidx:Pidx - 1],
                                        np.array([0]), Probs_aux[0, Pidx - 1:]])
        Probabilities[i, :] = Probs_aux

# Set threshold of probabilities (for more clear results)
for i in range(len(test_set)):
    comb = np.isnan(test_set[i, :])
    CSs = set(compress(header, comb))
    if {'CB'}.issubset(CSs):
        Probabilities[i, Probabilities[i, :] < 0.05] = 0
    else:
        Probabilities[i, Probabilities[i, :] < 0.1] = 0

# Write probabilities matrix to excel file
Probs = pd.DataFrame(Probabilities, index=SSN_testing, columns=AATs)
Probs.to_excel('Probabilities.xlsx')

# ================================= Plot ================================== #

x_labels = list(np.unique(AATs))
x_pos = list(range(0, len(x_labels)))
y_pos = list(range(0, len(test_set)))
legend_labels = x_labels.copy()

for i in reversed(range(0, Probabilities.shape[1])):
    if (Probabilities[:, i] == 0).all():
        legend_labels.pop(i)

x_vals = []
y_vals = []
p_vals = []

for i in y_pos:
    for j in x_pos:
        if Probabilities[i, j] != 0:
            x_vals.append(x_pos[j])
            y_vals.append(y_pos[i])
            p_vals.append(Probabilities[i, j])

plot_data = pd.DataFrame(columns=['label', 'residue', 'probability'])
plot_data['Label'] = x_vals
plot_data['Residue'] = y_vals
plot_data['Probability'] = p_vals

palette = sns.color_palette(cc.glasbey, n_colors=len(legend_labels))
h = sns.relplot(x="Label",
                y="Residue",
                hue="Label",
                size="Probability",
                sizes=(40, 300), alpha=.5, palette=palette, data=plot_data, aspect=0.3)

h.ax.margins(x=0.05, y=0.02)
h.despine(top=False, right=False)
plt.ylabel("Spin system", size=20)
plt.yticks(y_pos, SSN_testing, fontsize=8)
plt.xlabel("LDA classification", size=20)
plt.xticks(x_pos, x_labels, fontsize=10, rotation=60)
plt.grid(axis='x', color='k', linestyle='-', linewidth=0.2)
plt.grid(axis='y', color='k', linestyle=':', linewidth=0.2)
sns.move_legend(h, "upper right", bbox_to_anchor=(0.65, 0.8))
for t, l in zip(h._legend.texts, ['Labels'] + legend_labels):
    t.set_text(l)

h.fig.set_dpi(100)
h.fig.set_figheight(15)
h.fig.set_figwidth(25)
h.savefig('Probabilities.svg', dpi=100)
plt.show()
