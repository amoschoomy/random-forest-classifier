# %%

# %pip install arm-mango
# %pip install pandas
# %pip install numpy
# %pip install --pre --extra-index https://pypi.anaconda.org/scientific-python-nightly-wheels/simple scikit-learn
# %pip install matplotlib
# %%
# import necessary libraries
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from mango import Tuner, scheduler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# %%
# load dataset

dna_meth = pd.read_csv('G11_lung_dna-meth.csv', sep=',')
gene_expr = pd.read_csv('G11_lung_gene-expr.csv', sep=',')
cpg_annot = pd.read_csv("CpG_hg19_annot.csv")


# %%
fig, axs = plt.subplots(nrows=1)

# Before normalization, demo to show the differences of transformation
axs.hist(gene_expr.iloc[:, 5], bins=15, edgecolor='black')
axs.set_title('Before normalisation')

axs.set_xlabel('Value of Col 5 of gene_expr data')
axs.set_ylabel('Frequency')
plt.tight_layout()
plt.show()
# %%
# Normalization of data by log2 transformation

dna_meth[dna_meth.columns[2:]] = np.log2(
    dna_meth[dna_meth.columns[2:]] + 0.001)
gene_expr[gene_expr.columns[2:]] = np.log2(
    gene_expr[gene_expr.columns[2:]] + 0.001)

# %%

fig, axs = plt.subplots(nrows=1)

axs.hist(gene_expr.iloc[:, 5], bins=15, edgecolor='black')
axs.set_title('After normalisation')

axs.set_xlabel('Value of Col 5 of gene expr data')
axs.set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# %%

# Get the top log fold change in cpg and genes

solid_tissue_normal_df = dna_meth[dna_meth['Label']
                                  == 'Solid Tissue Normal'].iloc[:, 2:]
primary_tumor_df = dna_meth[dna_meth['Label'] == 'Primary Tumor'].iloc[:, 2:]

# Calculate the mean methylation level for each CpG site in each group
mean_solid_tissue_normal = solid_tissue_normal_df.mean()
mean_primary_tumor = primary_tumor_df.mean()

# # Calculate the log-fold change for each CpG site
logfc_cpg = np.log10((mean_primary_tumor+0.0001) /
                     (mean_solid_tissue_normal+0.0001))

largestlfc_12cpgs = logfc_cpg.abs().nlargest(12)
largestlfc_12cpgs

solid_tissue_normal_df = gene_expr[gene_expr['Label']
                                   == 'Solid Tissue Normal'].iloc[:, 2:]
primary_tumor_df = gene_expr[gene_expr['Label'] == 'Primary Tumor'].iloc[:, 2:]

mean_solid_tissue_normal = solid_tissue_normal_df.mean()
mean_primary_tumor = primary_tumor_df.mean()

# # # Calculate the log-fold change for each CpG site
logfc_gene = np.log10((mean_primary_tumor + 0.0001) /
                      (mean_solid_tissue_normal + 0.0001))

largestlfc_10genes = logfc_gene.abs().nlargest(10)
largestlfc_10genes


# %%
# Define a mapping dictionary
label_mapping = {'Solid Tissue Normal': 0, 'Primary Tumor': 1}

# Use the dictionary to replace the labels in dna_meth
dna_meth['Label'] = dna_meth['Label'].replace(label_mapping)

# Use the dictionary to replace the labels in gene_expr
gene_expr['Label'] = gene_expr['Label'].replace(label_mapping)
# %%

# Feature selection by lit review (refer to document for details)
# last two genes are selected arbitrarily
genes_lit_review = ["KRAS.3845", "RAF1.5894"]
cpg_lit_review = []

# %%

# selecting gene that starts with RAS based on logfold change
ras_top8_genes = logfc_gene[logfc_gene.index.str.startswith(
    'RAS')].abs().nlargest(8)


for i in ras_top8_genes.index:
    genes_lit_review.append(i)
genes_lit_review
# %%

# selecting cpg that that has TP53,MGMT,CDKN2A,BRCA, MLH1, CDH1, TIMP-3, ER based pm logfold change
logfc_cpg

cpg_gene = ['TP53', 'MGMT', 'CDKN2A', 'BRCA1', 'MLH1', 'CDH1', 'TIMP-3', 'ER']

selected_rows = pd.Series([False]*len(cpg_annot), index=cpg_annot.index)

# Check each gene
for gene in cpg_gene:
    selected_rows |= (cpg_annot['UCSC_RefGene_Name'].str.startswith(
        gene) | cpg_annot['UCSC_RefGene_Name'].str.endswith(gene))

# Use the boolean Series to select the rows
selected = cpg_annot[selected_rows][['Unnamed: 0', 'UCSC_RefGene_Name']]
selected

# %%

# Convert the dictionary to a DataFrame
logfc_df = pd.DataFrame(list(logfc_cpg.items()),
                        columns=['Unnamed: 0', 'logFC'])

# Merge the selected DataFrame with the logfc DataFrame
merged_df = pd.merge(selected, logfc_df, on='Unnamed: 0')
merged_df

# Sort by 'logFC' in descending order and get the top 10 rows
top_8_cpg = merged_df.sort_values(by='logFC', ascending=False).head(8)


# %%

# combining the cg values
cg_values = largestlfc_12cpgs.index.tolist()
cg_values_lit_review = top_8_cpg['Unnamed: 0'].tolist()

combined_cg_values = cg_values + cg_values_lit_review
combined_cg_values
# %%
# combining the gene values
gene_values = largestlfc_10genes.index.tolist()
gene_values_lit_review = genes_lit_review

combined_gene_values = gene_values + gene_values_lit_review
combined_gene_values
# %%
# feature filtering
filtered_dna_meth = dna_meth[combined_cg_values + ['Label']]
filtered_dna_meth
# %%
# feature filtering
filtered_gene_expr = gene_expr[combined_gene_values + ['Label']]
filtered_gene_expr
# %%
# ---------------------------------------
# First we will deal with CpG sites datasets
# First we split the data into training and test sets

X = filtered_dna_meth.drop('Label', axis=1)
y = filtered_dna_meth['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# %%
# We will then run a hyperparameter tuning to find the best parameters for the model
param_space = dict(min_samples_split=range(2, 10), n_estimators=range(75, 150), min_samples_leaf=range(
    1, 10), max_features=["sqrt", "log2", None], criterion=["gini", "entropy"])


@scheduler.serial
def objective(**params):
    clf = RandomForestClassifier(**params)
    score = cross_val_score(clf, X_train, y_train, cv=10).mean()
    return score


tuner = Tuner(param_space, objective, {'num_iteration': 30})
results = tuner.maximize()
print('best parameters:', results['best_params'])
print('best accuracy:', results['best_objective'])
# %%

# fit the model with the best parameters
params = results['best_params']
params = {'criterion': 'gini', 'max_features': 'log2',
          'min_samples_leaf': 9, 'min_samples_split': 4, 'n_estimators': 104}
# or hardcode the best parameters
# clf_dna_meth = RandomForestClassifier(criterion="gini", max_features='log2',min_samples_leaf=1, min_samples_split=3, n_estimators=95)
clf_dna_meth = RandomForestClassifier(criterion=params['criterion'], max_features=params['max_features'],
                                      min_samples_leaf=params['min_samples_leaf'], min_samples_split=params['min_samples_split'], n_estimators=params['n_estimators'])
clf_dna_meth.fit(X_train, y_train)
# %%
# Make predictions
y_pred = clf_dna_meth.predict(X_test)

# Check the F1 score of the model
print("F1 Score", f1_score(y_test, y_pred))
# %%
cm_dna_meth = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_dna_meth)
disp.plot()
plt.show()
# %%
# Now repeat the process for gene expression dataset

X = filtered_gene_expr.drop('Label', axis=1)
y = filtered_gene_expr['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# %%
param_space = dict(min_samples_split=range(2, 10), n_estimators=range(75, 150), min_samples_leaf=range(
    1, 10), max_features=["sqrt", "log2", None], criterion=["gini", "entropy"])


@scheduler.serial
def objective(**params):
    clf = RandomForestClassifier(**params)
    score = cross_val_score(clf, X_train, y_train, cv=10).mean()
    return score


tuner = Tuner(param_space, objective, {'num_iteration': 30})
results = tuner.maximize()
print('best parameters:', results['best_params'])
print('best accuracy:', results['best_objective'])
# %%
params = results['best_params']
params = {'n_estimators': 104, 'min_samples_split': 9,
          'min_samples_leaf': 9, 'max_features': 'log2', 'criterion': 'entropy'}

clf_gene_expr = RandomForestClassifier(criterion=params['criterion'], max_features=params['max_features'],
                                       min_samples_leaf=params['min_samples_leaf'], min_samples_split=params['min_samples_split'], n_estimators=params['n_estimators'])
clf_gene_expr.fit(X_train, y_train)
# %%
# Make predictions
y_pred = clf_gene_expr.predict(X_test)

# Check the accuracy of the model
print("F1 score:", f1_score(y_test, y_pred))
# %%
cm_gene_expr = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_gene_expr)
disp.plot()
plt.show()
# %%

mystery_dna_meth = pd.read_csv('mystery_dna-meth.csv', sep=',')
mystery_gene_expr = pd.read_csv('mystery_gene-expr.csv', sep=',')
# %%
mystery_dna_meth[mystery_dna_meth.columns[2:]] = np.log2(
    mystery_dna_meth[mystery_dna_meth.columns[2:]] + 0.001)
mystery_gene_expr[mystery_gene_expr.columns[2:]] = np.log2(
    mystery_gene_expr[mystery_gene_expr.columns[2:]] + 0.001)
# %%
# Define a mapping dictionary
label_mapping = {'Solid Tissue Normal': 0, 'Primary Tumor': 1}

# Use the dictionary to replace the labels in dna_meth
mystery_dna_meth['Label'] = mystery_dna_meth['Label'].replace(label_mapping)

# Use the dictionary to replace the labels in gene_expr
mystery_gene_expr['Label'] = mystery_gene_expr['Label'].replace(label_mapping)
# %%
y_true = mystery_dna_meth['Label']
filtered_mystery_dna_meth = mystery_dna_meth[combined_cg_values]

y_pred = clf_dna_meth.predict(filtered_mystery_dna_meth)


# %%
print("F1 Score", f1_score(y_true, y_pred))

# %%
cm_dna_meth_mystery = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_dna_meth_mystery)
disp.plot()
plt.show()
# %%
y_true = mystery_gene_expr['Label']
filtered_mystery_gene_expr = mystery_gene_expr[combined_gene_values]
y_pred = clf_gene_expr.predict(filtered_mystery_gene_expr)
# %%
print("F1 Score", f1_score(y_true, y_pred))
# %%
cm_gene_expr_mystery = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_gene_expr_mystery)
disp.plot()
plt.show()

# %%
