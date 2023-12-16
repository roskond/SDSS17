# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# %%
import os #access operating system
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, LabelBinarizer
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# %%
filepath = 'SDSS17.csv'

# %%
df = pd.read_csv(filepath)
df.head()

# %% [markdown]
#
#     obj_ID = Object Identifier, the unique value that identifies the object in the image catalog used by the CAS
#     alpha = Right Ascension angle (at J2000 epoch)
#     delta = Declination angle (at J2000 epoch)
#     u = Ultraviolet filter in the photometric system
#     g = Green filter in the photometric system
#     r = Red filter in the photometric system
#     i = Near Infrared filter in the photometric system
#     z = Infrared filter in the photometric system
#     run_ID = Run Number used to identify the specific scan
#     rereun_ID = Rerun Number to specify how the image was processed
#     cam_col = Camera column to identify the scanline within the run
#     field_ID = Field number to identify each field
#     spec_obj_ID = Unique ID used for optical spectroscopic objects (this means that 2 different observations with the same spec_obj_ID must share the output class)
#     class = object class (galaxy, star or quasar object)
#     redshift = redshift value based on the increase in wavelength
#     plate = plate ID, identifies each plate in SDSS
#     MJD = Modified Julian Date, used to indicate when a given piece of SDSS data was taken
#     fiber_ID = fiber ID that identifies the fiber that pointed the light at the focal plane in each observation
#

# %%
df.info()

# %%
for i in df:
    print(df.isnull().sum())

# %%
df['class'].nunique()

# %%
dfs = df.copy()

# %%
ds = df.copy()

# %%
f,ax = plt.subplots(figsize=(12,8))
sns.heatmap(ds.corr(), cmap="crest", annot=True, linewidths=0.5, fmt= '.2f',ax=ax)
plt.show()

plt.savefig("heatmap_sloan.png")

# %%
dfs = dfs.drop(['obj_ID','alpha','delta','run_ID','rerun_ID','cam_col','field_ID','fiber_ID','plate','spec_obj_ID'], axis = 1)

# %%
dfs.head(20)

# %%
#dfs = dfs.drop(['plate','spec_obj_ID'], axis = 1)

# %%
f,ax = plt.subplots(figsize=(12,8))
sns.heatmap(dfs.corr(), cmap=sns.cubehelix_palette(as_cmap=True), annot=True, linewidths=0.5, fmt= '.2f',ax=ax)
plt.show()



# %%
dim_stellar = dfs.copy()

# %%
#dim_stellar = dim_stellar.drop(['class'], axis = 1)

# %%
dim_stellar.head()

# %%
dim_stellar = dim_stellar.rename(columns={'u':'Ultraviolet Filter', 'r':'Red Filter', 'g': 'Green Filter', 
                                  'i': 'Near Infrared Filter', 'z':'Infrared FIlter', 'class':'Class'})

# %%
dim_stellar.head()

# %%
dim_stellar = dim_stellar.rename(columns={'redshift':'Redshift'})

# %%
dim_stellar.head()

# %%
f,ax = plt.subplots(figsize=(12,8))
sns.heatmap(dim_stellar.corr(), cmap=sns.cubehelix_palette(as_cmap=True), annot=True, linewidths=0.5, fmt= '.2f',ax=ax)
plt.show()



# %%
Pair_plot1 = sns.pairplot(dim_stellar, hue = 'Class')

# %%
plt.savefig("Pairplot3")

# %%
pip install imbalanced-learn

# %% [markdown]
# # Data Processing

# %%
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# %%
le = LabelEncoder()

# %%
dim_stellar['Class'] = le.fit_transform(dim_stellar['Class'])


# %%
dim_stellar['Class'].value_counts()

# %%
#x = dim_stellar(['class'], axis = 1)
#y = dim_stellar.loc[:,'class'].values

# %%
d_stellar = dim_stellar.copy()

# %%
X1 = d_stellar.drop('Class', axis = 1, inplace = False)
Y = d_stellar['Class']

# %%
from collections import Counter

# %%
sm = SMOTE(random_state=42)
print('Original dataset size %s' % Counter(Y))
X1, Y = sm.fit_resample(X1, Y)
print('Resampled dataset size %s' % Counter(Y))

# %%
scaler = RobustScaler()
scaler.fit(X1)
X1 = scaler.transform(X1)

# %%
print(X1.shape)

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y, test_size=0.25, random_state = None) 

# %% [markdown]
# # Random Forest Baseline

# %%
from sklearn.model_selection import GridSearchCV

# %%
rfc = RandomForestClassifier()
param = {'n_estimators' : [100, 200, 300], 'criterion' : ['gini', 'entropy','log_loss']}

gsearch = GridSearchCV(rfc, param_grid = param, cv = 5, scoring = 'f1_macro', n_jobs = 4)
gsearch.fit(X_train, Y_train)
best_model = gsearch.best_estimator_
print(best_model)

# %%
best_params = gsearch.best_params_
print(best_params)

# %%
rfc = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
rfc.fit(X_train, Y_train)

# %%
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score 

# %%
print("Cross-Validation Score:", cross_val_score(rfc, X_train, Y_train, cv=None, scoring=None))

# %%
from sklearn.metrics import precision_recall_fscore_support


# %%
def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['Accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp)
    results_pos['Precision'] = precision
    results_pos['Recall'] = recall
    results_pos['F1 Score'] = f_beta
    return results_pos


# %%
y_rfc_pred = rfc.predict(X_test)
evaluate_metrics(Y_test, y_rfc_pred)

# %%
target_names = ["0","1","2"]
y_cross_val_pred_rfc = cross_val_predict(rfc,X_train,Y_train, cv=None)
print(classification_report(Y_train, y_cross_val_pred_rfc, target_names=target_names))

# %%
from sklearn.metrics import plot_confusion_matrix

# %%
rfc_matrix = plot_confusion_matrix(rfc,X_test,Y_test,labels=[0,1,2], cmap="vlag")
plt.show()
plt.savefig("rfc_matrix.png")

# %% [markdown]
# # PCA Analysis

# %%
from sklearn.decomposition import PCA

pca = PCA()
pcomp = pca.fit_transform(X1)
pcomp = pd.DataFrame(pcomp)
print(pcomp.shape)
pcomp.head()

# %%
exp_var_pca = pca.explained_variance_ratio_

# %%
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

# %%
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual Explained Variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative Explained Variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# %%
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance")

# %%
stellar_cov=np.cov(X1.T)

# %%
print('Covariance matrix: \n%s' %stellar_cov)

# %%
plt.figure(figsize=(12,12))
#sns.heatmap(stellar_cov, vmax=1, square=True,annot=True,cmap='cubehelix')
sns.heatmap(stellar_cov, square=True,annot=True)
sns.color_palette("Paired", as_cmap=True)
plt.show()
plt.savefig("Covariance_matrix_heatmap.png")

# %%
e_vec, e_val = np.linalg.eig(stellar_cov)

print('Eigenvectors \n%s' %e_vec)
print('Eigenvalues \n%s' %e_val)


# %%
#setting components to 1

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X1)

X_pca = pd.DataFrame(X_pca)
print(X_pca.head())
print("Shape:", X_pca.shape)

# %%
#Making train_test split using component

X_train_pca, X_test_pca, Y_train, Y_test = train_test_split(X_pca, Y, 
                                                            test_size=0.25, shuffle=True, 
                                                            random_state=None)

rfc_pca = RandomForestClassifier(n_estimators=100, criterion='entropy')

rfc_pca.fit(X_train_pca, Y_train)

Y_rfc_pca_pred = rfc_pca.predict(X_test_pca)
evaluate_metrics(Y_test, Y_rfc_pca_pred)

# %%
target_names = ["0","1","2"]
Y_cross_val_pred_rfc_pca = cross_val_predict(rfc,X_train_pca,Y_train, cv=None)
print(classification_report(Y_train, Y_cross_val_pred_rfc_pca, target_names=target_names))

# %%
rfc_matrix = plot_confusion_matrix(rfc_pca,X_test_pca,Y_test,labels=[0,1,2], cmap="vlag")
plt.show()
plt.savefig("rfc_matrix_pca.png")

# %% [markdown]
# # Factor Analysis

# %%
from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis()
X_fa = fa.fit_transform(X1)

# %%
#print("Eigenvalues:", fa.get_eigenvalues()[0])
#print("Communalities:", fa.get_communalities)
#print("Spceific Variance:", fa.get_uniqueness)
#print("Factor Loadings:", fa.loadings_)

# %%
X_train_fa, X_test_fa, Y_train, Y_test = train_test_split(X_fa, Y, 
                                                            test_size=0.25, shuffle=True, 
                                                            random_state=None)
rfc_fa = RandomForestClassifier(n_estimators=100, criterion='entropy')

rfc_fa.fit(X_train_fa, Y_train)

Y_rfc_fa_pred = rfc_fa.predict(X_test_fa)
evaluate_metrics(Y_test, Y_rfc_fa_pred)

# %%
target_names = ["0","1","2"]
Y_cross_val_pred_rfc_fa = cross_val_predict(rfc_fa,X_train_fa,Y_train, cv=None)
print(classification_report(Y_train, Y_cross_val_pred_rfc_fa, target_names=target_names))

# %%
rfc_matrix = plot_confusion_matrix(rfc_fa,X_test_fa,Y_test,labels=[0,1,2], cmap="vlag")
plt.show()
plt.savefig("rfc_matrix_pca.png")

# %% [markdown]
# # TruncatedSVD

# %%
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD()
X_svd = svd.fit_transform(X1)

# %%
X_train_svd, X_test_svd, Y_train, Y_test = train_test_split(X_svd, Y, 
                                                            test_size=0.25, shuffle=True, 
                                                            random_state=None)
rfc_svd = RandomForestClassifier(n_estimators=100, criterion='entropy')

rfc_svd.fit(X_train_svd, Y_train)

Y_rfc_svd_pred = rfc_fa.predict(X_test_fa)
evaluate_metrics(Y_test, Y_rfc_svd_pred)

# %%
target_names = ["0","1","2"]
Y_cross_val_pred_rfc_svd = cross_val_predict(rfc_svd,X_train_svd,Y_train, cv=None)
print(classification_report(Y_train, Y_cross_val_pred_rfc_svd, target_names=target_names))

# %%
rfc_matrix = plot_confusion_matrix(rfc_svd,X_test_svd,Y_test,labels=[0,1,2], cmap="vlag")
plt.show()
plt.savefig("rfc_matrix_svd.png")

# %% [markdown]
# # Neural Network Classification

# %% [markdown]
#     Does the report include a section describing the data?
#
#     Does the report include a paragraph detailing the main objective(s) of this analysis?  
#
#     Does the report include a section with variations of a Deep Learning model and specifies which one is the model that best suits the main objective(s) of this analysis?
#
#     Does the report include a clear and well presented section with key findings related to the main objective(s) of the analysis?
#
#     Does the report highlight possible flaws in the model and a plan of action to revisit this analysis with additional data or different modeling techniques? 

# %%
from tensorflow.keras.layers import Dropout

# %%
#Network architecture

model = Sequential()
model.add(Dense(120, input_shape = (7,), activation = 'sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(3, activation = 'softmax'))
