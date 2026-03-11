from pymongo import MongoClient
import matplotlib.pyplot as plt
import pandas as pd
import csv
from collections import defaultdict
import numpy as np

client = MongoClient("mongodb://localhost:27017/")
db = client["progenetix"]

#rretrieve the analyses which have the statusmap and limited to 2000 for the test
analyse = list(db.analyses.find({"cnv_statusmaps": {"$exists": True}}, {"id": 1, "biosample_id": 1, "cnv_statusmaps": 1}).limit(20000))
#creation of a list of all the biosample_id found in analyses
id_biosamples = [doc.get('biosample_id') for doc in analyse]

biosample_label = {}
#looking for the biosample which are in the id list
for biosample in db.biosamples.find({"id": {"$in": id_biosamples}},{"id":1, "icdo_topography": 1, "icdo_morphology": 1}) :
    #initialisation 
    id_morpho, label_morpho = "", ""
    id_topo, label_topo = "", ""
    morpho = biosample.get("icdo_morphology",{})
    if isinstance(morpho,dict): # to check if it is a dictionnaire
        id_morpho = morpho.get('id','')
        label_morpho = morpho.get('label','')
    topo = biosample.get('icdo_topography',{})
    if isinstance (topo,dict):
        id_topo = topo.get('id','')
        label_topo = topo.get('label','')  
    #creation of a dict
    ## Each biosample (key = id) is associated with a dictionary containing its topographic and morphological information
    biosample_label[biosample.get("id", "")] = {
    "topo_id":    id_topo,
    "topo_label": label_topo,
    "morpho_id":  id_morpho,
    "morpho_label": label_morpho,
}
rows=[]
for cnv in analyse :
    #retrieve all information that we need in the differents analyses
    id_analyse = cnv.get('id','')
    id_biosamp = cnv.get('biosample_id','')
    statusmap = cnv.get('cnv_statusmaps','')
    cnv_dup = statusmap.get('dup','')
    cnv_del = statusmap.get('del','')
    cnv_hldup = statusmap.get ('hldup','')
    cnv_hldel = statusmap.get ('hldel','')
    labels = biosample_label.get(id_biosamp, {})
    #add to dict
    rows.append({
        "analysis_id":  id_analyse,
        "biosample_id": id_biosamp,
        "topo_id":     labels.get("topo_id",""),
        "topo_label":   labels.get('topo_label',''),
        "morpho_id":    labels.get('morpho_id',''),
        "morpho_label": labels.get('morpho_label',''),
        "cnv_dup":   cnv_dup,
        "cnv_del": cnv_del,
        "cnv_hldup": cnv_hldup,
        "cnv_hldel": cnv_hldel,
    })
df = pd.DataFrame(rows)

df.to_csv("CNV_Label.csv", index=False)

print(df.shape)   # affiche le nombre de lignes et colonnes
print(df.head())  # affiche les 5 premières lignes

#logistic regression
from sklearn.linear_model import LogisticRegression

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif,mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, GridSearchCV
from xgboost import XGBClassifier

# Combine CNV duplication and deletion features into a single numeric matrix
X = np.array([
    dup + del_ + hldup + hldel
    for dup, del_, hldup, hldel
    in zip(df["cnv_dup"], df["cnv_del"], df["cnv_hldup"], df["cnv_hldel"])
])
y = df["topo_label"]
# Keep only the top 10 most represented cancer types
top10 = y.value_counts().head(10).index
mask = y.isin(top10)
X_filtered = X[mask]
y_filtered = y[mask]

# Encode labels to numeric form for plotting
le = LabelEncoder()
y_encoded = le.fit_transform(y_filtered)  

print(f"Nombre d'analyses après filtrage : {X_filtered.shape[0]}")
print(y_filtered.value_counts())
# undersampling to have the same amount of data in each label
'''dfs = []
for classe in np.unique(y_encoded):
    # indices de cette classe
    indices = np.where(y_encoded == classe)[0]
    # prendre 276 au hasard
    indices_sample = np.random.RandomState(42).choice(indices, size=276, replace=False)
    dfs.append(indices_sample)

# combiner tous les indices
indices_balanced = np.concatenate(dfs)

X_balanced = X_filtered[indices_balanced]
y_balanced = y_encoded[indices_balanced]

print(f"Taille après undersampling : {X_balanced.shape}")
print(pd.Series(y_balanced).value_counts())'''

#split into dataset train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered,
    y_encoded,
    test_size=0.4,
    random_state=42,
    stratify=y_encoded
)
print(f"X_train shape : {X_train.shape}")

#=====With the original date (lots of sparses data)====
'''model = LogisticRegression(max_iter = 10000, class_weight="balanced")
#Train model

model.fit(X_train, y_train)
#Prediction on test set
y_pred = model.predict(X_test)

print("Accuracy Logisitic Regression before PCA:", metrics.accuracy_score(y_test, y_pred))'''

#=====Trie to handle the sparses data with PCA)
'''
#check how many components explain 90% of the variance
variance_cumulee = PCA().fit(X_train).explained_variance_ratio_.cumsum() # addition the ercentage of variance explained by each of the selected component 
n_composantes = (variance_cumulee < 0.90).sum() + 1 #count the number of True <90%
print(f"Composantes nécessaires pour 90% : {n_composantes}")

fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
#for the visualisation
X_reduced = PCA(n_components=3).fit_transform(X_filtered)  # ← remplace iris.data

scatter = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=y_encoded,       # ← remplace iris.target
    s=40,
)

ax.set(
    title="First three principal components - CNV data",
    xlabel="1st Principal Component",
    ylabel="2nd Principal Component",
    zlabel="3rd Principal Component",
)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

# remplace iris.target_names par tes labels
legend1 = fig.legend(
    scatter.legend_elements()[0],
    le.classes_.tolist(),
    loc="center left",        # ← position sur la figure
    bbox_to_anchor=(0.70, 0.40),  # ← à droite de la figure
    title="Cancer type",
    fontsize=7                # ← réduire la taille si trop grand
)


plt.show()
pca = PCA(n_components=n_composantes)  # ← utiliser n_composantes !
X_train_pca = pca.fit_transform(X_train)  # ← fit sur train seulement
X_test_pca  = pca.transform(X_test)       # ← transform sur test
#====Logistic Regression on PCA-reduced data
model_PCA = LogisticRegression(max_iter = 10000)
#Train model

model_PCA.fit(X_train_pca, y_train)
#Prediction on test set
y_pred_pca = model_PCA.predict(X_test_pca)

print("Accuracy Logisitic Regression before PCA:", metrics.accuracy_score(y_test, y_pred_pca))
'''
#======TRUNCATEDSVD=======
'''#conversion to a shape of sparses matrix (stock only no nuls values and their postions)
from scipy.sparse import csr_matrix
X_train_sparse = csr_matrix(X_train)
X_test_sparse  = csr_matrix(X_test) # shape (n_samples, n_bins)
#application of truncatedSVD

svd_exploration = TruncatedSVD(n_components=100, random_state=42)
svd_exploration.fit(X_train_sparse)

variance_cumulee = svd_exploration.explained_variance_ratio_.cumsum()
n_composantes = int((variance_cumulee < 0.90).sum() + 1)
print(f"\nComposantes nécessaires pour expliquer 90% de la variance : {n_composantes}")

# Visualiser la variance expliquée cumulée
plt.figure(figsize=(8, 4))
plt.plot(variance_cumulee)
plt.axhline(y=0.90, color='r', linestyle='--', label='90% variance')
plt.axvline(x=n_composantes, color='g', linestyle='--', label=f'{n_composantes} composantes')
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance expliquée cumulée")
plt.title("TruncatedSVD - Variance expliquée")
plt.legend()
plt.tight_layout()
plt.show()

# Appliquer SVD avec le bon nombre de composantes
svd = TruncatedSVD(n_components=n_composantes, random_state=42)
X_train_svd = svd.fit_transform(X_train_sparse)  # fit + transform sur train
X_test_svd  = svd.transform(X_test_sparse)        # transform seulement sur test

# StandardScaler APRÈS SVD (pas avant, sinon ça détruit la sparsité)
scaler = StandardScaler()
X_train_final = scaler.fit_transform(X_train_svd)  # fit + transform sur train
X_test_final  = scaler.transform(X_test_svd)        # transform seulement sur test
198
print(f"\nShape X_train avant SVD : {X_train.shape}")
print(f"Shape X_train après SVD  : {X_train_final.shape}")'''

#=====SELECTION FEATURES=====
#====SELECTFROMMODEL WITH DECISION TREE
''' Decision tree calcul the importance of each feature
    SelectFromModel keep the feature that are up a threshold '''
'''from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(X_train,y_train)

#threshold='mean' : keep features which importance > mean
selector_dt = SelectFromModel(dt, prefit=True, threshold='mean')
X_train_dt=selector_dt.transform(X_train)
X_test_dt = selector_dt.transform(X_test)
print(f"Avant : {X_train.shape[1]} features")
print(f"Après SelectFromModel (mean): {X_train_dt.shape[1]} feature")
for threshold in ['mean', 'median', '1.25*mean']:
    sel = SelectFromModel(dt, prefit=True, threshold=threshold)
    X_tr = sel.transform(X_train)
    X_te = sel.transform(X_test)
    bst = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                        objective='multi:softmax',
                        num_class=len(np.unique(y_encoded)),
                        random_state=42, n_jobs=-1, eval_metric='mlogloss')
    bst.fit(X_tr, y_train)
    acc = metrics.accuracy_score(y_test, bst.predict(X_te))
    print(f"  threshold={threshold} → {X_tr.shape[1]} features → Accuracy XGBoost : {acc:.3f}")
#=====LOGISTRIC REGRESSION SVD====
model_SVD = LogisticRegression(max_iter = 10000)
#Train model

model_SVD.fit(X_train_final, y_train)
#Prediction on test set
y_pred_svd = model_SVD.predict(X_test_final)

print("Accuracy Logisitic Regression with SVD:", metrics.accuracy_score(y_test, y_pred_svd)) '''
#=====SelectKbest====
best_acc = 0
best_k=0
'''for k in [100, 300, 500, 1000, 2000]:
    sel = SelectKBest(mutual_info_classif, k=k) #calcul how many feature help to predict the class 
    X_tr = sel.fit_transform(X_train, y_train) # select the k the best feature and tranform X_train
    X_te = sel.transform(X_test) #applique la meme selection au test
    bst = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='multi:softmax',
                        num_class=len(np.unique(y_encoded)),
                        random_state=42, n_jobs=-1, eval_metric='mlogloss')
    bst.fit(X_tr, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_train)
    acc = metrics.accuracy_score(y_test, bst.predict(X_te))
    acc = metrics.accuracy_score(y_test, bst.predict(X_te))

    print(f"  K={k} → Accuracy XGBoost : {acc:.3f}")
    if acc >best_acc :
        best_acc =acc
        best_k = k

# Garder le meilleur K (à ajuster selon les résultats ci-dessus)

selector_kbest = SelectKBest(mutual_info_classif, k=best_k)
X_train_kbest = selector_kbest.fit_transform(X_train, y_train)
X_test_kbest  = selector_kbest.transform(X_test)
print(f"\nSelectKBest retenu : K={best_k} → {X_train_kbest.shape[1]} features")
'''

mi_scores = mutual_info_classif(X_train, y_train)
threshold = np.mean(mi_scores)
selected_features = np.where(mi_scores > threshold)[0]
X_train_mi = X_train[:, selected_features]
X_test_mi  = X_test[:, selected_features]
#====SelectKbest and SVD =====
'''selector_kb = SelectKBest(mutual_info_classif, k=50)
X_train_svd_kb = selector_kb.fit_transform(X_train_final, y_train)
X_test_svd_kb  = selector_kb.transform(X_test_final)
print(f"Après SVD + SelectKBest : {X_train_svd_kb.shape}") '''
#===== RANDOM FOREST ====
from sklearn.ensemble import RandomForestClassifier
#=====RF WITH PCA====
'''model_forest_pca = RandomForestClassifier(n_estimators=100, random_state=42,class_weight = 'balanced')
model_forest_pca.fit(X_train_pca ,y_train)
y_pred_pca_forest = model_forest_pca.predict(X_test_pca)
print("Accuracy Random Forest after PCA:", metrics.accuracy_score(y_test, y_pred_pca_forest))
'''
#===== RF WITHOUT PCA =====
'''model_forest= RandomForestClassifier(n_estimators=100, random_state=42, class_weight = 'balanced')
model_forest.fit(X_train,y_train)
y_pred_forest =model_forest.predict(X_test)
print("Accuracy Random Forest before PCA:", metrics.accuracy_score(y_test, y_pred_forest))'''
'''print(f"Dimensions X_train_pca : {X_train_pca.shape}")
print(f"Dimensions X_test_pca : {X_test_pca.shape}")
print(f"Avant PCA : {X_train.shape}")
print(f"Après PCA : {X_train_pca.shape}")
print(f"Variance expliquée : {pca.explained_variance_ratio_.sum():.1%}")'''

#===RF with SVD ======
'''model_forest_svd = RandomForestClassifier (n_estimators=100, random_state=42, class_weight = 'balanced', n_jobs=-1)
model_forest_svd.fit(X_train_final, y_train)
y_pred_forest_svd = model_forest_svd.predict(X_test_final)
print("Accuracy Random Forest with SVD:", metrics.accuracy_score(y_test, y_pred_forest_svd))'''
#====RF with DecisionTree ====
'''model_forest_dt = RandomForestClassifier (n_estimators=100, random_state=42, class_weight = 'balanced', n_jobs=-1)
model_forest_dt.fit(X_train_dt, y_train)
y_pred_forest_dt = model_forest_dt.predict(X_test_dt)
print("Accuracy Random Forest with DecisionTree:", metrics.accuracy_score(y_test, y_pred_forest_dt))
'''
#====RF with SelectKbest=====
'''model_forest_kbest = RandomForestClassifier (n_estimators=100, random_state=42, class_weight = 'balanced', n_jobs=-1)
model_forest_kbest.fit(X_train_mi, y_train)
y_pred_forest_kbest = model_forest_kbest.predict(X_test_mi)
print("Accuracy Random Forest with SelectKbest:", metrics.accuracy_score(y_test, y_pred_forest_kbest))'''
#=====RF SelectKbest +. SVD ======
'''model_forest_kbest_svd = RandomForestClassifier (n_estimators=100, random_state=42, class_weight = 'balanced', n_jobs=-1)
model_forest_kbest_svd.fit(X_train_svd_kb, y_train)
y_pred_forest_kbest_svd = model_forest_kbest_svd.predict(X_test_svd_kb)
print("Accuracy Random Forest with SelectKbest +SVD:", metrics.accuracy_score(y_test, y_pred_forest_kbest_svd))'''

#===== XGBOOST====

from xgboost import XGBClassifier
#===== XGBOOST WITHOUT PCA and split normal ======
'''bst = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='multi:softmax',num_class=len(np.unique(y_encoded)),random_state=42,n_jobs=-1,eval_metric='mlogloss' )
# fit model
bst.fit(X_train, y_train)
# make predictions
preds = bst.predict(X_test)
print("Accuracy XGBOOST before PCA:", metrics.accuracy_score(y_test, preds))
'''

#======Cross-Validation======
'''cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_bst = cross_val_score(
    XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='multi:softmax',num_class=len(np.unique(y_encoded)),random_state=42,n_jobs=-1,eval_metric='mlogloss' ),
    X_filtered,  # tout le dataset
    y_encoded,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

print("XGBoost - StratifiedKFold (k=5) :")
print(f"  Accuracy par fold : {[round(s, 3) for s in scores]}")
print(f"  Moyenne           : {scores_bst.mean():.3f}")
print(f"  Ecart-type        : {scores_bst.std():.3f}")'''
#====== XGBOOST WITH PCA
'''bst_pca = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
# fit model
bst_pca.fit(X_train, y_train)
# make predictions
preds_pca= bst_pca.predict(X_test)
print("Accuracy XGBOOST after PCA:", metrics.accuracy_score(y_test, preds_pca))
'''
#====XGBOOST with SVD=====
'''bst_svd = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='multi:softmax', num_class=len(np.unique(y_encoded)),random_state=42,n_jobs=-1,eval_metric='mlogloss')
# fit model
bst_svd.fit(X_train_final, y_train)
# make predictions
preds_svd = bst_svd.predict(X_test_final)
print("Accuracy XGBOOST with SVD:", metrics.accuracy_score(y_test, preds_svd)) '''

#=====XGBOOST with DecisioTree=======
'''bst_dt = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='multi:softmax', num_class=len(np.unique(y_encoded)),random_state=42,n_jobs=-1,eval_metric='mlogloss')
# fit model
bst_dt.fit(X_train_dt, y_train)
# make predictions
preds_dt = bst_dt.predict(X_test_dt)
print("Accuracy XGBOOST with DecisionTree:", metrics.accuracy_score(y_test, preds_dt))'''
#=====XGBOOST with SelectKbest=====
'''bst_kbest = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='multi:softmax', num_class=len(np.unique(y_encoded)),random_state=42,n_jobs=-1,eval_metric='mlogloss')
# fit model
bst_kbest.fit(X_train_mi, y_train)
# make predictions
preds_kbest = bst_kbest.predict(X_test_mi)
print("Accuracy XGBOOST with SelectKbest", metrics.accuracy_score(y_test, preds_kbest))
'''
param_grid = {
    'n_estimators'  : [200, 300],
    'max_depth'     : [6, 8],
    'learning_rate' : [0.1, 0.2]
}
# → 2 × 2 × 2 = 8 combinaisons × 5 folds = 40 entraînements

xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y_encoded)), 
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)


grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1   # affiche la progression
)

grid_search.fit(X_train_mi, y_train)
# Tester le meilleur modèle sur le test set
best_model = grid_search.best_estimator_
preds_best = best_model.predict(X_test_mi)
print(f"Accuracy sur test set : {metrics.accuracy_score(y_test, preds_best):.3f}")

print(f"Meilleurs paramètres : {grid_search.best_params_}")
print(f"Meilleure accuracy CV : {grid_search.best_score_:.3f}")
#====Cross-Validation======
'''scores_bst_mi = []

for train_idx, test_idx in cv.split(X_filtered, y_encoded):
    # 1. Séparer train et test pour ce fold
    X_train_fold = X_filtered[train_idx]
    X_test_fold  = X_filtered[test_idx]
    y_train_fold = y_encoded[train_idx]
    y_test_fold  = y_encoded[test_idx]

    # 2. Sélection sur train UNIQUEMENT
    # fit_transform sur train → calcule les scores MI sur train seulement
    # transform sur test → applique la même sélection sans recalculer
    selector = SelectKBest(mutual_info_classif, k=500)
    X_train_sel = selector.fit_transform(X_train_fold, y_train_fold)
    X_test_sel  = selector.transform(X_test_fold)

    # 3. Entraînement sur train
    bst_fold = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='multi:softmax', num_class=len(np.unique(y_encoded)),random_state=42,n_jobs=-1,eval_metric='mlogloss')
    bst_fold.fit(X_train_sel, y_train_fold)

    # 4. Évaluation sur test
    score = metrics.accuracy_score(y_test_fold, bst_fold.predict(X_test_sel))
    scores_bst_mi.append(score)

scores_bst_mi = np.array(scores_bst_mi)'''
#=====SelectKBest + SVD=====
''''bst_kbest_svd = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='multi:softmax', num_class=len(np.unique(y_encoded)),random_state=42,n_jobs=-1,eval_metric='mlogloss')
# fit model
bst_kbest_svd.fit(X_train_svd_kb, y_train)
# make predictions
preds_kbest_svd = bst_kbest_svd.predict(X_test_svd_kb)
print("Accuracy XGBOOST with SelectKbest and svd", metrics.accuracy_score(y_test, y_pred_forest_kbest_svd))'''