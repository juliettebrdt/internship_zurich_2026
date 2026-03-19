from pymongo import MongoClient
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast                          # ← manquait pour convertir les listes du CSV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
from sklearn.utils.class_weight import compute_sample_weight



# ============================================================
# PART 1 — LOAD DATASET
# ============================================================

# use the dataset already existing
df = pd.read_csv("CNV_Label.csv")
print(f"Dataset chargé : {df.shape}")

# Les colonnes cnv_* sont sauvegardées comme strings dans le CSV
# il faut les reconvertir en listes avec ast.literal_eval
for col in ['cnv_dup', 'cnv_del', 'cnv_hldup', 'cnv_hldel']:
    df[col] = df[col].apply(ast.literal_eval)


# ============================================================
# PART 2 — FEATURE PREPROCESSING
# ============================================================

# Class is needed because mutual_info_classif is not directly usable inside sklearn pipeline
# We implement a custom transformer

class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """Sélectionne les features avec MI > mean(MI)"""
    
    def fit(self, X, y):
        mi = mutual_info_classif(X, y, random_state=42)
        self.threshold_ = np.mean(mi)
        self.selected_ = np.where(mi > self.threshold_)[0]
        return self

    def transform(self, X):
        return X[:, self.selected_]


# ============================================================
# PART 3 — BUILD FEATURE MATRIX
# ============================================================

X = np.array([
    dup + del_ + hldup + hldel + [
        sum(dup), sum(del_), sum(hldup), sum(hldel)
    ]
    for dup, del_, hldup, hldel in zip(
        df["cnv_dup"], df["cnv_del"], df["cnv_hldup"], df["cnv_hldel"]
    )
])

y = df["topo_label"]


# Keep top 10 classes
top10 = y.value_counts().head(10).index
mask = y.isin(top10)

X_filtered = X[mask]
y_filtered = y[mask]

le = LabelEncoder()
y_encoded = le.fit_transform(y_filtered)

print(f"Nombre d'analyses après filtrage : {X_filtered.shape[0]}")
print(y_filtered.value_counts())




# ============================================================
# PART 4 — TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First split → hold-out test set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_filtered,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Second split → validation set for early stopping
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,   # 0.25 * 0.8 = 0.2 of full dataset
    random_state=42,
    stratify=y_train_full
)

'''mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
threshold = np.mean(mi_scores)
selected_features = np.where(mi_scores > threshold)[0]

X_train_mi = X_train[:, selected_features]
X_val_mi = X_val[:, selected_features]
X_test_mi = X_test[:, selected_features]'''

sample_weights_train = compute_sample_weight("balanced", y_train)
sample_weights_train_full = compute_sample_weight("balanced", y_train_full)   

print(f"Train size : {X_train.shape}")
print(f"Validation size : {X_val.shape}")
print(f"Test size : {X_test.shape}")


# ============================================================
# PART 5 — PIPELINE + GRID SEARCH
# ============================================================

pipe = Pipeline([
    ('mi_selection', MutualInfoSelector()),
    ('xgb', XGBClassifier(
        n_estimators=200,          # ← initial value (will be tuned later)
        objective='multi:softmax',
        num_class=len(np.unique(y_encoded)),
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    ))
])

param_grid = {
    'xgb__max_depth': [4, 5, 6],
    'xgb__learning_rate': [0.05, 0.1]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search_pipe = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Grid search uses only the training set
grid_search_pipe.fit(X_train, y_train, xgb__sample_weight=sample_weights_train ) 

best_max_depth = grid_search_pipe.best_params_['xgb__max_depth']
best_learning_rate = grid_search_pipe.best_params_['xgb__learning_rate']

print(f"Best parameters : {grid_search_pipe.best_params_}")


# ============================================================
# PART 6 — EARLY STOPPING (FIND BEST N_ESTIMATORS)
# ============================================================

# Apply feature selection
pipe_selector = Pipeline([('mi_selection', MutualInfoSelector())])

pipe_selector.fit(X_train, y_train)

X_train_sel = pipe_selector.transform(X_train)
X_val_sel = pipe_selector.transform(X_val)
X_test_sel = pipe_selector.transform(X_test)

xgb_es = XGBClassifier(
    n_estimators=1000,
    max_depth=best_max_depth,
    learning_rate=best_learning_rate,
    objective='multi:softmax',
    num_class=len(np.unique(y_encoded)),
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss',
    early_stopping_rounds=20
)

# Early stopping must use the validation set (not the test set)
xgb_es.fit(
    X_train_sel,
    y_train,
    sample_weight=sample_weights_train,
    eval_set=[(X_val_sel, y_val)],
    verbose=False
)

best_n_estimators = xgb_es.best_iteration

print(f"Best n_estimators (early stopping) : {best_n_estimators}")

val_pred = xgb_es.predict(X_val_sel)
print(f"Validation accuracy : {metrics.accuracy_score(y_val, val_pred):.3f}")


# ============================================================
# PART 7 — FINAL CROSS VALIDATION
# ============================================================

# Cross-validation performed on training+validation set
# Test set remains untouched

cv_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_cv = Pipeline([
    ('mi_selection', MutualInfoSelector()),
    ('xgb', XGBClassifier(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        learning_rate=best_learning_rate,
        objective='multi:softmax',
        num_class=len(np.unique(y_encoded)),
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    ))
])

scores_pipe = cross_val_score(
    pipe_cv,
    X_train_full,
    y_train_full,
    cv=cv_final,
    scoring='accuracy',
    n_jobs=-1
)

print("\nCross-validation finale :")
print(f"Accuracy par fold : {[round(s, 3) for s in scores_pipe]}")
print(f"Moyenne : {scores_pipe.mean():.3f}")
print(f"Ecart-type : {scores_pipe.std():.3f}")


# ============================================================
# PART 8 — TRAIN FINAL MODEL ON FULL DATASET
# ============================================================

# train on X_filtered to test on new external data

pipe_final = Pipeline([
    ('mi_selection', MutualInfoSelector()),
    ('xgb', XGBClassifier(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        learning_rate=best_learning_rate,
        objective='multi:softmax',
        num_class=len(np.unique(y_encoded)),
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    ))
])

pipe_final.fit(X_train_full, y_train_full, xgb__sample_weight=sample_weights_train_full)

print("\nModèle final train on all the dataset")

joblib.dump(pipe_final, 'pipe_final.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(top10, 'top10_classes.pkl')


# ============================================================
# PART 9 — VALIDATION ON NEW DATA (MongoDB)
# ============================================================

client = MongoClient("mongodb://localhost:27017/")
db = client["progenetix"]

# skip the first dataset used previously
new_analyse = list(db.analyses.find(
    {"cnv_statusmaps": {"$exists": True}},
    {"id": 1, "biosample_id": 1, "cnv_statusmaps": 1}
).skip(20000).limit(20000))


# ============================================================
# BUILD NEW DATASET
# ============================================================

new_id_biosamples = [doc.get('biosample_id') for doc in new_analyse]

new_biosample_label = {}

for biosample in db.biosamples.find(
    {"id": {"$in": new_id_biosamples}},
    {"id": 1, "icdo_topography": 1}
):

    topo = biosample.get('icdo_topography', {})

    id_topo = ""
    label_topo = ""

    if isinstance(topo, dict):
        id_topo = topo.get('id', '')
        label_topo = topo.get('label', '')

    new_biosample_label[biosample.get("id", "")] = {
        "topo_id": id_topo,
        "topo_label": label_topo
    }


new_rows = []

for cnv in new_analyse:

    id_biosamp = cnv.get('biosample_id', '')
    statusmap = cnv.get('cnv_statusmaps', {})

    labels = new_biosample_label.get(id_biosamp, {})

    new_rows.append({
        "topo_label": labels.get('topo_label', ''),
        "cnv_dup": statusmap.get('dup', []),
        "cnv_del": statusmap.get('del', []),
        "cnv_hldup": statusmap.get('hldup', []),
        "cnv_hldel": statusmap.get('hldel', [])
    })


df_new = pd.DataFrame(new_rows)

mask_new = df_new["topo_label"].isin(top10)
df_new = df_new[mask_new]

print(f"New data top10 : {df_new.shape[0]}")
print(df_new["topo_label"].value_counts())


# ============================================================
# FEATURE BUILDING FOR NEW DATA
# ============================================================
X_new = np.array([
    dup + del_ + hldup + hldel + [
        sum(dup), sum(del_), sum(hldup), sum(hldel)
    ]
    for dup, del_, hldup, hldel in zip(
        df_new["cnv_dup"], df_new["cnv_del"], df_new["cnv_hldup"], df_new["cnv_hldel"]
    )
])
# le.transform() same encoding as training
y_new = le.transform(df_new["topo_label"])

y_pred_new = pipe_final.predict(X_new)


# ============================================================
# DIAGNOSTIC
# ============================================================

print(f"Brut récupéré : {len(new_analyse)}")
print(f"Après filtrage top10 : {df_new.shape[0]}")

print("\nDistribution originale (top10) :")
print(y_filtered.value_counts(normalize=True).round(3))

print("\nDistribution nouvelles données :")
print(df_new["topo_label"].value_counts(normalize=True).round(3))


# Test performance on hold-out test set
preds_known = pipe_final.predict(X_test)

print(f"\nAccuracy sur X_test connu : {metrics.accuracy_score(y_test, preds_known):.3f}")


print(f"\nAccuracy sur nouvelles données : {metrics.accuracy_score(y_new, y_pred_new):.3f}")

print("\nRapport détaillé nouvelles données :")

print(metrics.classification_report(
    y_new,
    y_pred_new,
    target_names=le.classes_
)) 