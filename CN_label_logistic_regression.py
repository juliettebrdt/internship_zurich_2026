import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

df = pd.read_csv("CNV_Label.csv", sep="\s+", header=None)

df.columns = [
    "sample_id","biosample_id","topo_code","topo_name",
    "morpho_code","morpho_name",
    "cnv_dup","cnv_del","cnv_hldup","cnv_hldel"
]

df["cnv_dup"] = df["cnv_dup"].apply(ast.literal_eval)
df["cnv_del"] = df["cnv_del"].apply(ast.literal_eval)

dup = pd.DataFrame(df["cnv_dup"].tolist())
dele = pd.DataFrame(df["cnv_del"].tolist())

X = pd.concat([dup, dele], axis=1)

le = LabelEncoder()
y = le.fit_transform(df["morpho_name"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))