# Esse é o codigo utilizado para verificar os dados e fazer testes em busca de
# encontrar a melhor resposta e modelo para predição dos clientes
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# %%

df = pd.read_csv("dados.csv", sep=",", index_col=0)
print(df.shape)
df.head(10)

# %%
# Verificações iniciais de qualidade dos dados
print(f"tipos de dados: {df.dtypes.value_counts()}")
print(f"o número de nulos é: {sum(df.isnull().sum())}")
print(f"O número de duplicados é: {df.duplicated().sum()}")

print(f"o valor minímo obs é: {df.min().min()}")
print(f"o valor máximo obs é: {df.max().max()}")
df.info()

# %%
# Divisão entre treino, calibração e teste
features = df.columns.tolist()[:-1]
resposta = df["class"]

stratify = sum(resposta) / len(resposta)  # 38% na classe 1

x_train, x_rest, y_train, y_rest = train_test_split(
    df[features], resposta, train_size=0.5, random_state=42, stratify=resposta
)
x_calib, x_test, y_calib, y_test = train_test_split(
    x_rest, y_rest, train_size=0.5, random_state=42, stratify=y_rest
)


# %%
# Análise exploratória inicial

# Primeiro eu dei um describe e como são apenas 50 variáveis eu fui olhando uma a uma pra ver se algumas ganhavam destaque

pd.set_option("display.max_columns", None)  # não limitar nº de colunas
pd.set_option("display.width", None)  # não quebrar linha
x_train.describe()  # features diferentes: 8, 17, 31, 50

# %%
print(x_train[["feat_8", "feat_17", "feat_31", "feat_50"]].head(20))
# Conseguimos ver que entre as variáveis que destaquei a 31 assume apenas dois valores
corr = x_train.corr()  # nenhuma variável apresenta correlação linear

sns.heatmap(corr, cmap="coolwarm")
plt.show()

# %%

model = RandomForestClassifier(
    random_state=42, n_estimators=100, min_samples_leaf=10, max_features="log2"
)
model.fit(x_train, y_train)

importances = model.feature_importances_
indices = np.argsort(importances)

myClassifiers = pd.Series(model.feature_importances_, index=x_train.columns)
myClassifiers.sort_values(ascending=False, inplace=True)
top_10 = pd.DataFrame(myClassifiers.head(10))


plt.figure(figsize=(8, 5))
plt.barh(top_10.index, top_10[0])
plt.xlabel("Importância")
plt.title("Importância das Variáveis (Random Forest)")
plt.grid(True, axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Define the features you want to plot
features_to_plot = ["feat_8", "feat_17", "feat_31", "feat_50"]

# Create a figure and a set of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
axes = axes.flatten()  # Flatten the 2x2 array of axes for easier iteration

# Loop through the features and create a scatter plot on each subplot
for i, feature in enumerate(features_to_plot):
    axes[i].scatter(x_train[feature], y_train, alpha=0.6)
    axes[i].set_title(f"y_train vs {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("y_train")
    axes[i].grid(True, linestyle="--", alpha=0.7)

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.show()

# Feat 31 é vazemento de dados com base na análise feita
# As outras 3 variáveis parecem oferecer resultados para discriminar a resposta.

# %%

teste = x_train.drop(["feat_31"], axis=1)
model = RandomForestClassifier(
    random_state=42, n_estimators=100, min_samples_leaf=10, max_features="log2"
)
model.fit(teste, y_train)

importances = model.feature_importances_
indices = np.argsort(importances)

myClassifiers = pd.Series(model.feature_importances_, index=teste.columns)
myClassifiers.sort_values(ascending=False, inplace=True)
top_10 = pd.DataFrame(myClassifiers.head(10))


plt.figure(figsize=(8, 5))
plt.barh(top_10.index, top_10[0])
plt.xlabel("Importância")
plt.title("Importância das Variáveis (Random Forest)")
plt.grid(True, axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

n_feats = len(features_to_plot)
n_cols = 5
n_rows = -(-n_feats // n_cols)  # teto da divisão

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey=True)
axes = axes.flatten()

for i, feature in enumerate(features_to_plot):
    axes[i].scatter(x_train[feature], y_train, alpha=0.6)
    axes[i].set_title(f"y_train vs {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("y_train")

# apaga axes sobrando
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()

# De fato olhando isso apenas a feat_8, 17 e 50 parecem se diferenciar das demais em termos de predição.

# %%

# Vamos ajustar um KNN com as 5 principais, mas acredito que apenas as 3 principais são melhores

vizinhos = [1, 2, 3, 4, 5, 10]
for i in vizinhos:
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(x_train[["feat_8", "feat_17", "feat_50", "feat_20", "feat_13"]], y_train)

    pred_calib = KNN.predict(
        x_calib[["feat_8", "feat_17", "feat_50", "feat_20", "feat_13"]]
    )
    print(classification_report(y_calib, pred_calib), f"neightbors = {i}")


vizinhos = [1, 2, 3, 4, 5, 10]
for i in vizinhos:
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(x_train[["feat_8", "feat_17", "feat_50"]], y_train)

    pred_calib = KNN.predict(x_calib[["feat_8", "feat_17", "feat_50"]])
    print(classification_report(y_calib, pred_calib), f"neightbors = {i}")


# %%

# 4) Preprocessor: discretiza feat_8, feat_17, feat_50 em 5 bins quantílicos
disc_cols = ["feat_8", "feat_17", "feat_50"]
pass_cols = ["feat_20", "feat_13"]
preprocessor = ColumnTransformer(
    [
        (
            "discretizer",
            KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="quantile"),
            disc_cols,
        ),
        ("passthrough", "passthrough", pass_cols),
    ]
)

# 5) Pipeline completo
pipeline = Pipeline(
    [
        ("prep", preprocessor),
        ("clf", LogisticRegression(solver="liblinear", random_state=42)),
    ]
)

# 6) Treino
pipeline.fit(x_train, y_train)

# 7) Avaliação em calibração
y_hat = pipeline.predict(x_calib)
p_hat = pipeline.predict_proba(x_calib)[:, 1]

print(classification_report(y_calib, y_hat))
print("ROC-AUC (calibração):", roc_auc_score(y_calib, p_hat))
# %%
# feat_drop = "feat_31"
feat_drop = ["feat_8", "feat_17", "feat_31", "feat_50"]
x_train = x_train.drop(feat_drop, axis=1)
x_calib = x_calib.drop(feat_drop, axis=1)
x_test = x_test.drop(feat_drop, axis=1)
# 1) Random Forest
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_calib)
p_pred_rf = rf.predict_proba(x_calib)[:, 1]

print("=== Random Forest ===")
print(classification_report(y_calib, y_pred_rf))
print("ROC-AUC RF:", roc_auc_score(y_calib, p_pred_rf), "\n")

# 2) Gradient Boosting (sklearn)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
gb.fit(x_train, y_train)
y_pred_gb = gb.predict(x_calib)
p_pred_gb = gb.predict_proba(x_calib)[:, 1]

print("=== Gradient Boosting ===")
print(classification_report(y_calib, y_pred_gb))
print("ROC-AUC GB:", roc_auc_score(y_calib, p_pred_gb))
