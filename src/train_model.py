import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression # Exemple simple
from sklearn.metrics import mean_squared_error, r2_score

train_data_path = "data/processed/train_data.csv"
model_path = "models/model.pkl"
metrics_path = "metrics/metrics.json"

print(f"Chargement des données d'entraînement depuis {train_data_path}...")
train_df = pd.read_csv(train_data_path)

# Séparation des features (X_train) et de la cible (y_train)
X_train = train_df.drop("silica_concentrate", axis=1)
y_train = train_df["silica_concentrate"]

# Initialisation et entraînement du modèle
print("Entraînement du modèle...")
model = LinearRegression() # Remplacez par le modèle de votre choix
model.fit(X_train, y_train)
print("Modèle entraîné.")

# Sauvegarde du modèle
joblib.dump(model, model_path)
print(f"Modèle sauvegardé dans {model_path}")

# Évaluation du modèle sur les données d'entraînement (pour un premier aperçu)
y_pred_train = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

# Sauvegarde des métriques (vous pouvez ajouter d'autres métriques)
metrics = {
    "train_mse": mse_train,
    "train_r2": r2_train
}
import json
with open(metrics_path, "w") as f:
    json.dump(metrics, f)
print(f"Métriques sauvegardées dans {metrics_path}")