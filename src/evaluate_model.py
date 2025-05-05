import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json
import numpy as np # Pour calculer la RMSE

test_data_path = "data/processed/test_data.csv"
model_path = "models/model.pkl"
metrics_path = "metrics/metrics.json" # Ajout des métriques de test à ce fichier

print(f"Chargement des données de test depuis {test_data_path}...")
test_df = pd.read_csv(test_data_path)

# Séparation des features (X_test) et de la cible (y_test)
X_test = test_df.drop("silica_concentrate", axis=1)
y_test = test_df["silica_concentrate"]

# Chargement du modèle
print(f"Chargement du modèle depuis {model_path}...")
model = joblib.load(model_path)

# Évaluation du modèle sur les données de test
print("Évaluation du modèle sur les données de test...")
y_pred_test = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)

# Chargement des métriques existantes et ajout des métriques de test
try:
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
except FileNotFoundError:
    metrics = {}

metrics["test_mse"] = mse_test
metrics["test_rmse"] = rmse_test
metrics["test_r2"] = r2_test

# Sauvegarde des métriques mises à jour
with open(metrics_path, "w") as f:
    json.dump(metrics, f)
print(f"Métriques de test sauvegardées dans {metrics_path}")