import pandas as pd
from sklearn.model_selection import train_test_split

raw_data_path = "data/raw/raw.csv"
processed_data_path = "data/processed/processed_data.csv"
train_data_path = "data/processed/train_data.csv"
test_data_path = "data/processed/test_data.csv"

print(f"Chargement des données brutes depuis {raw_data_path}...")
df = pd.read_csv(raw_data_path)

# Séparation des features (X) et de la cible (y)
X = df.drop("silica_concentrate", axis=1)
y = df["silica_concentrate"]

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Concaténation pour sauvegarder les ensembles séparément
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Sauvegarde des données prétraitées (optionnel, si vous voulez une version "complète" prétraitée)
df.to_csv(processed_data_path, index=False)
print(f"Données prétraitées sauvegardées dans {processed_data_path}")

# Sauvegarde des ensembles d'entraînement et de test
train_df.to_csv(train_data_path, index=False)
test_df.to_csv(test_data_path, index=False)
print(f"Ensembles d'entraînement et de test sauvegardés dans {train_data_path} et {test_data_path}")