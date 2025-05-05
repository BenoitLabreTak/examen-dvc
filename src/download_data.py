import pandas as pd

url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
output_path = "data/raw/raw.csv"

print(f"Téléchargement des données depuis {url}...")
df = pd.read_csv(url)
df.to_csv(output_path, index=False)
print(f"Données téléchargées et sauvegardées dans {output_path}")