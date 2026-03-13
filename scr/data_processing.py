import pandas as pd
import urllib.request
import os
from imblearn.over_sampling import SMOTE

# 1. CETTE PARTIE TROUVE LE DOSSIER DU PROJET AUTOMATIQUEMENT
# On cherche le dossier "project-1"
current_path = os.path.abspath(os.getcwd())
if "project-1" in current_path:
    # On remonte jusqu'à project-1 si on est dans scr
    base_dir = current_path.split("project-1")[0] + "project-1"
else:
    base_dir = current_path

data_folder = os.path.join(base_dir, "data")
source_path = os.path.join(data_folder, "heart.csv")
output_path = os.path.join(data_folder, "heart_balanced.csv")

# 2. Téléchargement
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
if not os.path.exists(source_path):
    urllib.request.urlretrieve(url, source_path)

# 3. Chargement et SMOTE
df = pd.read_csv(source_path)
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

df_final = pd.DataFrame(X_res, columns=X.columns)
df_final['DEATH_EVENT'] = y_res

# 4. Optimisation mémoire
def optimize_memory(df):
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type).startswith('int'):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif str(col_type).startswith('float'):
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df

df_final = optimize_memory(df_final)

# 5. SAUVEGARDE ET VÉRIFICATION
df_final.to_csv(output_path, index=False)

if os.path.exists(output_path):
    print("--- VÉRIFICATION RÉUSSIE ---")
    print(f"Le fichier est ici : {output_path}")
    print(f"Taille du fichier : {os.path.getsize(output_path) / 1024:.2f} KB")
else:
    print("--- ERREUR : Le fichier n'a pas pu être créé ---")
