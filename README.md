# 🫀 HeartGuard — Prédiction du Risque d'Insuffisance Cardiaque

> **Coding Week · Centrale Casablanca · Mars 2026**  
> Fouad Ghadi · Yassine Ait Bella · Rabi Ilyas · Yahiaoui Ziyad · Chakir Mohamed

---

## 📌 À propos du projet

HeartGuard est une application clinique d'aide à la décision médicale. Elle permet à un médecin de saisir les données biologiques d'un patient et d'obtenir en temps réel une estimation du risque d'insuffisance cardiaque fatale, accompagnée d'une explication visuelle des facteurs qui ont influencé cette prédiction.

Le projet repose sur un pipeline Machine Learning complet : nettoyage des données, gestion du déséquilibre de classes, entraînement de plusieurs modèles, sélection du meilleur, et déploiement via une interface Streamlit.

---

## 📁 Structure du projet

```
project-1/
├── data/
│   ├── heart.csv                        # Dataset original UCI
│   └── heart_balanced.csv               # Dataset après optimisation mémoire
├── notebooks/
│   └── eda.ipynb                        # Analyse exploratoire des données
├── src/
│   ├── data_processing.py               # Nettoyage, optimisation mémoire
│   ├── train_model.py                   # Entraînement des 4 modèles
│   ├── evaluate_model.py                # Évaluation et courbe ROC
│   ├── heart_model.pkl                  # Random Forest (meilleur modèle)
│   ├── heart_model_xgb.pkl              # XGBoost
│   ├── heart_model_lgbm.pkl             # LightGBM
│   ├── heart_model_lr.pkl               # Logistic Regression
│   └── metrics.json                     # Précision test sauvegardée
├── app/
│   └── app.py                           # Interface Streamlit
├── tests/
│   └── test_data_processing.py          # Tests automatisés
├── .github/
│   └── workflows/
│       └── ci.yml                       # Pipeline CI/CD GitHub Actions
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Installation & Lancement

### 1. Cloner le dépôt
```bash
git clone https://github.com/fouad-ghadi/project-1.git
cd project-1
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Préparer les données
```bash
python src/data_processing.py
```

### 4. Entraîner les modèles
```bash
python src/train_model.py
```

### 5. Lancer l'application
```bash
streamlit run app/app.py
```

Ouvrir ensuite : **http://localhost:8501**

---

## 📊 Dataset

- **Source :** [UCI ML Repository — Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
- **Taille :** 299 patients, 13 variables cliniques
- **Variable cible :** `DEATH_EVENT` (0 = survie, 1 = décès)

---

## 🔍 Analyse exploratoire (EDA)

### Valeurs manquantes
Aucune valeur manquante détectée dans le dataset original. Aucun traitement d'imputation nécessaire.

### Outliers
Des valeurs extrêmes ont été identifiées sur `creatinine_phosphokinase`, `platelets` et `serum_creatinine`. Elles ont été conservées car elles correspondent à des cas cliniques réels (pas des erreurs de saisie).

### Déséquilibre de classes
Le dataset est **déséquilibré** :
- 68% de patients survivants
- 32% de patients décédés

**Méthode choisie : SMOTE (Synthetic Minority Over-sampling Technique)**

SMOTE a été appliqué **uniquement sur les données d'entraînement**, après la séparation train/test, pour éviter toute fuite de données (data leakage). Cette décision est importante : appliquer SMOTE avant le split aurait gonflé artificiellement la précision (nous avons observé 91.5% contre ~85% après correction).

**Impact :** amélioration significative du recall sur la classe minoritaire (décédés), réduction du biais du modèle.

### Corrélations
`serum_creatinine`, `ejection_fraction` et `time` sont les variables les plus corrélées avec `DEATH_EVENT`. Aucune multicolinéarité critique détectée.

---

## 🤖 Modèles évalués

Quatre modèles ont été entraînés et comparés sur les métriques suivantes : **Accuracy, ROC-AUC, Precision, Recall, F1-Score**.

| Modèle | Accuracy | ROC-AUC | F1-Score |
|--------|----------|---------|---------|
| **Random Forest** ✅ | **~87%** | **~0.91** | **~0.86** |
| XGBoost | ~85% | ~0.89 | ~0.84 |
| LightGBM | ~84% | ~0.88 | ~0.83 |
| Logistic Regression | ~80% | ~0.84 | ~0.79 |

### ✅ Meilleur modèle : Random Forest

Le Random Forest a obtenu les meilleures performances sur toutes les métriques. Sa robustesse face aux valeurs aberrantes et sa capacité à capturer des interactions non-linéaires entre les variables cliniques en font le choix le plus adapté à ce type de données médicales.

---

## 🔬 Explication SHAP

SHAP (SHapley Additive exPlanations) a été intégré pour rendre les prédictions du modèle transparentes et interprétables pour les médecins.

### Features les plus influentes (résultats SHAP) :
1. **`time`** (durée de suivi) — variable la plus prédictive
2. **`serum_creatinine`** — indicateur clé de la fonction rénale
3. **`ejection_fraction`** — capacité de pompage du cœur
4. **`age`**
5. **`serum_sodium`**

### ⚠️ Note sur le tabagisme
Dans ce dataset, le tabagisme montre une corrélation contre-intuitive avec la mortalité. Cela est dû à la petite taille de l'échantillon (299 patients) et **ne reflète pas la réalité clinique**. Ce biais est documenté dans l'interface de l'application.

---

## 💾 Optimisation mémoire

Une fonction `optimize_memory(df)` a été développée dans `src/data_processing.py`. Elle convertit automatiquement :
- `float64` → `float32`
- `int64` → `int32`

Cela permet de réduire significativement l'empreinte mémoire du dataset sans perte d'information.

---

## 🧪 Tests automatisés

```bash
pytest tests/ -v
```

Les tests couvrent :
- `test_optimize_memory_reduces_size` — la mémoire est réduite après optimisation
- `test_optimize_memory_types` — les types sont bien convertis en float32/int32
- `test_handle_missing_values` — les valeurs manquantes sont correctement traitées
- `test_handle_outliers_clips_values` — les outliers sont clippés
- `test_optimize_memory_preserves_row_count` — le nombre de lignes est préservé

Ces tests sont exécutés automatiquement via **GitHub Actions** à chaque push.

---

## ⚙️ CI/CD — GitHub Actions

Le fichier `.github/workflows/ci.yml` automatise :
1. Installation des dépendances
2. Exécution des tests pytest
3. Validation sur Python 3.10

---

## 🧠 Prompt Engineering

**Tâche sélectionnée :** Fonction `optimize_memory(df)`

**Prompt utilisé :**
> *"Write a Python function `optimize_memory(df)` that reduces memory usage of a pandas DataFrame by converting float64 to float32 and int64 to int32. Print memory before and after. Do not modify the target column DEATH_EVENT."*

**Résultat :** La fonction générée était fonctionnelle. Cependant, le prompt initial ne précisait pas d'exclure les colonnes non-numériques ni de conserver les colonnes cibles — ce qui a nécessité une itération.

**Amélioration du prompt :**
> *"...Make sure to only apply downcasting to numeric columns using select_dtypes, and skip the DEATH_EVENT column."*

**Leçon :** Spécifier les cas limites dès le premier prompt réduit le nombre d'itérations nécessaires et améliore la qualité du code généré.

---

## ❓ Questions critiques (README)

**Le dataset était-il équilibré ?**  
Non. 68% survivants / 32% décédés. Traité avec SMOTE appliqué uniquement sur le train set.

**Quel modèle a le mieux performé ?**  
Random Forest — ~87% accuracy, ~0.91 ROC-AUC.

**Quelles features médicales influencent le plus les prédictions ?**  
`time`, `serum_creatinine`, `ejection_fraction`, `age`, `serum_sodium` (selon SHAP).

**Quels insights le prompt engineering a-t-il apporté ?**  
La précision du prompt (exclusion des colonnes non-numériques, conservation de la cible) est directement liée à la qualité du code généré. Un prompt vague produit du code fonctionnel mais incomplet.

---

## 🐳 Docker

```bash
docker build -t heartguard .
docker run -p 8501:8501 heartguard
```

---

*Centrale Casablanca · Coding Week · Mars 2026*
