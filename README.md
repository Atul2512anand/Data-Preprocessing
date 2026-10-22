# Titanic Data Cleaning and Preprocessing

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/Data-pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

A complete, beginner-friendly preprocessing pipeline for the Titanic dataset, written as a Google Colab/Jupyter notebook. It turns the raw 891-row dataset into a cleaned, encoded, scaled, outlier-filtered CSV ready for modelling.

## Overview

The notebook demonstrates the routine steps of preparing tabular data:

- inspecting structure and missing values
- imputing missing values
- dropping an unusable column
- encoding categorical variables
- standardising numerical features
- detecting and removing IQR outliers
- exporting the cleaned dataset with a before/after visual trail

The included output, `titanic_cleaned.csv`, contains **718 rows × 12 columns** after cleaning.

## Dataset

| Item | Value |
|---|---|
| Source | [Kaggle — Titanic Dataset (yasserh)](https://www.kaggle.com/datasets/yasserh/titanic-dataset) |
| Raw file | `Titanic-Dataset.csv` |
| Raw shape | 891 rows × 12 columns |
| Target column | `Survived` (0 = did not survive, 1 = survived) |
| Cleaned output | `titanic_cleaned.csv`, 718 rows × 12 columns |

## Pipeline

The notebook has 12 cells and performs these steps in order:

1. **Load** the dataset with `kagglehub.dataset_download("yasserh/titanic-dataset")` or from the local CSV.
2. **Inspect** with `df.info()`, `df.describe()`, and a missing-value heatmap.
3. **Handle missing values**
   - `Age` (177 missing) → median
   - `Embarked` (2 missing) → mode
   - `Cabin` (687 missing) → drop the column
   - Drop any remaining rows with nulls
4. **Encode categorical features**
   - `Sex`: `male → 0`, `female → 1`
   - `Embarked`: one-hot encoding with `drop_first=True`, producing `Embarked_Q` and `Embarked_S`
5. **Standardise numeric features**
   - `StandardScaler` is applied to `Age` and `Fare`
6. **Remove outliers**
   - IQR rule on the scaled `Age` and `Fare` values
   - Rows outside `Q1 − 1.5×IQR` to `Q3 + 1.5×IQR` are removed
7. **Visualise** the before/after distributions and boxplots.
8. **Export** `titanic_cleaned.csv` with `df.to_csv(..., index=False)`.

### Final Columns

```text
PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch,
Ticket, Fare, Embarked_Q, Embarked_S
```

`Name`, `Ticket`, and `PassengerId` are retained as identifiers; they are not model-ready features by themselves.

## Getting Started

### Option A — Google Colab

1. Open `data_preprocessing.ipynb` in Google Colab.
2. Run the cells from top to bottom.
3. The notebook downloads the dataset through `kagglehub` and triggers a browser download for the cleaned CSV.

### Option B — Local Jupyter

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install pandas numpy matplotlib seaborn scikit-learn kagglehub jupyter
jupyter notebook data_preprocessing.ipynb
```

The final cell imports `google.colab.files` to download the output. Outside Colab, remove that import/download pair; the preceding `to_csv()` call already writes `titanic_cleaned.csv` to the working directory.

If you prefer not to use KaggleHub, replace the download cell with:

```python
df = pd.read_csv("Titanic-Dataset.csv")
```

## Project Structure

```text
Data-Preprocessing/
├── data_preprocessing.ipynb     # 12-cell preprocessing notebook
├── Titanic-Dataset.csv          # Raw input data
├── titanic_cleaned.csv          # Cleaned output (718 × 12)
└── README.md
```

## Results

| Stage | Rows | Columns |
|---|---:|---:|
| Raw dataset | 891 | 12 |
| After missing-value handling | 891 | 11 (`Cabin` removed) |
| After encoding | 891 | 12 (`Embarked_Q`, `Embarked_S` added) |
| After IQR filtering | **718** | **12** |

## Notes and Limitations

- The scaler is fitted on the full dataset before any train/test split. For a real model, fit `StandardScaler` on the training fold only to avoid data leakage.
- Outlier removal is performed on standardised values; the IQR bounds are equivalent to applying the rule to the raw values because standardisation is linear.
- `Name`, `Ticket`, and `PassengerId` remain in the output. Drop or engineer them before training.
- No feature engineering (title, family size, cabin deck) is performed in this notebook; it focuses on cleaning only.
- The notebook is educational and assumes the Titanic schema.

## Dataset Credit

The dataset is provided by Kaggle user **yasserh** and originates from the classic Titanic dataset. Refer to the Kaggle page for its licensing and usage terms.

## Author

**Atul Anand**  
B.Tech–M.Tech (Cyber Security), National Forensic Sciences University  
GitHub: [@Atul2512anand](https://github.com/Atul2512anand)
