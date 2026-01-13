# Dataset Information

This folder contains the dataset used for training and evaluating the **Stress Level Classification from HRV Signals** project.

The dataset file used during training is:


---

## 🔹 Contents

| File name                | Description                                      |
|--------------------------|--------------------------------------------------|
| `train_cleaned_final.csv`| Preprocessed training dataset with numeric HRV features and target column `stress_class` |

---

## 🔹 Columns

The dataset contains **only numeric features** used for model training:

- `SDRR`
- `RMSSD`
- `KURT`
- `SKEW`
- `MEAN_REL_RR`
- `MEDIAN_REL_RR`
- `SDRR_RMSSD_REL_RR`
- `LF_NU`
- `sampen`
- `stress_class` *(target label — 5 classes)*

---

## 🔹 Source / Access

The dataset is **not uploaded to the repository** due to size and privacy considerations.

To obtain the dataset, you may:

- Request access from the author of this repository, or  
- Replace with your own HRV dataset having the features listed above.

---

## 🔹 Expected Format

The file `train_cleaned_final.csv` must be a **numeric tabular dataset** where:

- Each **row** represents a sample
- Each **column except `stress_class`** represents an input feature
- `stress_class` is the **integer target label** in the range **0 to 4**

### Example structure

| SDRR | RMSSD | KURT | SKEW | MEAN_REL_RR | MEDIAN_REL_RR | SDRR_RMSSD_REL_RR | LF_NU | sampen | stress_class |
|------|-------|------|------|-------------|----------------|--------------------|-------|--------|--------------|
| 125  | 39.2  | 2.10 | 0.25 | 0.98        | 1.01           | 0.74               | 52.1  | 1.28   | 3            |
| 112  | 30.4  | 1.85 | 0.11 | 1.03        | 1.00           | 0.69               | 47.8  | 0.92   | 1            |
| ...  | ...   | ...  | ...  | ...         | ...            | ...                | ...   | ...    | ...          |

---

### 🔻 Note for users replacing the dataset

If you want to use your own dataset:

1. Make sure the column names match **exactly** the feature list above  
2. Maintain the **same order of columns**
3. Save your file with the same name:

