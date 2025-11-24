---
# Credit Card Fraud Detection  
Machine Learning Project | Imbalanced Classification | Python

## 📌 Overview
This project builds a complete fraud detection pipeline using the popular **Credit Card Fraud Detection dataset**, containing real transaction patterns with **extreme class imbalance (0.17% fraud)**.  
The goal is to identify fraudulent transactions while minimizing false positives using machine learning techniques and proper imbalance handling.

---

## 📂 Project Structure
## 📊 Dataset Summary
- **Rows:** 284,807  
- **Features:** 30 PCA-transformed components (V1–V28), Time, Amount  
- **Target:**  
  - `0` → legitimate transactions  
  - `1` → fraudulent transactions  
- **Fraud Ratio:** 0.17% (highly imbalanced)

Because features are anonymized via PCA transformation, interpretability is limited — but the statistical patterns still allow accurate fraud prediction.

---

## 🔎 Phase 1 — Initial Inspection
- No missing values.  
- Features are scaled PCA components + Time + Amount.  
- Dataset is ready for EDA and modeling without heavy preprocessing.

---

## 📈 Phase 2 — EDA (Amount & Time Analysis)

### 💵 Transaction Amount Insights
- Fraud transactions have **small median amounts** (~9 units).
- Fraud includes occasional **medium spikes** (~1000–2000).
- Non-fraud has **very large outliers** (up to 25k).
- Amount is **highly skewed**, so log-scaling is important.

### 🕒 Time Pattern Insights
- Fraud clusters between **40,000–130,000 seconds**, not uniform.
- Indicates fraud may occur during less monitored periods.
- Useful as a temporal feature.

---

## 🔍 Phase 3 — PCA & Correlation Analysis
PCA components carry the strongest fraud-detection signals.

### 🔥 Most Predictive PCA Features
- **V14** → sharp negative clustering for fraud  
- **V17** → clear separation  
- **V12** → moderate separation  
- **V2** → weaker but still useful

### Heatmap Observation
- PCA features show meaningful relationships with the target.
- Direct interpretation is limited, but important for ML performance.

---

## ⚖️ Phase 4 — Handling Class Imbalance
Fraud = 0.17% → models must be adjusted or they predict "0" for everything.

### Methods Applied:
#### 1. **Class-Weighted Logistic Regression**
- High recall (0.92)  
- Very low precision (0.06)  
- Detects fraud but creates too many false alarms.

#### 2. **Undersampling**
- Perfect precision/recall on balanced subset  
- NOT realistic because 99% of data is removed  
- Used only as a reference.

#### 3. **SMOTE Oversampling**
- Creates synthetic fraud samples  
- Better F1 (0.23)  
- Still inferior to tree-based models

---

## 🤖 Phase 5 — Model Training & Evaluation

### Models evaluated:
- Logistic Regression (class weights)
- Logistic Regression + SMOTE
- Logistic Regression (undersampled)
- **Random Forest (class_weight='balanced') → BEST MODEL**

### 📊 Final Model Comparison (Fraud Class Only)

| Model                             | Precision | Recall | F1-score |
|----------------------------------|-----------|--------|----------|
| Logistic Regression (class_weight) | 0.06      | 0.92   | 0.11     |
| Logistic Regression (SMOTE)        | 0.13      | 0.90   | 0.23     |
| Random Forest (class_weight)       | **0.96**  | **0.76** | **0.85** |

### 🏆 Final Model Selected: **Random Forest**
- Excellent precision → very few false alarms  
- Strong recall → detects most fraud  
- Robust to noise and non-linear patterns  
- Best overall fraud-detection trade-off

---

## 🧠 Business Interpretation (High-Level)
- Fraud tends to be **low-value**, **time-patterned**, and concentrated in specific PCA ranges.
- The Random Forest model provides:
  - High accuracy on real-world skewed data
  - High precision to avoid investigation overload
  - Strong recall to capture most fraudulent cases

---

## 📌 Limitations
- PCA features lack interpretability.  
- Data only reflects transactions over two days.  
- No categorical/contextual features (e.g., merchant type, geography).  
- XGBoost was attempted but **not supported in the macOS Python environment** due to `libomp` dependency issues.

---

## 🚀 Future Improvements
- Add anomaly detection (Isolation Forest, Autoencoders)
- Try deep learning architectures on a GPU-enabled environment
- Use cost-sensitive learning to further optimize precision/recall trade-offs
- Deploy model via FastAPI + Docker

---

## 📎 How to Run This Project
```bash
pip install -r requirements.txt
jupyter notebook