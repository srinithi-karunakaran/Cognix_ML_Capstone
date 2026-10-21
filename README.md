# Cognix ML Capstone

End-to-end machine learning pipeline covering **Regression**, **Classification** and **Clustering**, built for the **23CSE301 Machine Learning** capstone by **Team Cognix**. Each track has a full notebook (EDA → preprocessing → modelling → evaluation) and the three tracks are also available in an interactive **Streamlit app (Cognix ML Studio)**.

## Tracks at a Glance

| Track | Dataset | Task | Best Model | Key Result |
|---|---|---|---|---|
| Regression | WHO Life Expectancy (2,938 rows, 193 countries, 2000–2015) | Predict life expectancy | Tuned Gradient Boosting | R² = 0.9588, RMSE = 1.91 |
| Classification | Telco Customer Churn (7,043 customers) | Predict churn (Yes/No) | Logistic Regression | F1 = 0.8001, ROC-AUC = 0.8460 |
| Clustering | TMDB 5000 Movies (4,740 after cleaning) | Group movies by profile | K-Means (k = 5) | Silhouette = 0.166 |

## Project Structure

```
Cognix_ML_Capstone/
├── app/
│   └── app.py                 # Streamlit app: Cognix ML Studio
├── data/
│   ├── regression/            # Life Expectancy Data.csv
│   ├── classification/        # WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── clustering/            # tmdb_5000_movies.csv, tmdb_5000_credits.csv
├── models/                    # Artifacts saved by the classification notebook
├── notebooks/
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
├── requirements.txt
└── README.md
```

## 1. Regression: Life Expectancy (WHO)

**Goal:** predict life expectancy from socio-economic and health indicators.

**Pipeline**
- Audit: 14 columns with missing values (Population 22%, Hepatitis B 19%, GDP 15%); `Country` dropped as a high-cardinality identifier.
- Stratified 80:20 split on `Status` (Developed/Developing).
- Median/mode imputation, IQR capping, one-hot encoding of `Status`, standard scaling, all fit on train only.
- Engineered features: `Schooling_Income`, `BMI_Schooling`.

**Models (10), tuned with GridSearchCV / RandomizedSearchCV:**

| Rank | Model | R² | RMSE | MAE |
|---|---|---|---|---|
| 1 | **Gradient Boosting (tuned)** | **0.9588** | **1.9105** | 1.2710 |
| 2 | KNN (tuned) | 0.9573 | 1.9439 | 1.2416 |
| 3 | Random Forest (tuned) | 0.9558 | 1.9789 | 1.2761 |
| 4 | SVR (tuned) | 0.9397 | 2.3104 | 1.4806 |
| 5 | Polynomial (degree 2) | 0.9106 | 2.8133 | 2.0025 |
| 6 | Decision Tree (tuned) | 0.9097 | 2.8279 | 1.8903 |
| 7–10 | Linear / Lasso / Ridge / ElasticNet | ≈ 0.85 | ≈ 3.62 | ≈ 2.70 |

5-fold CV of the top two: Gradient Boosting R² = 0.9576 ± 0.0032, KNN R² = 0.9417 ± 0.0071.

**Findings:** non-linear models clearly beat linear ones. Most important features (Random Forest): **HIV/AIDS (0.59)**, **Income composition of resources (0.15)** and **Adult Mortality (0.14)**. `infant deaths` and `under-five deaths` are nearly collinear, so linear coefficients should be read as indicative only.

## 2. Classification: Telco Customer Churn

**Goal:** predict whether a customer will churn (26.5% churn, so the target is imbalanced and accuracy alone is misleading).

**Pipeline**
- `TotalCharges` converted to numeric; its 11 missing values belong to tenure-0 customers and are set to 0 (not the median).
- Stratified 80:20 split, IQR capping, one-hot encoding (`drop='if_binary'`), standard scaling, all fit on train only.
- Engineered features: `Charges_Per_Tenure`, `NumServices`.

**Models (10):** Part A (Logistic Regression, Gaussian NB, Decision Tree, SVM, KNN) and Part B ensembles (Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP), tuned with `GridSearchCV`.

| Rank | Model | Accuracy | F1 (weighted) | ROC-AUC |
|---|---|---|---|---|
| 1 | **Logistic Regression** | 0.8077 | **0.8001** | 0.8460 |
| 2 | MLP | 0.8013 | 0.7978 | 0.8413 |
| 3 | Gradient Boosting | 0.8055 | 0.7955 | 0.8444 |
| 4 | Random Forest | 0.8020 | 0.7937 | 0.8400 |
| 5 | AdaBoost | 0.7991 | 0.7922 | 0.8463 |
| 6 | SVM | 0.8034 | 0.7916 | 0.8314 |
| 7 | Bagging | 0.7942 | 0.7847 | 0.8450 |
| 8 | KNN | 0.7729 | 0.7715 | 0.8091 |
| 9 | Decision Tree | 0.7459 | 0.7590 | 0.8144 |
| 10 | Gaussian Naive Bayes | 0.6913 | 0.7093 | 0.8080 |

**Final model:** Logistic Regression. In 5-fold CV it is statistically tied with the MLP on F1 (0.7986 ± 0.0108 vs 0.8006 ± 0.0151) but has higher CV ROC-AUC (0.8486 vs 0.8426) and lower variance, and it is simpler and more interpretable. Its weakness is churn-class recall (0.53), so threshold tuning is a logical next step.

**Churn drivers:** short tenure, month-to-month contracts, fiber optic internet, and no online security/tech support raise churn risk; two-year contracts and long tenure lower it.

**Saved artifacts** (`models/`): `best_classifier.pkl`, `preprocessing.pkl`, `model_comparison.csv`.

## 3. Clustering: TMDB 5000 Movies

**Goal:** discover natural groups of movies (unsupervised, so no target or train/test split).

**Pipeline**
- Movies and credits files merged on movie id.
- Zero `budget`/`revenue`/`runtime` treated as missing and median-filled; movies with no votes removed (4,740 left); outliers capped at the 1st/99th percentile.
- `log1p` on skewed columns, then standard scaling; `main_genre` (top 5 + Other) one-hot encoded; `is_english` flag.
- Engineered features: `cast_size`, `crew_size`, `release_year` (15 features total).

**Algorithms:** K-Means and Agglomerative (Ward), both with k = 5 (elbow curve and silhouette scan over k = 2–10). Linkage comparison showed `single` and `average` produce one giant cluster, so Ward was used.

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|
| **K-Means** | **0.166** | 1.725 | **839.6** |
| Agglomerative (Ward) | 0.136 | **1.696** | 641.6 |

**K-Means cluster profiles:** big-budget blockbusters, mainstream medium-to-high budget films, older high-rated films (average year ≈ 1971), low-budget low-revenue films, and films with very few votes and low ratings. Scores are modest, which is expected for overlapping real-world movie data. Visualised with PCA (52% variance in 2 components) and t-SNE.

## Cognix ML Studio (Streamlit App)

`app/app.py` is an interactive front end for all three tracks. Pick a track in the sidebar, adjust the inputs, and get a prediction.

| Track | Input | Output |
|---|---|---|
| Regression | Country indicators (Status, mortality, schooling, etc.) | Predicted life expectancy in years |
| Classification | Customer account and service details | Churn prediction (Yes/No) and churn probability |
| Clustering | Movie popularity, runtime, vote average, vote count | Assigned cluster and distance from its centre |

**How it works**
- The bundled CSVs in `data/` load automatically, so no upload is needed.
- When a track is opened, the app **trains five candidate models on the bundled data**, picks the best on a held-out 20% split (R² for regression, weighted F1 for classification), then refits it on all rows. Results are cached for the session.
  - Regression candidates: Random Forest, Extra Trees, Gradient Boosting, Ridge, Decision Tree.
  - Classification candidates: Random Forest, Extra Trees, Gradient Boosting, Logistic Regression, Decision Tree.
- For clustering, K-Means is run for k = 2–6 on four numeric features (`popularity`, `runtime`, `vote_average`, `vote_count`) and the k with the best silhouette score is kept.
- The app builds its own preprocessing pipeline and does **not** load the `.pkl` files in `models/`, so its selected models and scores can differ from the notebooks.

**Run it**

```bash
streamlit run app/app.py
```

## Setup

```bash
git clone https://github.com/srinithi-karunakaran/Cognix_ML_Capstone.git
cd Cognix_ML_Capstone
pip install -r requirements.txt
```

Main libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, joblib, streamlit.

To run the notebooks, open them in Jupyter from the `notebooks/` folder and run all cells. Regression and classification read data from `../data/...`; the clustering notebook expects `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the same folder as the notebook, so copy them there or update the paths.

## Reproducibility

`random_state=42` is used everywhere, all preprocessing is fit on training data only (no leakage), and the test set is used only for final evaluation.

## Team

**Team Cognix**, Amrita Vishwa Vidyapeetham, Coimbatore
- Srinithi Karunakaran
- swethamohankumarr
- SanaVenkataVarshitha

## Course

23CSE301 Machine Learning