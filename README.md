# 🌽 Maize Yield Prediction Using Supervised Machine Learning

### *A Complete End-to-End Data Science Project*

📌 **Presentation Website:**

[https://sites.google.com/view/maizeyieldprediction/](https://sites.google.com/view/maizeyieldprediction/)

---

## 📘 **Project Overview**

Agricultural productivity is crucial for food security, economic stability, and sustainable development. This project develops and evaluates machine learning models for predicting **maize yield** using climatic, temporal, and agricultural input variables across multiple countries from  **1990 to 2013** .

The final model—an optimized  **ExtraTreesRegressor** —achieves strong predictive performance and provides insight into the factors driving maize yield variation globally and regionally.

---

# 🔍 **1. Problem Definition**

Maize is one of the world’s most important cereal crops. Predicting maize yield accurately is valuable for:

* Food security planning
* Market forecasting
* Policy development
* Agricultural resource management

However, maize yield depends on complex interactions between:

* Climate (temperature, rainfall)
* Agricultural inputs (pesticides)
* Temporal trends
* Country-level factors (policies, practices, soil qualities)

**Goal:**

Develop a supervised machine learning model to predict maize yield using annual climate and agricultural data, while analyzing yield trends and understanding key drivers of productivity.

---

# 📥 **2. Data Collection**

Data were obtained from international agricultural datasets and include:

* **Yield (target variable)**
* **Average rainfall**
* **Average temperature**
* **Pesticides usage**
* **Year**
* **Country/Region (“Area”)**

Time span: **1990–2013**

Number of samples: **~4,100**

---

# 🧹 **3. Data Preprocessing**

### **3.1 Handling Missing Values**

Minor missing entries were imputed as needed.

### **3.2 Encoding Categorical Variables**

`Area` (country) was one-hot encoded, expanding the feature space substantially:

* Final training matrix shape: **(3208, 94)**

### **3.3 Scaling Numerical Features**

A **RobustScaler** was applied to mitigate skewness and outlier influence.

### **3.4 Train/Test Split (Temporal)**

To mimic real forecasting:

* **Training:** 1990–2008 → (3208 samples)
* **Testing:** 2009–2013 → (913 samples)

---

# 📊 **4. Exploratory Data Analysis (EDA)**

### **4.1 Distribution Analysis**

* Yield, rainfall, and pesticides show  **strong right skew** .
* Temperature shows **multimodal** behavior reflecting climatic diversity.

### **4.2 Correlation Insights**

* **Temperature** has the strongest linear relationship with yield (–0.55).
* Rainfall and pesticides show weaker correlations.
* Year shows a moderate positive correlation (+0.24), reflecting global productivity growth.

### **4.3 Time-Series Patterns**

* **Global maize yield increased steadily** from 1990 to 2013.
* Country-level analysis revealed:
  * Uganda & Rwanda: **dramatic yield improvements post-2007**
  * Kenya: **stable but fluctuating yields**
  * Burundi: **gradual decline then recovery**

---

# 🤖 **5. Model Training**

Multiple tree-based regression models were trained:

* RandomForestRegressor
* ExtraTreesRegressor
* GradientBoostingRegressor
* AdaBoostRegressor
* XGBoostRegressor

### **Performance Comparison (Test Set)**


| Model                | MAE            | RMSE           | R²             |
| -------------------- | -------------- | -------------- | --------------- |
| **ExtraTrees** | **5308** | **8692** | **0.915** |
| RandomForest         | 6053           | 10393          | 0.879           |
| XGBoost              | 7157           | 11160          | 0.860           |
| GradientBoosting     | 10057          | 14330          | 0.770           |
| AdaBoost             | 15546          | 19206          | 0.586           |


**Winner:** **ExtraTreesRegressor**

Best combination of accuracy, stability, and generalization.

---

# 🔧 **6. Hyperparameter Tuning**

A GridSearchCV (5-fold) explored 243 candidate configurations (1215 total fits).

**Best Parameters:**

{
  "model__max_depth": 40,
  "model__max_features": "log2",
  "model__min_samples_leaf": 1,
  "model__min_samples_split": 2,
  "model__n_estimators": 400
}


**Best CV RMSE:** 21,987

---

# 🏁 **7. Final Model Performance**

The tuned ExtraTreesRegressor achieved:

| Metric         | Value              |
| -------------- | ------------------ |
| **MAE**  | **6857.82**  |
| **RMSE** | **10300.17** |
| **R²**  | **0.8810**   |

### Interpretation:

* Explains **88% of yield variance** for unseen years.
* Accurate for most countries, but sometimes underpredicts in regions with **sudden yield growth** (e.g., Rwanda, Uganda, Albania).
* Very strong overall predictive power.

---

# 📉 **8. Model Diagnostics**

### **8.1 Residual Distribution**

* Centered near zero → **no global bias**
* Right-skewed tail → model underpredicts unusually high yields

### **8.2 Actual vs Predicted Scatterplot**

* Tight clustering around diagonal
* Underprediction at yields > 100,000

### **8.3 Country-Level Error Analysis**

Mean absolute residual by country (East Africa):

* Rwanda: ~8300
* Kenya: ~5200
* Burundi: ~4200
* **Uganda: ~2000 (best performance)**

### Insight:

Model performs well where trends are stable, but struggles with  **structural breaks** .

---

# 🌟 **9. Feature Importance Analysis**

Top continuous predictors:

| Feature            | Importance       |
| ------------------ | ---------------- |
| **avg_temp** | **0.1294** |
| Year               | 0.0428           |
| pesticides         | 0.0409           |
| avg_rainfall       | 0.0364           |

**Key insight:**

Temperature is the single most important climate predictor.

Country dummies (one-hot encoded) collectively carry substantial weight.

---

# 🧠 **10. Key Findings**

* Tree-based models effectively capture complex yield patterns.
* ExtraTreesRegressor offers strong accuracy and interpretability.
* Temperature is the most influential environmental factor.
* Rainfall and pesticide effects are weaker and more nonlinear.
* Temporal split shows strong generalization but reveals challenges predicting sharp post-2007 yield increases.

---

# 🧭 **11. Recommendations**

### **1. Add more agronomic & socioeconomic features**

* Soil fertility
* Irrigation coverage
* Fertilizer types
* Seed varieties
* Agricultural investment data

  Will greatly improve model accuracy.

### **2. Explore temporal or hybrid models**

* LSTM / RNN
* Temporal Fusion Transformers
* Gradient boosting with lag features

  These models learn time dependencies better.

### **3. Build country-specific submodels**

Better for countries with unique yield dynamics or recent structural changes.

### **4. Model structural breaks explicitly**

Add indicators for policy changes or major agricultural reforms.

### **5. Expand dataset beyond 2013**

Improves generalization and captures recent productivity trends.

### **6. Include uncertainty estimation**

Quantile forests or Bayesian models → more actionable predictions.

---

# 🎯 **12. Conclusion**

This project demonstrates the effectiveness of machine learning—particularly  **ExtraTreesRegressor** —in predicting maize yield across diverse countries and climatic conditions. With an R² of  **0.88** , the final model provides reliable forecasts and highlights temperature as the most critical driver of maize productivity.

While the model generalizes well, its challenges predicting sudden structural shifts emphasize the need for additional features and temporal modeling techniques. With further refinement, this approach has strong potential for supporting food security planning, agricultural investment strategies, and policy development.

📌 **Full presentation and narrative available here:**

[https://sites.google.com/view/maizeyieldprediction/](https://sites.google.com/view/maizeyieldprediction/)
