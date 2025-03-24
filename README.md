# Demonstration ML Project

This project demonstrates the workflows and technologies common to many machine learning (ML) projects. The focus is on stardard ML, as opposed to deep learning which uses neural networks. It is inspired by the popular ["Emergency - 911 Calls"](https://www.kaggle.com/datasets/mchirico/montcoaler) dataset hosted on Kaggle.

### Simualted 911 Scenario

Local officials in Montgomery County, PA have approved funds to purchase additional vehicals and hire additional staff to be placed at existing facilities. It is assumed that a new fire truck, police cruiser, and ambulance (one of each) will be purchased for each participating township. A regional planning commission (RPC) has been tasked with providing a data-driven report containing information that can help the county, township, and municiple officials decide how to distribute the vehicles among the members and for which work shifts to hire new staff.

**Montgomery County Map**

Source: [https://mcato.us/member-directory/montgomery-county-map/](https://mcato.us/member-directory/montgomery-county-map/)

![https://mcato.us/wp-content/uploads/2023/07/montgomerymap.png](https://mcato.us/wp-content/uploads/2023/07/montgomerymap.png)

**PA County Map**

Source: [https://www.pa.gov/agencies/penndot/maps/county-type-10.html](https://www.pa.gov/agencies/penndot/maps/county-type-10.html)

![https://s7d9.scene7.com/is/image/statepa/pa-multicolor](https://s7d9.scene7.com/is/image/statepa/pa-multicolor)

### Revelance of the Simulation

ML projects create the most value when ML engineers and SMEs collaborate together to create relevant and actionable information products. As a ML demonstration project, there is no access to a true subject matter expert (SME) in 911 response systems or programs. Without the participation of a 911 SME, the outputs of this project will not be relevant to real-world scenarios. However, the project will demonstration a valid application of workflow steps, techniques, and tools in the ML space.

## Workflow

The following steps TODO

1. Objectives Definition (see below)
2. Data Collection (see below)
3. Data Preprocessing & Cleaning (see below)
4. Exploratory Data Analysis (EDA)

## 1. Objectives Definition

The RPC has decided to include the following ML predictions in their analysis report.

1. Given a call's time and location, what most likely service required
2. Given a call's service and location, what is the most likely day of week.
3. Given a call's service, location, and day of week, what is the most likely shift

### Success Criteria

- The ML models predict the specified class in the training set with 90% confidence and 90% accuracy

_NOTE: This objective definition has been contrived without the advice of a 911 SME._

## 2. Data Collection

- **Emergency (911) Calls: Fire, Traffic, EMS for Montgomery County, PA**
  > Mike Chirico. (2020). Emergency - 911 Calls [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/1381403

## 3. Data Preprocessing and Cleaning

1. Notebook - **[Feature Engineering - twp - 1](https://github.com/thomaspurk/ml-demo-911/blob/main/data-engineering/feature-engineering-twp-1.ipynb)**

   - Removed records with null values - Removed records covering calls in adjacent counties
   - Normalized 'twp' so every name had either 'TOWNSHIP' or 'BOROUGH'
   - Unique TWP Count: 68 -> 62 - Record Count: 663,522 -> 649,696
   - Decision was to normalize twp values and add a twp_type column.
     <br><br>

2. Notebook - **[Feature Engineering - lat/lng - 1](https://github.com/thomaspurk/ml-demo-911/blob/main/data-engineering/feature-engineering-lat-lng-1.ipynb)**

   - Data had low null rates, but analysis show large number of records using default lat,lng values that were not associated with the actual position
   - Decision was to remove the lat/lng columns since the data quality issues make it difficult to use these columns for imputation of other missing features, and they are not good candidate for direct feed into a ML model.
     <br><br>

3. Notebook - **[Feature Engineering - desc - 1](https://github.com/thomaspurk/ml-demo-911/blob/main/data-engineering/feature-engineering-desc-1.ipynb)**

   - The desc column has no nulls and is highly unique
   - Values seem to contain multiple features delimited by ";"
   - But investigation revealed that ";" is not a reliable delimiter
   - There could be a station feature embedded in the desc.
   - After extracting a station feature from desc, 34% were null
   - Given the high presentange of nulls, and uncertainty in how a station feature would improve model analysis over other features, like addr. The decision was to delete both the desc and station features.
     <br><br>

4. Notebook - **[Feature Engineering - addr - 1](https://github.com/thomaspurk/ml-demo-911/blob/main/data-engineering/feature-engineering-addr-1.ipynb)**

   - No nulls
   - 38,860 unique
   - The unique counts are possibily exaggerated due to reversed streets, Main St.& Elm St. vs Elm St. & Main St.
   - 25,792 addrs are repeated 10 less than 10 times
   - Road names can be repeat accross geographic regions
   - The top 5 repeated address, when manually searched via google maps, are assited living, health centers, apartment complex, and an outlet mall.
   - Given the high number of unique value, and the low number of records covered by address that repeat 10 or more times, the decision is to create a road_type feature based on the road type suffix extracted from addr.
     <br><br>

5. Notebook - **[Feature Engineering - title - 1](https://github.com/thomaspurk/ml-demo-911/blob/main/data-engineering/feature-engineering-title-1.ipynb)**

   - No nulls - 147 unique values
   - There seems to be two features seperated by a ':'
   - Some values have " - " at the end. Removing will make them part of another group
   - Decision is to split into two new feature 'service_type' (EMS, FIRE, TRAFFIC) and 'service_desc' (88 unique values)
     <br><br>

6. Notebook - **[Feature Engineering - timeStamp - 1](https://github.com/thomaspurk/ml-demo-911/blob/main/data-engineering/feature-engineering-timestamp-1.ipynb)**

   - No nulls
   - high degree of uniqueness - Assumption, season of the year, for which month is a proxy could impact the type and frequency of 911 calls
   - Assumption, day time or night time could impact the type and frequency of 911 calls
   - Assumption, day of week could impact the type and frequency of 911 calls
   - So extracted month, day of week, and day/night from timeStamp to create new columns
   - Dropped the timeStamp column
     <br><br>

7. Notebook - **[Feature Engineering - zip -1](https://github.com/thomaspurk/ml-demo-911/blob/main/data-engineering/feature-engineering-zip-1.ipynb)**
   - 12% null values
   - 164 unique values
   - twp is also a geographic identifier like zip code, but twp is 0% null
   - zip has greater geographic resolution compared to twp, but may not be as strong of a proxy for as twp values for other factors such population density and demographics.
   - Also zip codes could cross county political boundaries
   - Decision is to drop zip in favor of twp as a geographic identifier.

## Exploratory Data Analysis (EDA)

- All data is categorical
- Create a Cramer's V Correlation Matrix to investigate relationship among categories.
- Tested groups of features correlation to service_type and service_desc.
- The combination of month, day_of_week, night_day, and twp had the strongest correlation.

**Final Pre-Processing**

- Removed road_type and twp_type
- One-hot encoded month, day_of_week, night_day, and twp

![alt text](<images/categorical correlation - cramers v.png>)

#########################################
GPT notes

Great question! Here’s a **high-level overview** of the general steps involved in a **Machine Learning (ML) project**, from start to finish:

---

### 1. **Problem Definition**

- **Understand the business or research problem.**
- Define the **objective**: What are you trying to predict or classify?
- Understand the **constraints** and **success criteria** (accuracy, speed, cost, etc.).

---

### 2. **Data Collection**

- Gather relevant data from various sources: databases, APIs, web scraping, sensors, etc.
- Consider the **quality and quantity** of data needed for the problem.

---

### 3. **Data Preprocessing & Cleaning**

- Handle **missing data**, **inconsistent formats**, and **outliers**.
- **Feature engineering**: Create new features, encode categorical variables, normalize/scale features.
- Split data into **training, validation, and test sets**.

---

### 4. **Exploratory Data Analysis (EDA)**

- Analyze data distributions, correlations, and patterns.
- Visualize data (e.g., histograms, scatter plots, heatmaps).
- Understand relationships and potential biases in the dataset.

---

### 5. **Model Selection**

- Choose appropriate ML algorithms (e.g., linear regression, decision trees, neural networks).
- Consider baseline models for comparison.

---

### 6. **Model Training**

- Train selected models on the **training dataset**.
- Tune **hyperparameters** using techniques like grid search or random search.
- Monitor training for **overfitting/underfitting**.

---

### 7. **Model Evaluation**

- Test model performance on **validation and test datasets**.
- Use relevant metrics (e.g., accuracy, precision, recall, F1-score, RMSE).
- Analyze **confusion matrix**, ROC curves if applicable.

---

### 8. **Model Optimization**

- Improve performance via:
  - **Feature selection/engineering**.
  - Hyperparameter tuning.
  - Trying different algorithms or architectures (for deep learning).
- Address **bias-variance tradeoff**.

---

### 9. **Deployment**

- Integrate the model into a production environment (API, web app, embedded system, etc.).
- Ensure **scalability** and **reliability**.
- Monitor for **concept drift** and model performance over time.

---

### 10. **Monitoring & Maintenance**

- Track performance using real-world data.
- Set up alerts if performance degrades.
- Retrain or update the model as needed (continuous learning pipelines).

---

### Optional (but important):

- **Documentation & Reporting**: Keep clear records of data sources, modeling decisions, and results.
- **Explainability**: Use tools like SHAP or LIME to explain model decisions.
- **Ethics & Fairness**: Ensure no biases or unintended consequences.

---

If you want, I can give an **example walkthrough** of these steps with a real-world scenario (like predicting house prices or spam detection). Let me know!
