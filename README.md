# Demonstration ML Project

This project demonstrates the workflows and technologies common to many machine learning (ML) projects. The focus is on stardard ML, as opposed to deep learning which uses neural networks. It is inspired by the popular ["Emergency - 911 Calls"](https://www.kaggle.com/datasets/mchirico/montcoaler) dataset hosted on Kaggle.

### Simualted 911 Scenario

Local officials in Montgomery County, PA have approved funds to purchase additional vehicals and hire additional staff to be placed at existing facilities. It is assumed that a new fire truck, police cruiser, and ambulance (one of each) will be purchased for each participating township.

A regional planning commission (RPC) has been tasked with providing a data-driven report containing information that can help the county, township, and municiple officials decide how to distribute the vehicles among the members and for which work shifts to hire new staff. 

### Subject Matter Expertise
As a ML demonstration project, there is no access to a true subject matter expert (SME) in 911 response systems or programs. ML projects create the most value when ML engineers and SMEs collaborate together to create relevant and actionable information products. Without the participation of a 911 SME, the outputs of this project will not be relevant to real-world scenarios. However, the project will demonstration valid application of workflow, techniques, and tools in the ML space.

### Workflow
The following steps

1. Objectives Definition
2. Data Collection
3. Data Preprocessing & Cleaning
4. Exploratory Data Analysis (EDA)

## 1. Objectives Definition

The RPC has decided to include the following ML predictions in their analysis report.

1. Given a call's time and location, what most likely service required
2. Given a call's service and location, what is the most likely day of week.
3. Given a call's service, location, and day of week, what is the most likely shift

   ### Success Criteria

- The ML models predict the specified class in the training set with 90% confidence and 90% accuracy

*NOTE: This objective definition has been contrived without the advice of a 911 SME.*
