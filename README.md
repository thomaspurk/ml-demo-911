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

## 2. Data Collection

- **Emergency (911) Calls: Fire, Traffic, EMS for Montgomery County, PA**
> Mike Chirico. (2020). Emergency - 911 Calls [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/1381403
