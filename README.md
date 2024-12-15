# AI bootcamp project 2 : Stroke survival prediction
### What are the potential outcomes of someone experiencing an acute stroke at the end of a 6-month period?
* What effect this mortality rate?
* How can we use factors to predict potential outcome?

# Table of content
1. [Introduction](https://github.com/T-Haight/AI_bootcamp_proj2/tree/Asmaa?tab=readme-ov-file#introduction)
2. [Objective](https://github.com/T-Haight/AI_bootcamp_proj2/edit/Asmaa/README.md#objective)
3. [Installation](https://github.com/T-Haight/AI_bootcamp_proj2/edit/Asmaa/README.md#installation)
4. [Dataset](https://github.com/T-Haight/AI_bootcamp_proj2/edit/Asmaa/README.md#dataset)

## Introduction

Stroke carries a high risk of death. Survivors can experience loss of vision and/or speech, paralysis and 
confusion. Stroke is so called because of the way it strikes people down. The risk of further episodes is 
significantly increased for people having experienced a previous stroke. The risk of death depends on the 
type of stroke, effective treatement planning, resource allocation, and improving patient care. With our 
interest in healthcare and parents aging into a new decade, we chose this Stroke Dataset for our AI project. 

## Objective 

Through this project, we will predict the likelihood of a stroke patient surviving or dying within 6 months 
using machine learning models in order to bring awareness to individuals and help healthcare providers and 
insurers estimate risk and cost.

## Installation
### The requirement:
To run this project you need to install python 3.8 or higher,also you need to instal the packages and libraries bellow:
```python
pip install pandas
```
```python
pip install numpy
```
```python
pip install matplotlib
```
```python
pip install -U scikit-learn
```
```python
pip install imblearn
```
```python
pip install pipeline
```
```python
pip install seaborn
```
### Steps to run the program 
1. Clone the repository.
2. Navigate to the project directory.
3. Run the jupyter notebook file (with VS Code or colab)

## Dataset

1. Dataset description:
  The International Stroke Trial (IST) was conducted between 1991 and 1996 (including the pilot phase between 
1991 and 1993). It was a large, prospective, randomised controlled trial, with 100% complete baseline
data and over 99% complete follow-up data.For the purposes of our analysis, we carefully selected only the 
features from the dataset that were revelant to our objectives, by focusing on these essential columns we 
avoid unnecessary complexity of the analysis.

2. Source : [open the link here](http://www.trialsjournal.com/content/12/1/101)

3. Features:
   * Before the data clean and selection: [Columns names and comments](https://trialsjournal.biomedcentral.com/articles/10.1186/1745-6215-12-101/tables/2)
   * After the data clean and selection:

4. Size:
   * Before the data clean and selection: 19435 rows × 112 columns
   * After the data clean and selection: 9405 rows × 41 columns 

5. Data Cleaning:
   * Missing values were handled by `fillna()` function, filling the NaN values with {U (unknown), C (can't assess)} for non numerival values and {(0)} for numerical values.
   * Categorical variables were encoded using `LabelEncoder()` method.
   * Droping rows with NAN values and also droping unnecessary columns.
   * Selecting only the important features for our Algorithm.

6. License: 
   * The authors ask that any publications arising from the use of this dataset acknowledges 
   the source of the dataset, its funding and the the collaborative group that collected the data.

## Model Selection and Training

* Generate models for the dataset and find the best one. 
* Models tested : 
  * Logistic regression
  * SVC algorithm
  * K-Neighbors Classifier
  * Decision tree classifier
  * Random Forest Classifier
  * Extra Trees Classifier
  * Gradient Boosting Classifier
  * AdaBoost Classifier
* Model selected:
  * Gradient Boosting Classifier: it has the best Accuracy Score 87.1% . 

## Model Evaluation

* *Good performance for Survival Prediction*, the model is performing well in predicting survival(class 1) with high precision, recall and F1-score. This means it's generally accurate in identifiying patients who will survive.
* *Lower perforance for Death Prediction*, the model is less accurate in predicting death (class 0), particularly in terms of recall. This indicates that it's missing a significant portion of patients who will actully die within 6 months.

## Primary Results:

After a few adjustments, the result that we received through the Gradient Boosting Classifier was an 87.1% accuracy score. This implies that we are able to predict future outcomes of patients in a 6-month time period utilizing factors recorded within the first two weeks of an acute stroke.

## Obstacles and Future Work:

* Continious correction of the data and figures (Imbalanced data)
* Hyperparameter Scoring (score was dropped).
* Two rounds of cleanup (Removing more columns and redundancy)
* Ultimate Result: 87.3%
* The model is reasonably good at predicting stroke survival but struggles more with predicting death more likely due to class imbalanced in the data. if we had more time we would use resampling techniques.

