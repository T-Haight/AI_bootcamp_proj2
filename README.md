# AI_bootcamp_proj2
Stroke survival prediction : will the patient live or die within 6 months ?
______________________________________________________________________________________________________________________________

## Introduction

Stroke carries a high risk of death. Survivors can experience loss of vision and/or speech, paralysis and 
confusion. Stroke is so called because of the way it strikes people down. The risk of further episodes is 
significantly increased for people having experienced a previous stroke. The risk of death depends on the 
type of stroke, effective treatement planning, resource allocation, and improving patient care. With our 
interest in healthcare and parents aging into a new decade, we chose this Stroke Dataset for our AI project. 


________________________________________________________________________________________________________________________________

## Objective 

Through this project, we will predict the likelihood of a stroke patient surviving or dying within 6 months 
using machine learning models in order to bring awareness to individuals and help healthcare providers and 
insurers estimate risk and cost.
________________________________________________________________________________________________________________________________

## Dataset

1. Dataset description:
  The International Stroke Trial (IST) was conducted between 1991 and 1996 (including the pilot phase between 
1991 and 1993). It was a large, prospective, randomised controlled trial, with 100% complete baseline
data and over 99% complete follow-up data.For the purposes of our analysis, we carefully selected only the 
features from the dataset that were revelant to our objectives, by focusing on these essential columns we 
avoid unnecessary complexity of the analysis.

2. Source : 

3. Features:

4. Size: 19435 rows and 77 columns

5. Data Cleaning:
   Missing values were handled by ...
   Categorical variables were encoded using one hot coding.

6. License

# Note :
Hello team! I have some comments that concern the data and our question, I would like to share it with you , based on what I understood and analyzed there are four types of stroke (final diagnostic) :
  DDIAGISC Ischemic stroke (Y/N/U=unknown)
  DDIAGHA Hemorrhagic stroke(Y/N/U=unknown) 
  DDIAGUN Indeterminate stroke(Y/N/U=unknown) 
  DNOSTRK Not a stroke(Y/N/U=unknown) 

same with Indicator variables for specific causes of death after 6 months , there are 17 different variables.
 
At the end we need one column that represent the y  also we need correlated features.

my plan is to ...