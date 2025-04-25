# Purpose of this Folder

This folder should contain the scaffolded project files to get a student started on their project. This repo will be added to the Classroom for students to use, so please do not have any solutions in this folder.


### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

This project can be run using Python version 3. Ensure the following packages are installed before running the code:

1. pandas
2. numpy
3. Spacy: ! python -m spacy download en_core_web_sm
4. scikit-learn


## Project Motivation<a name="motivation"></a>



## File Descriptions <a name="files"></a>

This project consists of two main Jupyter Notebooks and a README file that showcase the analysis of the CMS HRRP (Hospital Readmissions Reduction Program) dataset. The notebooks focus on answering key business questions about predicting and classifying hospital readmissions. Here are the details of the files:

1. Regression_model.ipynb: This notebook covers the regression task, which predicts the Number of Readmissions at CMS hospitals. The notebook explores the correlation between features and the target variable, applies feature selection techniques, and builds regression models like Random Forest Regression, Linear Regression, and Decision Tree Regression. The goal is to identify the most important features and accurately predict the number of readmissions based on hospital characteristics.

2. Classification_model.ipynb: This notebook focuses on building a classification model to predict which hospitals are at risk of high readmission volumes, which could lead to penalties. It involves the creation of a target variable for high readmission volume and the implementation of classification models such as XGBoost and Random Forest. Key steps include feature selection, model training, and evaluation, with the aim to classify hospitals that are likely to face penalties due to excessive readmissions.

3. README.md: This file, which you are currently reading, provides an overview of the project, the business questions addressed, the data understanding and preparation steps, modeling techniques used, and the evaluation of both the regression and classification models.

4. Dataset_FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv: Contains the dataset used in this project [here](https://data.cms.gov/provider-data/dataset/9n3s-kdb3#data-table).
   
## Results<a name="results"></a>

The final 
Accuracy: 0.8482384823848238
Precision: 0.8501131221719457
Recall: 0.9901185770750988
F1 Score: 0.9147900182592819

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

[License](LICENSE.txt)

