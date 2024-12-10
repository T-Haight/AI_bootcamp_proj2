from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, RidgeCV, LinearRegression, SGDRegressor
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def drop_top_vif(df):
    # Calculate the VIF for each column in the dataframe
    vif = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    # Get the maximum VIF value (ignore nan values)
    max_vif = np.max([vif_value for vif_value in vif if not pd.isna(vif_value)])
    # Get the index of the maximum VIF value
    max_index = vif.index(max_vif)

    return df.drop(columns=df.columns[max_index], inplace=True)


def preprocess_data(df, target_column):
    """
    Accepts a dataframe and target column, then returns the 
    data split into training and testing sets.
    """
    # Drop the target column from the dataframe
    X = df.drop(columns=target_column)
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return X_train, X_test, y_train, y_test


def r2_adj(x, y, model):
    """
    Calculates adjusted r-squared values given an X variable, 
    predicted y values, and the model used for the predictions.
    """
    r2 = model.score(x,y)
    n_cols = x.shape[1]
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_cols - 1)

def check_metrics(X_test, y_test, model):
    """
    Checks various accuracy and error metrics of a model.
    """
    # Use the pipeline to make predictions
    y_pred = model.predict(X_test)

    # Print out the MSE, r-squared, accuracy, and cross val values
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared: {r2_adj(X_test, y_test, model)}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Cross Val Score: {cross_val_score(model, X_test, y_test, scoring='r2').mean()}\n")

    return r2_adj(X_test, y_test, model)

def get_best_pipeline(pipelines, df, target_column):
    """
    Accepts a list of pipelines and data.
    Splits the data for training the different 
    pipelines, then evaluates which pipeline performs
    best.
    """
    # Apply the preprocess_rent_data step
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column)

    # Train and score each pipeline to find the best one
    best_pipeline = None
    best_score = 0  
    for pipeline in pipelines:
        print(f"{pipeline.named_steps['clf']}")
        pipeline.fit(X_train, y_train)
        score = check_metrics(X_test, y_test, pipeline)
        if score > best_score:
            best_score = score
            best_pipeline = pipeline
    print(f"Best pipeline: {best_pipeline.named_steps['clf']}\nR-squared: {best_score}")
    return best_pipeline

def stroke_model_generator(df, target_column):
    """
    Accepts a dataframe and returns a list of pipelines
    to be used for training.
    """
    pipelines = []
    pipelines.append(Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ]))
    pipelines.append(Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC())
    ]))
    pipelines.append(Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier())
    ]))
    pipelines.append(Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier())
    ]))
    pipelines.append(Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier())
    ]))
    pipelines.append(Pipeline([
        ('scaler', StandardScaler()),
        ('clf', ExtraTreesClassifier())
    ]))
    pipelines.append(Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier())
    ]))
    pipelines.append(Pipeline([
        ('scaler', StandardScaler()),
        ('clf', AdaBoostClassifier(algorithm='SAMME'))
    ]))

    return get_best_pipeline(pipelines, df, target_column)