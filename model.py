import os
os.system('pip install -r requirements.txt')

# Standard 
import pandas as pd
import numpy as np

# Models 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Our modules 
from helper_methods import *

# Locate and access the data
def load_and_merge_the_data_sets(store_data_path='./data/store.csv', data_path='./data/train.csv'):
    # Read in store data
    df_store = pd.read_csv(store_data_path, low_memory=False)

    # Read in store data
    df_train = pd.read_csv(data_path, low_memory=False)
    
    # merge the datasets
    df_full = pd.merge(df_train, df_store, on=["Store"])
    
    return df_full

# Clean the data
def clean_data(df):
    
    # DROP COLUMNS
    #     Date - Drop date because we have this information encoded in DaysOfWeek
    #     CompetitionOpenSinceMonth, CompetitionOpenSinceYear - We think this information was encoded in the CompetitionDistance
    #     Promo2SinceWeek, Promo2SinceYear, PromoInterval - Large percentage of nulls (47%) so not worth cutting all those rows or imputing
    columns_to_remove = [
        "Date", 
        "CompetitionOpenSinceMonth", 
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval"
    ]
    df = df.drop(columns_to_remove, axis=1)
    
    # DROP ROWS
    #     Sales - Cannot predict rows where there are null sales
    #     Store - drop the nulls because we can't impute this
    #     DayOfWeek - drop nulls because it's a small % of the dataset
    #     Promo - Drop nulls for promo as they are a small percentage of the dataset
    df = df.dropna(subset=["Sales", "Store", "DayOfWeek", "Promo"])
    
    # Open - Drop closed days since sales should be 0 on these days
    #      - Remove Open column aftwards as it now provides no more information
    df = df.loc[df.loc[:, 'Open'] == 1]
    df = df.drop('Open', axis=1)
    
    # Sales - Remove rows where sales are 0 as this breaks the RMSPE score
    df = df.loc[df.loc[:, "Sales"] != 0]
    
    # IMPUTE DATA
    # CompetitionDistance - Fill with the average
    mean_competition_distance = df.loc[:, 'CompetitionDistance'].mean()
    df.loc[:, "CompetitionDistance"] = df.loc[:, "CompetitionDistance"].fillna(mean_competition_distance)
    
    """LOOK INTO THIS AS THIS IS ACTUALLY A BIT OF FEATURE ENGINEERING"""
    # Customers - Fill with the average number of customers per store
    #           - We do not know this information ahead of time so we need to use historical averages
    #           - Drop the old Customers column after we add the new average customers per store column
    mean_customers = df.groupby('Store')['Customers'].transform('mean').astype('int')
    df.loc[:, 'average_customers_per_store'] = mean_customers
    df = df.drop(["Customers"], axis=1)
    
    return df

# Encode the data
def encode_data(df):
    
    # Dummy Variables
    #     StateHoliday
    #     SchoolHoliday
    #     DayOfWeek
    #     StoreType
    #     Assortment
    dummy_columns = [
        'StateHoliday',
        'SchoolHoliday',
        'DayOfWeek',
        'StoreType',
        'Assortment'
    ]
    dummy_column_names = [
        'public_holiday',
        'easter_holiday',
        'christmas',
        'not_school_holiday',
        'school_holiday',
        'monday',
        'tuesday',
        'wednesday',
        'thursday',
        'friday',
        'saturday',
        'sunday',
        'store_model_1',
        'store_model_2',
        'store_model_3',
        'store_model_4',
        'basic',
        'extra',
        'extended'
    ]
    df_columns = [
        'Store',
        'Sales',
        'Promo',
        'CompetitionDistance',
        'Promo2',
        'average_customers_per_store'
    ]
    new_df_columns = df_columns + dummy_column_names
    df = pd.get_dummies(data=df, columns=dummy_columns)
    df = df.drop('StateHoliday_0', axis=1)
    df.columns = new_df_columns
    
    # MEAN ENCODING
    # STORE - add average sales per store
    #       - remove Store column as it is now redundant
    df.loc[:, 'average_sales_per_store'] = df.groupby('Store')['Sales'].transform('mean')
    df = df.drop('Store', axis=1)
    
    # Finally reset the index after the data has been entirely transformed
    df = df.reset_index(drop=True)
    
    return df

# Get Data Functions
def get_data(store_data_path='./data/store.csv', data_path='./data/train.csv'):
    
    # Load and merge the datasets 
    df = load_and_merge_the_data_sets(
        store_data_path=store_data_path, 
        data_path=data_path
    )
    
    # Clean the data
    df_cleaned = clean_data(df)
    
    # Encode the data
    df_encoded = encode_data(df_cleaned)
    
    return df_encoded

def get_x_and_y_datasets(store_data_path='./data/store.csv', data_path='./data/train.csv'):
    
    # get the data
    df = get_data(
        store_data_path=store_data_path,
        data_path=data_path
    )
    
    # split into X and y 
    X = df.loc[:, df.columns.difference(['Sales'])]
    y = df.loc[:, "Sales"]

    return [X, y]

# Machine Learning Model Functions
def build_and_run_mean_regressor(y_train, y_test):
    # broadcast the mean predictions 
    mean_predictions = [y_train.mean()]
    mean_predictions = np.array(mean_predictions * y_test.shape[0])
    
    return metric(mean_predictions, y_test.to_numpy())

def build_and_run_linear_regression(X_train, y_train, X_test, y_test):
    regressor = LinearRegression().fit(X_train, y_train)
    linear_regression_predictions = regressor.predict(X_test)
    
    return metric(linear_regression_predictions, y_test.to_numpy())

def build_and_run_random_forest(X_train, y_train, X_test, y_test):
    regressor_random_forest = RandomForestRegressor(n_estimators=40, min_samples_leaf=2, max_features=0.99, n_jobs=-1,oob_score=True)
    regressor_random_forest.fit(X_train, y_train)
    random_forest_predictions = regressor_random_forest.predict(X_test)
    
    return metric(random_forest_predictions, y_test.to_numpy())

def build_and_run_xgboost(X_train, y_train, X_test, y_test):
    regressor_xgboost = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    regressor_xgboost.fit(X_train, y_train)
    xgboost_predictions = regressor_xgboost.predict(X_test)
    
    return metric(xgboost_predictions, y_test.to_numpy())

# Results Functions
def get_model_results(X_train, y_train, X_test, y_test):
    
    print()
    print("TRAIN AND EVALUATE THE MODELS")
    print("Training and evaluating the Mean Regressor model")
    mean_regressor_metric = build_and_run_mean_regressor(y_train, y_test)

    print("Training and evaluating the Linear Regression model")
    linear_regression_metric = build_and_run_linear_regression(X_train, y_train, X_test, y_test)

    print("Training and evaluating the Random Forest model")
    random_forest_metric = build_and_run_random_forest(X_train, y_train, X_test, y_test)

    print("Training and evaluating the XGBoost model")
    xgboost_metric = build_and_run_xgboost(X_train, y_train, X_test, y_test)
    
    return { 
        "Mean Regressor": mean_regressor_metric, 
        "Linear Regression": linear_regression_metric, 
        "Random Forest": random_forest_metric, 
        "XGBoost": xgboost_metric
    }

def print_model_results(results):
    print()
    print("RESULTS")
    for model, result in results.items():
        percentage_result = round(result, 2)
        print(f"The RMSPE for the {model} model was {percentage_result}%")

def build_models_and_print_results(test_data_path, store_data_path='./data/store.csv', train_data_path='./data/train.csv'):
    
    # load the train dataset
    print("LOAD THE DATASETS")
    print("Loading the train dataset")
    X_train, y_train = get_x_and_y_datasets(store_data_path=store_data_path, data_path=train_data_path)
    
    # load the test dataset
    print("Loading the test dataset")
    X_test, y_test = get_x_and_y_datasets(store_data_path=store_data_path, data_path=test_data_path)
    
    # get model results
    model_results = get_model_results(X_train, y_train, X_test, y_test)
    
    # print model results
    print_model_results(model_results)