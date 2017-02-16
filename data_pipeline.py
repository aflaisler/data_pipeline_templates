import pandas as pd
import numpy as np
from zipfile import ZipFile
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from pandas_summary import DataFrameSummary
from pandas.tools.plotting import scatter_matrix


def load_data(filename):
    '''
    - input: zipfile location containing the data
    - output: pandas dataframe with subset of data with the following features:
        MachineID
        ModelID
        datasource
        YearMade
        Saleprice
        fiBaseModel
        fiSecondaryDesc
        State
        ProductGroup
        Enclosure
        Ride_Control
        Stick
        Hydraulics
    - how it works: unzip the data, load data from zip file to a pandas df,
    select the features previously identify during EDA, remove missing data and outliers

    - DtypeWarning: Columns (13,39,40,41) have mixed types but we won't use them
    '''
    zf = ZipFile(filename + '.zip')
    df = pd.read_csv(filename + '.csv')
    # selecting specific features we will work with
    keep_cols = ["MachineID", "ModelID", "datasource", "YearMade", "MachineHoursCurrentMeter", "SalePrice",
                 "fiBaseModel", "fiSecondaryDesc", "state", "ProductGroup", "Enclosure", "Ride_Control", "Stick", "Hydraulics"]
    df = df[keep_cols]
    # removing weird year - will focus on those later on if we have time
    df_out = df.ix[df['YearMade'] > 1900].reset_index(drop=True)
    # removing nan hours
    df_out = df_out.ix[
        np.isnan(df_out['MachineHoursCurrentMeter']) == False]
    df_out = df_out.reset_index(drop=True)
    # save df with missing year
    df_year_missing = df.ix[df['YearMade'] <= 1900].reset_index(drop=True)
    # save df with missing hours
    df_machineHour_missing = df.ix[np.isnan(
        df['MachineHoursCurrentMeter'].values)].reset_index(drop=True)
    return df_out, df_year_missing, df_machineHour_missing


def missing_data_imputation(df_train, df_with_missing_data, column_to_fill, predictors=['YearMade', 'SalePrice']):
    '''
    - input: pandas dataframe x2 , str, lst
    - output: pandas dataframe
    - how it works: fill the df with missing data using RF trained on the df with
    data
    '''
    y = df_train.ix[:, column_to_fill].values
    X = np.array(df_train.loc[:, predictors])
    X_to_predict = df_with_missing_data.loc[:, predictors]
    # train a random forest to fill hours
    md = RandomForestRegressor()
    md.fit(X, y)
    predicted_data = md.predict(X_to_predict)
    y_pred = pd.DataFrame(predicted_data, columns={column_to_fill})

    # merge the data
    df_out = pd.concat((df_with_missing_data.drop(
        column_to_to, 1), y_pred), 1)

    return df_out


if __name__ == '__main__':
    # Don't add the extension to the filename as it will extract zip to csv
    filename = '../data/Train'
    df, df_year_missing, df_machineHour_missing = load_data(filename)
    # fill missing hours data
    df_hours = missing_data_imputation(
        df, df_machineHour_missing, column_to_fill='MachineHoursCurrentMeter', predictors=['YearMade', 'SalePrice'])
    # final date with prediction of the hours
    df_final = pd.merge(df, df_hours)
