import pandas as pd
import numpy as np
import sys
import yaml
import os
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))['preprocess']

# Fill missing value in a column based on 
# a reference column. If two rows share the same
# values on the refrence column, fill the missing 
# value of one row with of the other row.
def fill_na(df, col, ref):
        
    index = df[df[col].isna()].index
    for idx in index:
        rf = df.loc[idx, ref]
        df.loc[idx, col] = df[df[ref] == rf][col].mean()
        
    return df
                
def multiple_fill_na(df, cols, ref):
    for col in cols:
        df = fill_na(df, col, ref)
        
    return df


# Treat outliers on one column
def treat_outliers(df, col):
    """
    treats outliers in a variable
    col: str, name of the numerical variable
    """
    Q1 = df[col].quantile(0.25)  # 25th quantile
    Q3 = df[col].quantile(0.75)  # 75th quantile
    IQR = Q3 - Q1
    Lower_Whisker = Q1 - 1.5 * IQR
    Upper_Whisker = Q3 + 1.5 * IQR

    # all the values smaller than Lower_Whisker will be assigned the value of Lower_Whisker
    # all the values greater than Upper_Whisker will be assigned the value of Upper_Whisker
    df[col] = np.clip(df[col], Lower_Whisker, Upper_Whisker)

    return df

# Treat outliers on a list of columns
def treat_outliers_all(df, col_list):
    """
    treat outlier in all numerical variables
    col_list: list of numerical variables
    """
    for c in col_list:
        df = treat_outliers(df, c)

    return df


def preprocessor(input_path, train_output_path, test_output_path):
    data = pd.read_csv(input_path)

    # Drop unecessary columns and write back the data
    data.drop(columns=['CustomerID', 'NumberOfChildrenVisiting', 'OwnCar', 'Gender', 'Passport', 'TypeofContact'], inplace=True)
    
    # Fix missing values
    data['DurationOfPitch'].fillna(value=data['DurationOfPitch'].median(), inplace = True)
    data['NumberOfTrips'].fillna(value=data['NumberOfTrips'].median(), inplace = True)
    data['NumberOfFollowups'].fillna(value=data['NumberOfFollowups'].median(), inplace = True)

    data = multiple_fill_na(data, ['Age', 'MonthlyIncome'], 'Designation')

    # Treat outliers
    data = treat_outliers_all(data, ['MonthlyIncome', 'NumberOfTrips', 'DurationOfPitch'])

    # Encode string columns to categories
    data.replace({
        "ProductPitched": {'Basic': 0, 'Standard': 1, 'Deluxe': 2, 'Super Deluxe': 3, 'King': 4},
        "Occupation": {'Free Lancer': 0, 'Salaried': 1, 'Small Business': 2, 'Large Business': 3},
        "Designation": {'Manager': 0, 'Senior Manager': 1, 'Executive': 2, 'AVP': 3, 'VP': 4},
        "MaritalStatus": {'Single': 0, 'Married': 1, 'Divorced': 2, 'Unmarried': 3},
        "PreferredPropertyStar": {'Unknown': 0}
                }, inplace=True)
    
    # Fix column dtype
    cat_columns = ['ProdTaken', 'CityTier', 'Occupation', 'ProductPitched', 'PreferredPropertyStar', 
                'MaritalStatus', 'PitchSatisfactionScore', 'Designation']
    data[cat_columns] = data[cat_columns].astype('category')

    # Split the data
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # Save preprocessed data
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    train.to_csv(train_output_path)
    print(f"Preprocessed training data saved to {train_output_path}")

    os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
    test.to_csv(test_output_path)
    print(f"Preprocessed test data saved to {test_output_path}")

if __name__ == "__main__":
    preprocessor(params["input"], params['output']['train'], params['output']['test'])