import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

dataset = 'healthcare-dataset-stroke-data/healthcare-dataset-stroke-data.csv'

dataset = 'healthcare-dataset-stroke-data/a.csv'

# original dataset
df = pd.read_csv(dataset)

# drop id column (its unnecessary and doesnt help with the predictions)
df = df.drop(['id'], axis=1)


# encode some categorical-string columns using factorize (for example, for gender column values female->0, male->1)
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# https://pandas.pydata.org/docs/reference/api/pandas.factorize.html
df[categorical_columns] = df[categorical_columns].apply(lambda x: pd.factorize(x)[0])

# print(df)

# replace in the smoking_status column unknown values with nan (after the encoding the unknown value equals to 3, therefore if we find 3 in the smoking_status column set it to nan)
df.loc[df['smoking_status'] == 3, 'smoking_status'] = np.nan

# check in which columns we have nan values
# print(df.isna().sum())

# drop smoke column as it contains unknown values and bmi column due to nan values
df_1 = df.drop(['bmi', 'smoking_status'], axis=1)

# print(df_1)

# drop smoke column as it contains caterogical values so we cant find the mean 
df_2 = df.drop(['smoking_status'], axis=1)

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
# in case of nan values fill with the according mean
df_2 = df_2.fillna(value=df.mean())

# print(df_2)

# drop smoke column as it contains caterogical values so we cant use  linear regression 
df_3 = df.drop(['smoking_status'], axis=1)

# dataframe to series conversion using squeeze() for interpolation below
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.squeeze.html
df_3 = df_3.squeeze()

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
# limit_direction = both to fill both ways not only forward
df_3 = df_3.interpolate(method='linear', limit_direction='both')

# print(df_3)

# drop bmi column as it contains numerical values 
df_4 = df.drop(['bmi'], axis=1)

# print(df_4.isna().sum())

# https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
imputer = KNNImputer(n_neighbors=3)

# apply the knn imputer to the dataset  
df_4 = pd.DataFrame(imputer.fit_transform(df_4), columns = df_4.columns)

# round the results of the smoking status column (more details about this action will be included in the report, also neighbors number is odd for a reason!)
# because we need integers due to it is a classification problem
df_4.smoking_status = df_4.smoking_status.round()

# print(df_4)

df_5 = df

# set the bmi col with the linear found column
df_5['bmi'] = df_3['bmi']

# set the smoking sttus col with the knn found column
df_5['smoking_status'] = df_4['smoking_status']

# print(df_5)


