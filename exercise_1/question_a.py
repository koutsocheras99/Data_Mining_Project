import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dataset = 'healthcare-dataset-stroke-data/healthcare-dataset-stroke-data.csv'

# original dataset
df = pd.read_csv(dataset)

# get a general describtion of the dataset columns types
print (df.describe())

# find out which columns have nan values
print (df.isna().sum())

# drop nan values
# df = df.dropna()


# age distribution and mean value #

'''
plt.style.use('classic')
# create the graph which shows the age distribution using:
# https://seaborn.pydata.org/generated/seaborn.distplot.html
sns.distplot(df['age'], color='blue', kde=True)
# limits of the y axes which will point the mean value below
min_ylim, max_ylim = plt.ylim()
# show also also the mean age
plt.text(df['age'].mean()*1.05, max_ylim*0.95, 'Mean Age value: {:.2f}'.format(df['age'].mean()))
# labels and show
plt.xlabel('Age (in years)')
plt.title('Distribution of Ages')
plt.show()
'''

# bmi distribution and mean value #

'''
plt.style.use('classic')
# create the graph which shows the bmi distribution using:
# https://seaborn.pydata.org/generated/seaborn.distplot.html
sns.distplot(df['bmi'], color='blue', kde=True)
# limits of the y axes which will point the mean value below
min_ylim, max_ylim = plt.ylim()
# show also also the mean age
plt.text(df['bmi'].mean()*1.05, max_ylim*0.95, 'Mean BMI value: {:.2f}'.format(df['bmi'].mean()))
# labels and show
plt.xlabel('BMI')
plt.title('Distribution of BMI')
plt.show()
'''

# gender statistics #

'''
# get the exact values
print(df.gender.value_counts())
sns.set_theme(style='darkgrid')
ax = sns.countplot(data=df, x='gender')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.tight_layout()
plt.show()
'''

# hypertension statistics #

'''
# get the exact values
print(df.hypertension.value_counts())
sns.set_theme(style='darkgrid')
ax = sns.countplot(data=df, x='hypertension')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.tight_layout()
plt.show()
'''

# work-type statistics #

'''
# get the exact values
print(df.work_type.value_counts())
sns.set_theme(style='darkgrid')
ax = sns.countplot(data=df, x='work_type')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.tight_layout()
plt.show()
'''

# marriage statistics #

'''
# get the exact values
print(df.ever_married.value_counts())
sns.set_theme(style='darkgrid')
ax = sns.countplot(data=df, x='ever_married')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.tight_layout()
plt.show()
'''

# residence type statistics #

'''
# get the exact values
print(df.Residence_type.value_counts())
sns.set_theme(style='darkgrid')
ax = sns.countplot(data=df, x='Residence_type')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.tight_layout()
plt.show()
'''

# smoke type statistics #

'''
# get the exact values
print(df.smoking_status.value_counts())
sns.set_theme(style='darkgrid')
ax = sns.countplot(data=df, x='smoking_status')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.tight_layout()
plt.show()
'''

# stroke statistics #

'''
# get the exact values
print(df.stroke.value_counts())
sns.set_theme(style='darkgrid')
ax = sns.countplot(data=df, x='stroke')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.tight_layout()
plt.show()
'''
