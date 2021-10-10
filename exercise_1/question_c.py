from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from question_b import df, df_1, df_2, df_3, df_4, df_5

# X is the dataset that contains the required features (all columns except stroke, that will need to be predicted)
X = df_5.drop('stroke', axis=1)

# y is a vector that contains only the stroke data (that we want to predict essentially)
y = df_5['stroke']

# split training-testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

forest_cls = RandomForestClassifier(max_features='sqrt', n_estimators=161)

# gia veltiwsh tha prepei grid search + standard h minmax scaler(den beltiwnei kati telika) !
'''
#setting max_depth to any number(default is none) worsens alot f1 score
parameters = {
    'n_estimators': list(range(1,200,20)),
    'max_features': ['auto', 'sqrt', 'log2'],
}

forest_cls = GridSearchCV(forest_cls, parameters, cv= 5)
'''

# fit the train dataset (fitting is equal to training)
forest_cls.fit(X_train, y_train)
 
# make predictions  with a the predict method call 
y_pred = forest_cls.predict(X_test)

# evaluate the accuracy of the classification
# parameters are y_true (y_test as they are from the orignial dataset) and what we predicted (y_pred)
matrix = confusion_matrix(y_test,y_pred)

print(matrix)

# get a report showing the main classification metrics
# parameters are y_true (y_test as they are from the orignial dataset) and what we predicted (y_pred)
print(classification_report(y_test, y_pred))

# for the grid searchCV
# print(forest_cls.best_params_)
