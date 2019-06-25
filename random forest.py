

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('credit.csv')


cat_vars=['checking_balance','credit_history','purpose','savings_balance','employment_length','personal_status','other_debtors','property','installment_plan','housing','telephone','foreign_worker','job']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data=data[to_keep]
X = data.loc[:, data.columns != 'default']
y = data.loc[:, data.columns == 'default']
data=X.join(y)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=9,max_depth=6)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

correct=confusion_matrix[0][0]+confusion_matrix[1][1]
Total=confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1]
print("accuracy: %f" % ((correct*100/Total)))
