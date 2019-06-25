
def conv(y):
    for i in range(len(y)):
        if y[i]==2:
            y[i]=0
    return y



def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    return X.iloc[:, variables]

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('credit.csv')


cat_vars=['checking_balance','credit_history','purpose','savings_balance','employment_length','personal_status','other_debtors','property','installment_plan','housing','telephone','foreign_worker','job']
#indices=[]
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
#cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data=data[to_keep]
cols=['amount',
      'checking_balance_< 0 DM',
      'checking_balance_> 200 DM',
      'checking_balance_unknown',
      'credit_history_fully repaid',
      'credit_history_fully repaid this bank',
      'credit_history_repaid',
      'purpose_car (used)',
      'purpose_furniture',
      'purpose_radio/tv',
      'savings_balance_> 1000 DM',
      'savings_balance_unknown',
      'employment_length_4 - 7 yrs',
      'employment_length_> 7 yrs',
      'other_debtors_guarantor',
      'property_real estate',
      'installment_plan_stores',
      'telephone_yes','default']

data=data[cols]
X = data.loc[:, data.columns != 'default']
y = data.loc[:, data.columns == 'default']

from statsmodels.stats.outliers_influence import variance_inflation_factor    
X=calculate_vif_(X)


"""z=conv(y['default'].tolist())
y=pd.DataFrame({'default':z})

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())"""


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
#print(confusion_matrix)

correct=confusion_matrix[0][0]+confusion_matrix[1][1]
Total=confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1]
print("accuracy: %f" % ((correct*100/Total)))