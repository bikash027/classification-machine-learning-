# Split a dataset based on an attribute and an attribute value
def test_split_num(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
def test_split_cat(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] == value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a dataset

def get_split(dataset,num_feat,cat_feat):
    class_values=list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    twos_train=[row[-1] for row in dataset].count(2)
    for index in num_feat:
        uniq=list(set(row[index] for row in dataset))
        for val in uniq:
            groups=test_split_num(index,val,dataset)
            gini=gini_index(groups, class_values)
            if gini<b_score:
                b_index, b_value, b_score, b_groups = index, val, gini, groups
    for index in cat_feat:
        for val in range(2):
            groups=test_split_cat(index,val,dataset)
            gini=gini_index(groups, class_values)
            if gini<b_score:
                b_index, b_value, b_score, b_groups = index, val, gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups,'leaf':False,'total':0,'twos':0,'total_train':len(dataset),'twos_train':twos_train}
                
# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return {'label':max(set(outcomes), key=outcomes.count),'leaf':True,'total':0,'twos':0}

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth,num_feat,cat_feat):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left,num_feat,cat_feat)
		split(node['left'], max_depth, min_size, depth+1,num_feat,cat_feat)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right,num_feat,cat_feat)
		split(node['right'], max_depth, min_size, depth+1,num_feat,cat_feat)

# Build a decision tree
def build_tree(train, max_depth, min_size,num_feat,cat_feat):
	root = get_split(train,num_feat,cat_feat)
	split(root, max_depth, min_size, 1,num_feat,cat_feat)
	return root

# Print a decision tree
"""def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))"""

dataset = [[2.771244718,1.784783929,1,0],
	[1.728571309,1.169761413,1,0],
	[3.678319846,2.81281357,1,0],
	[3.961043357,2.61995032,1,0],
	[2.999208922,2.209014212,0,0],
	[7.497545867,3.162953546,0,1],
	[9.00220326,3.339047188,0,1],
	[7.444542326,0.476683375,0,1],
	[10.12493903,3.234550982,0,1],
	[6.642287351,3.319983761,0,1]]

def predict(node,row,num_feat,cat_feat):
    if node['leaf']:
        return node['label']
    if node['index'] in cat_feat:
        if row[node['index']]==node['value']:
            return predict(node['left'],row,num_feat,cat_feat)
        else:
            return predict(node['right'],row,num_feat,cat_feat)
    if node['index'] in num_feat:
        if row[node['index']]<node['value']:
            return predict(node['left'],row,num_feat,cat_feat)
        else:
            return predict(node['right'],row,num_feat,cat_feat)
def predict_all(x_test,tree,num_feat,cat_feat):
    ans=list()
    for row in x_test:
        ans.append(predict(tree,row,num_feat,cat_feat))
    return ans    

#following functions are required for pruning the tree
def classify(T,row,num_feat,cat_feat):
    T['total']+=1
    if row[-1]==2:
        T['twos']+=1
    if T['leaf']==False:
        if T['index'] in cat_feat:
            if row[T['index']]==T['value']:
                T['left']=classify(T['left'],row,num_feat,cat_feat)
            else:
                T['right']=classify(T['right'],row,num_feat,cat_feat)
        if T['index'] in num_feat:
            if row[T['index']]<T['value']:
                T['left']=classify(T['left'],row,num_feat,cat_feat)
            else:
                T['right']=classify(T['right'],row,num_feat,cat_feat)
    return T
def prune(T):
    if T['leaf']==True:
        if T['label']==2:
            return {'val':T['total']-T['twos'],'subt':T}
        else:
            return {'val':T['twos'],'subt':T}
    else:
        l=prune(T['left'])
        r=prune(T['right'])
        T['left']=l['subt']
        T['right']=r['subt']
        error=l['val']+r['val']
        if error<min(T['twos'],T['total']-T['twos']):
            return {'val':error,'subt':T}
        else:
            temp=T['total']-T['twos']
            if T['twos']>temp:
                T={'label': 2,'leaf': True,'twos':T['twos'],'total':T['total']}
                return {'val':T['total']-T['twos'],'subt':T}
            else:
                temp=T['twos_train']
                T={'label': 1,'leaf': True,'twos':T['twos'],'total':T['total']}
                return {'val':T['twos'],'subt':T}
def rep(T,S,num_feat,cat_feat):
    for i in range(len(S)):
        classify(T,S[i],num_feat,cat_feat)
    tr=prune(T)
    return tr['subt']







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
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data=data[to_keep]
X = data.loc[:, data.columns != 'default']
y = data.loc[:, data.columns == 'default']
data=X.join(y)

X = data.iloc[:, :-1].values
y = data.iloc[:, 61].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 0)
train_data = np.zeros(shape=(800,62))
for i in range(800):
    for j in range(61):
        train_data[i][j]=X_train[i][j]
    train_data[i][61]=y_train[i]

num_feat=[0,1,2,3,4,5,6]
cat_feat=[]
for i in range(54):
    cat_feat.append(i+7)
train_data,val_data=train_test_split(train_data,test_size=0.31,random_state=0)

tree = build_tree(train_data, 10, 1,num_feat,cat_feat)

rep(tree,val_data,num_feat,cat_feat)
y_pred=predict_all(X_test,tree,num_feat,cat_feat)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
#print(confusion_matrix)
correct=confusion_matrix[0][0]+confusion_matrix[1][1]
Total=confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1]
print("accuracy: %f" % ((correct*100/Total)))