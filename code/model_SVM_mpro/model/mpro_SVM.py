#!/usr/local/bin/python3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import math
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import svm,linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel,RFE,RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler,KBinsDiscretizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR,LinearSVR
from sklearn import model_selection
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from numpy import transpose
from collections.abc import Iterable
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance

#fname='./PostEra_data_17_03_all_MD.csv'
#fname='./molecular_descriptor_mpro_maestro_620-out.csv'
#fname='./PostEra_data_17_03_all_MD.csv'
#fname='./PostEra_data_log_17_03_ic50_pic50_filt_2.csv'
#fname='./PostEra_data_log_20_03_FP.csv'
#fname='./PostEra_data_log_20_03_no_duplicates_MD_FP.csv'
#fname = 'prova.csv'
fname='data.csv'

#activity_treshold

'''
class my_SVR(SVR):
    def fit(self, *args, **kwargs):
        super(my_SVR, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_
'''
'''
imputing_missing_values(array)

fills missing values (NaN) with the mean value of the column
'''
def imputing_missing_values(data):
    return SimpleImputer(missing_values=np.NaN, strategy='mean')
'''
write_ft(titles, features)

write in 'ft_selected.txt' file the features from the index list 'features' 
'''
def write_ft(titles,features):
    #print(titles)
    #print(len(titles))
    with open('ft_selected.txt','w') as f:
        for ft in features:
            f.write(titles[ft]+'\n')


    
'''
read_data(fname,target,excluded=[])

read fname file, assign label index in target
target:   int, index of classification result

optionally indicating in excluded list the index feauture to be ignored

In titles are stored the name of the MDs,
In classification are stored the activity data (IC50 values)

returns the np.array representation of file 'fname'
'''
def read_data(fname,target,excluded=[]):
    classification,titles,data = [],[],[]
    
    with open (fname, 'r') as ds:
        titles=ds.readline().strip().split(',')
        target=len(titles)+target if target<0 else target
        titles=[titles[i] for i in range(len(titles)) if i not in excluded and i != target]
        for x in ds:
            line=x.strip().split(',')
            classification.append(float(line[target]))
            data.append([float(line[i]) if len(line[i])>0 else np.NaN for i in range(len(line)) if i not in excluded and i != target ]) 
    return np.array(data),np.array(classification),titles

'''
constant_columns(data)

delete columns having the same values
returns matrix withot constant colums

'''
def constant_columns(X):
    X = X.transpose()
    col_0s = [i1  for i1 in range(len(X)) if all([X[i1][i]==X[i1][0] for i in range(len(X[0]))])]
    #print('Deleting columns {}'.format(col_0s))
    #print(len(col_0s))
    return col_0s


'''
Given dataset and target as numpy array returns the n main features,
where:
    X: dataset
    y: classification results
    n: number of feature in output

The target values y (class labels in classification, real numbers in regression).

feature_selection_RFECV_reg(X,y,n_tree)
Application of RFECV on Training set X,y by using 'n_tree' trees
Returns a list of selected features's index

Feature selection with Random Forest, combined to ecursive feature elimination and cross validation.
Source: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)

'''
def feature_selection_RFECV(X,y,n_tree):
    sel = RandomForestClassifier(n_estimators=n_tree,random_state=None)  #random_state=40)
    #sel =linear_model.Lars()
    rfecv = RFECV(estimator=sel, step=1,cv=StratifiedKFold(5),   scoring='accuracy')

    selector = rfecv.fit(X,y)
    r= permutation_importance(selector,X,y, n_repeats=10, random_state=0)
    print(r)
    
    for i in r.importances_mean.argsort()[::-1]:
    	if r.importances_mean[i] -2 * r.importances_std[i] >0:
           print(f"{titles[i]:<8}"
                f"{r.importances_mean[i]:.3f}"
                f"+/-{r.importances_std[i]:.3f}")
    print(selector.support_)
    print(selector.ranking_)

    print("\tOptimal number of features : %d" % selector.n_features_)
    # Plot number of features VS. cross-validation scores
    #plt.figure()
    #plt.xlabel("Number of features selected")
    #plt.ylabel("Cross validation score")
    #plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
    #plt.show()
    return [i for i in range(len(selector.support_)) if selector.support_[i]==True]

def exclude_duplicates_rows(X,y):
	dop=set()
	h_dict = {}
	new_X,new_y=[],[]
	for i in range(len(X)):
		digest = hashlib.sha1(str(X[i]).encode()).digest() 
		if digest not in h_dict.keys():
			h_dict[digest]=[(X[i],y[i])]
		else:
			h_dict[digest]+=[(X[i],y[i])]
		#dop.add(i)
		#dop.add(i1)
	for v in h_dict.values():
		if len(v)==1:
			new_X+=[v[0][0]]
			new_y+=[v[0][1]]
		else:
			if(len(set(([int(el[1]) for el in v])))==1):
				for el in v:
					# avg =
					mean = np.mean([e[1] for e in v])
					new_X+=[el[0]]
					new_y+=[mean]

	return np.array(new_X),np.array(new_y)

'''
show_heatmap(X_train)
X_train is the cleaned training set

Calculates the correlation that exist between Molecular descriptors
Shows correlation with a heatmap
'''
def show_heatmap(X_train):
    correlation_mat_clean=X_train.corr().abs()
    upper_half=correlation_mat_clean.where(np.triu(np.ones(correlation_mat_clean.shape), k=1).astype(np.bool))
    plt.figure(figsize=(50, 50))
    upper_filt_hmap=sns.heatmap(upper_half)
    plt.show()


def log(X, s):
    print('#'*10+'\nlenght of {}: {}\n length of one line of {}: {}\n{}\n'.format(s,len(X),s,1 if not isinstance(X[0], Iterable) else len(X[0]),X)+'#'*10)

'''
dump_model(fname,model,X_test,y_test)
Export the trained model in fname file
'''
def dump_model(fname,model,X_test,y_test):
    joblib.dump(model, fname)

def evaluation(y_pred, y_test):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp + tn)/(tp+tn+fp+fn)
    precision_score = tp / (tp + fp)
    recall_score = tp / (tp + fn)
    print('\tevaluation: Accuracy = {}\n\tPrecision  = {}\n\tRecall = {}'.format(accuracy,precision_score,recall_score))


if __name__ == '__main__':
    
    if len(sys.argv)!=3:
        print('Usage: python3 prog <n_tree> <corr_ind>')
        exit(1) 
    n_tree = int(sys.argv[1])
    correlation_index=float(sys.argv[2])
    print('n_tree = {}, corr_index = {}'.format(n_tree,correlation_index))
    #print('INPUT: n_tree {}'.format(n_tree))


    target, excluded = -1,[0] # target = cluster col, excluded = list of columns to exclude

    X,y,titles = read_data(fname,target,excluded)   # titles do not consider target and excluded
    #log(X,'X after reading file')
    #log(y,'y after reading file')
    #log(titles,'titles after reading file')
    X,y = exclude_duplicates_rows(X,y)

   
    '''
    Normalize before feature selection seems to be a best practise -> https://stackoverflow.com/questions/46062679/right-order-of-doing-feature-selection-pca-and-normalization
    
    For tiny datasets substitute null values with sample mean seems to be a valid choice -> https://www.datasciencelearner.com/deal-with-missing-data-python/

    need check on better scientific sources
    '''
    random_state = np.random.RandomState(4)
    X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=random_state)

    #log(X_train,'X_train after train_test_split')
    #log(X_test,'X_test after train_test_split')
    #log(y_train,'y_train after train_test_split')
    #log(y_test,'y_test after train_test_split')
    #y_train=[2 if el<=30 else 1 if el <=60 else 0 for el in y_train]
    #y_test=[2 if el<=30 else 1 if el<=60 else 0 for el in y_test]

    #target_scaler = MinMaxScaler()
    #y_train = target_scaler.fit_transform(y_train.reshape(-1,1))
    #y_test = target_scaler.transform(y_test.reshape(-1,1))
    #y_train = y_train.reshape(1,-1)[0]
    #y_test = y_test.reshape(1,-1)[0]
    #n_bin = math.ceil((max(y_train)-min(y_train))) // len(y_train)
    #print(max(y_train)-min(y_train))
    #print(n_bin)
    #exit()
    enc = KBinsDiscretizer(n_bins=2,encode='ordinal',strategy='uniform')
    y_orig = y_test
    y_train = enc.fit_transform(y_train.reshape(-1,1))
    y_test = enc.transform(y_test.reshape(-1,1))
    
    y_train = y_train.reshape(1,-1)[0]

    
    y_test = y_test.reshape(1,-1)[0]
    #y_train=[0 if el<activity_treshold else 1 for el in y_train]
    #y_test=[0 if el<activity_treshold else 1 for el in y_test]

    #print(y_test)
    #y_train = [0 if el<15 else 1 for el in y_train]
    #y_test = [0 if el<15 else 1 for el in y_test]
    #print(y_test)
    #exit()
    #print(y_test)
    #print(max(y_train))
    #print(min(y_train))
    #exit()
    # managing NULL values
    
    imp = imputing_missing_values(X_train)
    imp.fit(X_train)
    X_train = imp.transform(X_train)    
    X_test = imp.transform(X_test)
    joblib.dump(imp, 'imputizer_cv.bin', compress=True)
    exit()
    # managing constant columns checking on train set and remove them from train and test set

    col_0s = constant_columns(X_train)
    X_train = [[X_train[i][i1] for i1 in range(len(X_train[0])) if i1 not in col_0s] for i in range(len(X_train))]
    X_test = [[X_test[i][i1] for i1 in range(len(X_test[0])) if i1 not in col_0s] for i in range(len(X_test))]
    titles = [titles[i] for i in range(len(titles)) if i not in col_0s]
    
    
    # Correlation matrix    
    df=pd.DataFrame(X_train, columns=titles)

    #print(df)

    correlation_mat=df.corr().abs()
    #print(correlation_mat.shape)
    upper_half=correlation_mat.where(np.triu(np.ones(correlation_mat.shape), k=1).astype(np.bool))
    
    # High correlated descriptors to remove
    to_drop=[column for column in upper_half.columns if any(upper_half[column]>correlation_index)]
    
    #show_heatmap(df)
    X_train_clean=df.drop(df[to_drop], axis=1)

    # Write csv
    #X_train_clean.to_csv('dataset_cleaned.csv')

    # Show heatmap  
    #show_heatmap(X_train_clean)

    #log(to_drop,'high correlated features')


    # Remove to_drop features from train, test and titles
    X_train = X_train_clean.to_numpy()
    X_test = [[X_test[i][i1] for i1 in range(len(X_test[0])) if titles[i1] not in to_drop] for i in range(len(X_test))]
    titles = [titles[i] for i in range(len(titles)) if titles[i] not in to_drop]


    #log(X_train,'X_train after remove high correlated ft')
    #log(X_test,'X_test after remove high correlated ft')
    #log(titles,'titles after remove high correlated ft')

    # Standard Scaler computed from train set and applied to train an)d test

    X_train=np.array(X_train)
    X_test=np.array(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    #Export Scaler
    joblib.dump(scaler, 'std_scaler.bin', compress=True)
    exit()
    open('dimension_for_scaler.txt','w').write(str(titles))
    # test is normalized according to train scaler
    X_test = scaler.transform(X_test)

    

    #log(X_train,'X_train before feature selection')
    #log(y_train,'y_train before feature selection')
    

    ft_indices = feature_selection_RFECV(X_train,y_train,n_tree)
    #print([titles[i] for i in ft_indices])
    write_ft(titles,ft_indices)
    X_train = [[el[i] for i in range(len(el)) if i in ft_indices] for el in X_train]
    X_test = [[el[i] for i in range(len(el)) if i in ft_indices] for el in X_test]
    X = [[el[i] for i in range(len(el)) if i in ft_indices] for el in X]
    
    #log(X_train,'X_train after feature selection')
    #log(ft_indices,'ft_indices feature selected')
    
    sel_descr=[titles[i] for i in ft_indices]
    
    parameters = {'kernel':['rbf'], \
    'C':[eval('1e+{}*0.001'.format(x)) for x in range(6)],\
    'gamma': [eval('1e-{}'.format(x)) for x in range(4)]
    }
    


    svc=svm.SVC()
    #print('GridSearchCV\n'+'-'*12)
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train,y_train)
    f=clf.best_params_
    #print('best parameters ', f)
    #print('-'*12)


    confidence=clf.score(X_test,y_test)

    y_pred=clf.predict(X_test)
    
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #print("Precision: ", metrics.precision_score(y_test,y_pred,average='macro')) 
    # saving model
    #print(confusion_matrix(y_test,y_pred))
    #print(y_test)
    #print(y_orig)
    #print(y_pred)
    #print(y)
    dump_model('Classifier.sav',clf,X_test,y_test)

    evaluation(y_pred,y_test)
    print('#'*10)
    

