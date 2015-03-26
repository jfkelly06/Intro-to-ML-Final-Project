#!/usr/bin/python

import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label


### load the dictionary containing the dataset
##Remove Outliers
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL", 0)

##Create New Features
def computeFraction( poi_messages, all_messages ):
    if poi_messages and all_messages != 'NaN':
        fraction = float(poi_messages) / float(all_messages)
    else:
        fraction = 0.
    return fraction
  
for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["to_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi

#features_list = ['poi', 'expenses', 'deferred_income', 'shared_receipt_with_poi', 'bonus', 'total_stock_value', 'from_poi_to_this_person', 'from_this_person_to_poi', 'restricted_stock', 'salary', 'total_payments', 'fraction_to_poi', 'exercised_stock_options']
features_list = ['poi', 'salary', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'fraction_from_poi', 'exercised_stock_options', 'fraction_to_poi', 'long_term_incentive', 'restricted_stock', 'shared_receipt_with_poi']
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

##Will do feature scaling because of PCA. PCA makes dimensional tradeoff while RandomForest does not
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)

##Split into training and testing data
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.1, random_state=0)


##Train Algorithm & Tune Parameters

#Try Random Forest
#from sklearn.grid_search import GridSearchCV
#from sklearn.ensemble import RandomForestClassifier
#parameters = {'min_samples_split':[2, 10], 'n_estimators':[50, 250], 'criterion':['gini', 'entropy']}
#RF = RandomForestClassifier()
#clf = GridSearchCV(RF, parameters, scoring = 'recall')
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#print {"Best Estimator1": clf.best_estimator_}
#print {"Feature Importances1": clf.best_estimator_.feature_importances_}

#Try SVM
#from sklearn.grid_search import GridSearchCV
#from sklearn.svm import SVC
#SVM = SVC()
#parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
#clf = GridSearchCV(SVM, parameters, scoring = 'recall')
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#print {"Best Estimator1": clf.best_estimator_}
#print {"Feature Importances1": clf.best_estimator_.feature_importances_}

#Try Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)


##Remove Features-take out those that are not very important-All contribute >5%

##Do PCA-And tune # of components
n_components = 15
from sklearn.decomposition import RandomizedPCA
pca = RandomizedPCA(n_components=n_components, whiten=True)
eigenfeatures = pca.fit_transform(features)
##print {"Explained Variance Ratio": pca.explained_variance_ratio_}


##Re-split into training and testing data
from sklearn.cross_validation import StratifiedKFold
skf = StratifiedKFold( labels, n_folds=3 )
accuracies = []
precisions = []
recalls = []
for train_idx, test_idx in skf: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( eigenfeatures[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( eigenfeatures[jj] )
        labels_test.append( labels[jj] )

##Retrain Algorithm        
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    

##Evaluate
    ### for each fold, print some metrics
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
     
    print "accuracy score: ", round(accuracy_score( labels_test, pred ), 3)
    print "precision score: ", round(precision_score( labels_test, pred ), 3)
    print "recall score: ", round(recall_score( labels_test, pred ), 3)

    accuracies.append( accuracy_score(labels_test, pred) )
    precisions.append( precision_score(labels_test, pred) )
    recalls.append( recall_score(labels_test, pred) )

### aggregate precision and recall over all folds
#print {"Best Estimator2": clf.best_estimator_}
print"average accuracy: ", round(sum(accuracies)/3., 3)
print "average precision: ", round(sum(precisions)/3., 3)
print "average recall: ", round(sum(recalls)/3., 3)





### dump your classifier, dataset and features_list so 
### anyone can run/check your results
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )
pickle.dump(eigenfeatures, open("eigenfeatures.pkl", "w") )


