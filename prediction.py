#data preprocessing
import pandas as pd
#produces a prediction model in the form of an ensemble of weak prediction models, typically decision tree
import xgboost as xgb
#the outcome (dependent variable) has only a limited number of possible values. 
#Logistic Regression is used when response variable is categorical in nature.
from sklearn.linear_model import LogisticRegression
#A random forest is a meta estimator that fits a number of decision tree classifiers 
#on various sub-samples of the dataset and use averaging to improve the predictive 
#accuracy and control over-fitting.
from sklearn.ensemble import RandomForestClassifier
#a discriminative classifier formally defined by a separating hyperplane.
from sklearn.svm import SVC
# Standardising the data.
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#for measuring training time
from time import time 

# F1 score (also F-score or F-measure) is a measure of a test's accuracy. 
#It considers both the precision p and the recall r of the test to compute 
#the score: p is the number of correct positive results divided by the number of 
#all positive results, and r is the number of correct positive results divided by 
#the number of positive results that should have been returned. The F1 score can be 
#interpreted as a weighted average of the precision and recall, where an F1 score 
#reaches its best value at 1 and worst at 0.
from sklearn.metrics import classification_report
#f1_score, precision_score, recall_score

#Compute the precision
#The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
#The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
#The best value is 1 and the worst value is 0.

#Compute the recall
#The recall is intuitively the ability of the classifier to find all the positive samples.
#The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.
#The best value is 1 and the worst value is 0.
# sklearn.metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)[source]

## steps:
# 1. visualise data
# 2. classify SVM xgboost etc
# 3. validate classifier with confusion matrix

file = 'data/final_dataset.csv'

df = pd.read_csv(file, parse_dates=True)
print(df['home'].unique())

# dropping redundant column? not sure if this is true
drop_columns = ['Unnamed: 0','date','home','away','round','fthp',
                'ftap','hm4','hm5','am4','am5',
                'rn','htformptsstr','atformptsstr','HTWinStreak3',
       'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5', 'ATWinStreak3',
       'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5']
# keep: FTR	HTP	ATP	HM1	HM2	HM3	AM1	AM2	AM3	HTGD	ATGD	DiffFormPts	DiffLP
data = df.drop(drop_columns, axis=1)

# Separate into feature set and target variable
#FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
X_all = data.drop(['ftr'],1)
y_all = data['ftr']


#Center to the mean and component wise scale to unit variance.
cols = [['htgd','atgd','htp','atp','diffpts','htps','htpc','atps','atpc','htformpts','atformpts']]
for col in cols:
    X_all[col] = scale(X_all[col])
    
#last 3 wins for both sides
X_all['hm1'] = X_all['hm1'].astype('str')
X_all['hm2'] = X_all['hm2'].astype('str')
X_all['hm3'] = X_all['hm3'].astype('str')
X_all['am1'] = X_all['am1'].astype('str')
X_all['am2'] = X_all['am2'].astype('str')
X_all['am3'] = X_all['am3'].astype('str')

#we want continous vars that are integers for our input data, so lets remove any categorical vars
def preprocess_features(X):
    ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))
print(X_all.head(5))


# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size = 50,
                                                    random_state = 22,
                                                    stratify = y_all)


def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    
    end = time()
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))
    return classification_report(target, y_pred), confusion_matrix(target, y_pred)


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    print('')
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    report, conf = predict_labels(clf, X_train, y_train)
    # print(f1, acc, prec, recall)
    print('Training data results:')
    print(report, conf)
    
    report, conf = predict_labels(clf, X_test, y_test)
    print('Test data results:')
    print(report, conf)

    # Initialize the three models (XGBoost is initialized later)
clf_A = LogisticRegression(random_state = 22, solver='lbfgs', max_iter=1000)
clf_B = SVC(random_state = 912, kernel='rbf', gamma='scale')
#Boosting refers to this general problem of producing a very accurate prediction rule 
#by combining rough and moderately inaccurate rules-of-thumb
clf_C = xgb.XGBClassifier(seed = 82)

train_predict(clf_A, X_train, y_train, X_test, y_test); print('')
# train_predict(clf_B, X_train, y_train, X_test, y_test); print('')
# train_predict(clf_C, X_train, y_train, X_test, y_test); print('')
