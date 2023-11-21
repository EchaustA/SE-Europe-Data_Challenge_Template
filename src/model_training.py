import pandas as pd
import argparse
#import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
#from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from sklearn import metrics
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import LeaveOneOut
from copy import deepcopy

#from sklearn.model_selection import TimeSeriesSplit
#from sklearn.utils import resample # for Bootstrap sampling
import warnings
#from sklearn.ensemble import AdaBoostClassifier
warnings.filterwarnings("ignore")

def load_data(file_path):
    # TODO: Load processed data from CSV file
    df = pd.read_csv(file_path,index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

def split_data(df, val_size = .2):
    # TODO: Split data into training and validation sets (the test set is already provided in data/test_data.csv)
    
    # TODO: cross-validation

    y_lab = 'label'
    features = df.columns[df.columns != y_lab]
    X = df[:-1][features]
    y = df[1:][y_lab]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_size)

    return X_train, X_val, y_train, y_val

def split_data2(df):
    y_lab = 'label'
    df.sort_index(inplace=True)
    data = df[:-1].copy()
    data["label"] = df[1:].label.to_numpy()
    features = data.columns[data.columns != y_lab]
    X = data[features].astype(float)#.to_numpy()
    y = data[y_lab].astype(str)#.to_numpy()
    #X.reset_index(drop=True,inplace=True)
    #y.reset_index(drop=True,inplace=True)
    return X, y

# def cross_val1(classifier,X,y,scores_dict={},num_splits=10):
#     model = classifier
#     scores = cross_val_score(model,X,y,cv=num_splits)
#     scores_dict[type(model).__name__] = scores
#     return scores_dict

def cross_val2(classifier,X,y,scores_dict={},num_splits=10):
    """Cross validation for time series"""
    tss = TimeSeriesSplit(n_splits = num_splits)
    accuracy = []
    f1_score = []
    for train_index, test_index in tss.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = train_model2(deepcopy(classifier), X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy.append(metrics.accuracy_score(y_pred, y_test))
        f1_score.append(metrics.f1_score(y_pred, y_test,average='macro'))
    scores = (round(np.mean(accuracy),2),round(np.mean(f1_score),2))
    scores_dict[classifier] = scores

    return scores_dict

def cross_val3(classifier,X,y,scores_dict={}):
    """LOOCV"""
    loocv = LeaveOneOut()
    accuracy = []
    f1_score = []
    for train_index, test_index in loocv.split(X):
        #print("Train: ", (min(train_index),max(train_index)), "; test: ", (min(test_index),max(test_index)))
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = train_model2(deepcopy(classifier), X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy.append(metrics.accuracy_score(y_pred, y_test))
        f1_score.append(metrics.f1_score(y_pred, y_test,average='macro'))
    scores = (round(np.mean(accuracy),2),round(np.mean(f1_score),2))
    scores_dict[classifier] = scores

    return scores_dict

def train_model2(clf, X_train, y_train):
    #clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    return clf

def save_model(model, model_path):
    # TODO: Save your trained model
    import pickle
    pickle.dump(model, open(model_path, 'wb'))
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/train.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl', 
        help='Path to save the trained model'
    )
    return parser.parse_args()

def get_classifiers():
    clf = DecisionTreeClassifier()
    classifiers = [clf]
    xg_rng = np.arange(15,600,15)
    # for x in xg_rng:
    #     xg = XGBClassifier(n_estimator=x)
    #     classifiers.append(xg)
    for x in xg_rng:
        abc = AdaBoostClassifier(n_estimators=x, learning_rate=1)
        classifiers.append(abc)
    return classifiers

def skipme():
    """Auxiliary to pick the model"""
    num_splits = 30
    scores_dict = {}
    
    df = load_data(input_file)
    X, y = split_data2(df)
    
    df2 = load_data("./data/test.csv")
    X2, y2 = split_data2(df2)

    #from sklearn.preprocessing import LabelEncoder
    #le = LabelEncoder()
    #y=le.fit_transform(y)
    #from sklearn.preprocessing import MultiLabelBinarizer
    #mlb = MultiLabelBinarizer()
    #y = mlb.fit_transform(y)
    classifiers = get_classifiers()
    iter = 0
    for classifier in classifiers:
        scores_dict = cross_val2(classifier,X,y,scores_dict,num_splits)
        iter += 1
        if iter//10 == 0:
            print(f"{round(iter/len(classifiers)*100,2)}%",end="\n")

    [scores_dict[key] for key in scores_dict.keys()]

def main(input_file, model_file):
    
    df = load_data(input_file)
    X, y = split_data2(df)

    classifier = DecisionTreeClassifier()
    model = train_model2(deepcopy(classifier), X, y)
    save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)