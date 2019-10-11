import re
import pandas as pd
import numpy as np
import os.path
import pathlib
from numpy.ma import column_stack

# for supressing warnings
import warnings

warnings.filterwarnings("ignore")

# giving the file path and the required features
data_csv_path = "Sample Transactions with Target Variable.csv"
features = ["PROCESSING_CODE", "POS_DATA", "TRANSACTION_AMOUNT", "POS_ENTRY_MODE", "CARD_ACCEPTOR_ACTIVITY",
            "CODE_ACTION"]
features.sort()
print(features)
length = len(features)
abc = pathlib.Path(data_csv_path)
read_csv_variable = pd.read_csv


# importing csv file
def importdata_csv_path(data_csv_path):
    if abc.exists():
        csv_data = read_csv_variable(data_csv_path, dtype=object)
        return csv_data
    else:
        raise FileNotFoundError("data_csv_path not found")
    return data_csv_path


# checking the required features are present in the csv file
def featureExtraction(data_frame, fearures, length):
    if len(features) == length:
        if all(x in list(data_frame) for x in features):
            data_frame = data_frame[features]
            return list(data_frame)
        else:
            raise NameError("Features not found")
    else:
        raise IndexError("Length not matching")


# calling the import method and it returns a dataframe
data = importdata_csv_path(abc)

# passing that dataframe for checking the features
feature = featureExtraction(data, features, length)
# print(feature)
# if the features are present it will return a list of features that will be passed as an argument for the dataframe
# for selecting that particular features alone
data_with_req_features = data[feature]

# making the POS_DATA column as 12 digits using zfill fn
# data_with_req_features[features[2]] = data_with_req_features[features[2]].str.zfill(12)

# replacing the ??? in the POS Entry Mode to '000' finding the POS_ENTRY_MODE by index
index = feature.index('POS_ENTRY_MODE')
data_with_req_features[features[index]] = data_with_req_features[features[index]].replace('???', '000')

# finding the columns having Exponential values
# having_E = data_with_req_features[features[2]].str.contains('E')

# filtering ot those columns with not operator
# dataframe = data_with_req_features[~having_E]

# assigning appropriate types for all the columns

convert_dict = {'POS_ENTRY_MODE': str,
                'POS_DATA': str,
                'PROCESSING_CODE': str,
                'TRANSACTION_AMOUNT': float,
                'CARD_ACCEPTOR_ACTIVITY': str,
                'CODE_ACTION': str
                }

dataframe = data_with_req_features.astype(convert_dict)
print(dataframe.dtypes)

index = list(dataframe).index('POS_DATA')

# splitting the POS_DATA column into 12 different columns
dataframe['TERM_CARD_READ_CAP'] = dataframe['POS_DATA'].str[0:1]
dataframe['TERM_CH_VERI_CAP'] = dataframe['POS_DATA'].str[1:2]
dataframe['TERM_CARD_CAPTURE_CAP'] = dataframe['POS_DATA'].str[2:3]
dataframe['TERM_ATTEND_CAP'] = dataframe['POS_DATA'].str[3:4]
dataframe['CH_PRESENCE_IND'] = dataframe['POS_DATA'].str[4:5]
dataframe['CARD_PRESENCE_IND'] = dataframe['POS_DATA'].str[5:6]
dataframe['TXN_CARD_READ_IND'] = dataframe['POS_DATA'].str[6:7]
dataframe['TXN_CH_VERI_IND'] = dataframe['POS_DATA'].str[7:8]
dataframe['TXN_CARD_VERI_IND'] = dataframe['POS_DATA'].str[8:9]
dataframe['TRACK_REWRITE_CAP'] = dataframe['POS_DATA'].str[9:10]
dataframe['TERM_OUTPUT_IND'] = dataframe['POS_DATA'].str[10:11]
dataframe['PIN_ENTRY_IND'] = dataframe['POS_DATA'].str[11:12]

# deleting the POS_DATA column after splitting
dataframe.drop(columns=['POS_DATA'], inplace=True)

dataframe.to_csv('task.csv')

# print(new[0])
# print(new)
# print(dataframe.iloc[70445])
print(dataframe.dtypes)
# = dataframe[feature[index]]
# print(a)

# removing indexes
# dataframe.reset_index()

# for time being deleting the POS_DATA column for training and creating a model
# dataframe.drop(columns=['POS_DATA'], inplace=True)

# selecting X and y for the training data, X is the inputs and y is the response variable
X = dataframe.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
y = dataframe.iloc[:, 1]
print(X)
print(y)

# importing required packages for train test split, confusion matrix,accuracy score and for KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
#
# # from sklearn.model_selection import  GridSearchCV
# # parameters={'n_neighbours':np.array([1,3,5,7,9])}
# # knn=KNeighborsClassifier()
# # cv=GridSearchCV(knn,param_grid=parameters,cv=5)
# # cv.fit(X,y)
# # print(cv.cv_results_)
# # print(cv.best_params_)
# # print(cv.best_score_)
# # print(cv.best_estimator_)
#
# splitting the training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2019,stratify=y)

# applying KNN classification algorithm (as our response variable is response code, categorical data) with nearest
# neighbours = 3
try:
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    # print confusion matrix and classification report
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # printing accuracy score
    print(accuracy_score(y_test, y_pred))

except ValueError as e:
  print(e)




# from sklearn.metrics import roc_curve, roc_auc_score
#
# y_pred_proba = knn.predict_proba(X_test)[:, 3]
# fpr, tpr, treshold = roc_curve(y_test, y_pred_proba)

# testdf=pd.read_csv("/home/diwakar/FORTIATE/BUILD/WORKSPACES/PYTHON/Ml-Transaction-Model/test.csv",dtype=object,index_col=False)
# dum_df=(testdf.iloc[:,1:])
# print(dum_df.head())
# print(dum_df.dtypes)
# dum_df = dum_df.astype({'Merchant Category Code':'object','POS Entry Mode':'object','POS_DATA':'str','PROCESSING_CODE':'object','Response Code':'object','Transaction Amount':'double'})
# print(dum_df.dtypes)
# # dum_df=dum_df['POS_DATA'].filter(regex='!+')
#
# dum_df['POS_DATA'] = dum_df['POS_DATA'].astype('str')
# print(dum_df['POS_DATA'].dtypes)
# dum_df['POS_DATA'].str.split(expand=True)
# print(dum_df['POS_DATA'])
# # print(dum_df.iloc[20])
#
# # dum_df.POS_DATA.str.extractall('', flags=re.U)[0].unstack().rename_axis(None, 1)
#
# # dum_df['POS_DATA'].apply(lambda x: pd.Series(list(x)))
#
# 1
# 2

# df3 = pd.DataFrame(dum_df.POS_DATA.str.split().tolist().split())
# print(df3)

# dum_df=dum_df.dropna(dum_df['POS_DATA'])
# dum_df = dum_df[~dum_df['POS_DATA'].str.contains('E')]
# print(data_with_req_features.dtypes)
# print(data_with_req_features.loc[200])
# print(all(x in list(df) for x in features))
# if list(df).__contains__(features):
#     print("Yes")
# else:
#     print("No")


# print(df)
# print(list(df))
# df=df[features]
# print(list(df))
# print(df1)
# df1['POS_DATA'] = df1['POS_DATA'].apply(lambda x: x.zfill(12))
# df1['POS_DATA']=df1['POS_DATA'].apply(lambda x: '{0:0>12}'.format(x))

# #df1['POS Entry Mode']=df1['POS Entry Mode'].fillna(value=0)
# #df1.loc[df1['POS Entry Mode'].isnull(),'POS Entry Mode']=0
# df1['POS Entry Mode'] = df1['POS Entry Mode'].replace('???', 0)
# data_with_req_features=pd.read_csv("/home/diwakar/FORTIATE/BUILD/WORKSPACES/PYTHON/Ml-Transaction-Model/data_csv_path2.csv",dtype=object)
# print(data_with_req_features.head(4000))
# print(data_with_req_features.loc[3256])


# from datetime import datetime
# start = datetime.now()
# dataframe=pd.read_csv(data_csv_path,dtype=object)
# end = datetime.now()
# time_taken = end - start
# print('Time: ',time_taken)


# print(data_with_req_features.iloc[2600])
# data_with_req_features[features[2]] = data_with_req_features[features[2]].apply(lambda x: x.zfill(12))
# data_with_req_features[features[1]] = data_with_req_features[features[1]].replace('???', '000')
# data_with_req_features[features[1]] = data_with_req_features[features[1]].apply(lambda x: x.zfill(3))


# print(data_with_req_features.iloc[2600])
# data_with_req_features[features[1]] = data_with_req_features[features[2]].replace('????????????', '000000000000')
# data_with_req_features[features[1]] = data_with_req_features[features[2]].replace('????????????', '000000000000')
# print(data_with_req_features.dtypes)
# data_with_req_features[feature[2]].astype(float)
# data_with_req_features[feature[2]]=pd.to_numeric(data_with_req_features[feature[2]])
# print(data_with_req_features)


# new=dataframe['POS_DATA'].str.split(r',\d*', expand=True)
