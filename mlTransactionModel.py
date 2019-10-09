import pandas as pd
import numpy as np
#import os.path
import pathlib
file = "/home/diwakar/Downloads/Sample Transactions-test Data.csv"
features=["PROCESSING_CODE","POS_DATA","Transaction Amount","Merchant Category Code","POS Entry Mode","Response Code"]
features.sort()
print(features)
length=len(features)


abc= pathlib.Path(file)
print(abc.exists())
def importFile(file):
    if abc.exists():
        # df = pd.read_csv(file, dtype=object)
        df=pd.read_csv(file, dtype={'POS_DATA': 'str'})
        print(df.head(2600))
        return df
    else:
        raise FileNotFoundError("File not found")
    #print(df)
    return file

def featureExtraction(file,fearures,length):
    if len(features)==length:
        if(all(x in list(df) for x in features)):
            df1=file[features]
            #print(features)
            # df1[features[2]] = df1[features[2]].apply(lambda x: x.zfill(12))
            # df[features[1]] = df1[features[1]].replace('???', 0)
            # #df1.to_csv("test.csv",header=True)
            return list(df1)
        else:
            raise NameError("Features not found")
    else:
        raise IndexError("Length not matching")

    #print(df1)


df = importFile(abc)
feature=featureExtraction(df,features,length)
df2=df[feature]
print(df2.iloc[2600])
# df2[features[2]] = df2[features[2]].apply(lambda x: x.zfill(12))
# df2[features[1]] = df2[features[1]].replace('???', 0)
# df2[features[1]] = df2[features[1]].apply(lambda x: x.zfill(3))

df2[features[2]] = df2[features[2]].str.zfill(12)
df2[features[1]] = df2[features[1]].replace('???', '000')
#df2[features[1]] = df2[features[2]].replace('????????????', '000000000000')
#df2[features[1]] = df2[features[2]].replace('????????????', '000000000000')
print(df2.dtypes)
#df2[feature[2]].astype(float)
#df2[feature[2]]=pd.to_numeric(df2[feature[2]])
print(df2)
# print(df2.dtypes)
# print(df2.loc[200])
# print(all(x in list(df) for x in features))
# if list(df).__contains__(features):
#     print("Yes")
# else:
#     print("No")




#print(df)
# print(list(df))
# df=df[features]
# print(list(df))
#print(df1)
# df1['POS_DATA'] = df1['POS_DATA'].apply(lambda x: x.zfill(12))
#df1['POS_DATA']=df1['POS_DATA'].apply(lambda x: '{0:0>12}'.format(x))

# #df1['POS Entry Mode']=df1['POS Entry Mode'].fillna(value=0)
# #df1.loc[df1['POS Entry Mode'].isnull(),'POS Entry Mode']=0
#df1['POS Entry Mode'] = df1['POS Entry Mode'].replace('???', 0)
# df2=pd.read_csv("/home/diwakar/FORTIATE/BUILD/WORKSPACES/PYTHON/Ml-Transaction-Model/file2.csv",dtype=object)
# print(df2.head(4000))
# print(df2.loc[3256])



# from datetime import datetime
# start = datetime.now()
# dataframe=pd.read_csv(file,dtype=object)
# end = datetime.now()
# time_taken = end - start
# print('Time: ',time_taken)


