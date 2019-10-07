import pandas as pd
#import os.path
import pathlib
file = "/home/diwakar/Downloads/Sample Transactions-test Data.csv"

abc= pathlib.Path(file)
print(abc.exists())
def importFile(file):
    if abc.exists():
        df = pd.read_csv(file, dtype=object)



    else:
        raise FileNotFoundError("File not found")
    #print(df)
    return file




importFile(abc)
# def div(x,y):
#     if y==0:
#         raise ZeroDivisionError("cannot div by zero")
#     else:
#         return x/y
# div(5,0)