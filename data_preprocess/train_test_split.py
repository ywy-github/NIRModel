import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data=pd.read_excel("../data/二期双十.xlsx",sheet_name="train+val")
    y=data.loc[:,"tumor_nature"]
    X=data.drop(["tumor_nature"],axis=1)
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.17,random_state=2)
    train=pd.concat([x_train,y_train],axis=1)
    test=pd.concat([x_test,y_test],axis=1)
    train.to_excel("../data/train1.xlsx",index=False)
    test.to_excel("../data/test1.xlsx",index=False)