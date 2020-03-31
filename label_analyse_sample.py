import pandas as pd
import numpy as np
train_file = "/home/chongfaqin/data/example_train_v4.csv"
df=pd.read_csv(train_file,delimiter="\t",names=["label","user_key","user_id","goods_id"],dtype=str)
print(df.head(10))
val_count=df["label"].value_counts()
print(type(val_count))
print("0",val_count[0])
print("1",val_count[1])
pos=val_count[1]
neg_sample_df=df[df["label"]==0].sample(n=pos)
pos_sample_df=df[df["label"]==1].sample(n=pos)
sample_df=pd.concat([neg_sample_df,pos_sample_df])
print("sample_df",np.shape(sample_df))
test_count=50000
test_df=sample_df.sample(n=test_count)
print("test_df",np.shape(test_df))
print(test_df.head(10))
# train_df = pd.concat([user_item_lable_df, test_df], axis=0)   # 拼接
# train_df=user_item_lable_df
train_df=sample_df.drop(test_df.index)
print("train_df",np.shape(train_df))
print(train_df.head(10))
test_df.to_csv("/home/chongfaqin/data/example_test_data_v4.txt",sep="\t",header=False,index=False)
train_df.to_csv("/home/chongfaqin/data/example_train_data_v4.txt",sep="\t",header=False,index=False)

