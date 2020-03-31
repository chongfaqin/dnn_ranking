####################### load packages #####################
import numpy as np
import tensorflow as tf
import math
import pandas as pd
from sklearn.utils import shuffle
from tf_common.nn_module import dense_block
####################### dataset #####################
# train_file = "/home/chongfaqin/data/example_train_data_v4.txt"
# test_file = "/home/chongfaqin/data/example_test_data_v4.txt"
# dw_user_embedding_file="/home/chongfaqin/data/dw_user_embedding.data"
# dw_goods_embedding_file="/home/chongfaqin/data/dw_goods_embedding.data"

train_file = "data/example_train_data_v4.txt"
test_file = "data/example_test_data_v4.txt"
dw_user_embedding_file="data/dw_user_embedding.data"
dw_goods_embedding_file="data/dw_goods_embedding.data"
ae_user_embedding_file="data/ae_user_embedding.data"
ge_goods_embedding_file="data/ge_goods_embedding.data"
######## 网络相关参数 ########
batch_size = 64
learning_rate = 0.001
training_epochs = 12
display_epoch = 1
feature_size = 264
label_count=2
fc_dim=128
fc_dropout=0.05
####################### 读取数据 #####################
def load_dw_data():
    dw_user={}
    dw_user_file=open(dw_user_embedding_file,"r",encoding="utf-8")
    for line in dw_user_file.readlines():
        arr=line.strip().split("\t")
        if(len(arr)==2):
            dw_user[arr[0]]=arr[1].split(",")
    dw_user_file.close()

    dw_goods = {}
    dw_goods_file=open(dw_goods_embedding_file,"r",encoding="utf-8")
    for line in dw_goods_file.readlines():
        arr=line.strip().split("\t")
        if(len(arr)==2):
            dw_goods[arr[0]]=arr[1].split(",")
    return dw_user,dw_goods

def load_ge_data():
    ge_goods={}
    ge_goods_file=open(ge_goods_embedding_file,"r",encoding="utf-8")
    for line in ge_goods_file.readlines():
        arr=line.strip().split("\t")
        if(len(arr)==2):
            ge_goods[arr[0]]=arr[1].split(",")

    ae_user={}
    ae_user_file=open(ae_user_embedding_file,"r",encoding="utf-8")
    for line in ae_user_file.readlines():
        arr=line.strip().split("\t")
        if(len(arr)==2):
            ae_user[arr[0]]=arr[1]
    return ge_goods,ae_user

kg_goods_embedding_dic,ae_user_embedding_dic=load_ge_data()
print(len(kg_goods_embedding_dic),len(ae_user_embedding_dic))

dw_user_embedding_dic,dw_goods_embedding_dic=load_dw_data()
print(len(dw_user_embedding_dic),len(dw_goods_embedding_dic))
default_arr=np.zeros(100)
# default_kg_arr=np.zeros(32)
default_ae_arr=np.zeros(64)
def read_data(pos, batch_size, data_lst):
    '''
    :param pos:         数据起始位置
    :param batch_size:  batch size
    :param data_lst:    数据流
    :return:            batch size的数据
    获取batch size的数据
    '''
    ######## 获取batch size数据 ########
    batch = data_lst[pos:pos + batch_size]

    # print(batch["items"].head(10))
    user_key_list=batch["user_key"]
    user_id_list=batch["user_id"]
    goods_list = batch["goods_id"]
    x_embedding=[]
    for uk,ui,g in zip(user_key_list,user_id_list,goods_list):
        item_embedding=[]
        if(ui in dw_user_embedding_dic):
            item_embedding.extend(dw_user_embedding_dic[ui])
        else:
            item_embedding.extend(default_arr)
        if(uk in ae_user_embedding_dic):
            item_embedding.extend(ae_user_embedding_dic[uk])
        else:
            item_embedding.extend(default_ae_arr)
        if(g in dw_goods_embedding_dic):
            item_embedding.extend(dw_goods_embedding_dic[g])
        else:
            item_embedding.extend(default_arr)
        # if(g in kg_goods_embedding_dic):
        #     item_embedding.extend(kg_goods_embedding_dic[g])
        # else:
        #     item_embedding.extend(default_kg_arr)
        x_embedding.append(item_embedding)

    #print(np.shape(x_embedding))
    y = batch["label"]
    # print(np.shape(x_embedding))
    # print(np.shape(x_embedding),x_embedding)
    return np.array(x_embedding),np.array(y)


####################### 定义模型 #####################
random_seed=2020
def multilayer_perceptron(x):
    hidden_units = [fc_dim]
    dropouts = [fc_dropout] * len(hidden_units)
    out_unit = dense_block(x, hidden_units=hidden_units, dropouts=dropouts, densenet=False, reuse=False, training=True, seed=random_seed, bn=True)
    return out_unit

######## x,y placeholder ########
x_emb = tf.placeholder(tf.float32, shape=[None, feature_size],name="feature")
y_batch = tf.placeholder(tf.int64, shape=[None],name="label")
########## define model, loss and optimizer ##########
#### model pred 前向计算,判断结果 ####
pred = multilayer_perceptron(x_emb)
#### loss 损失计算 ####
nce_weights = tf.Variable(tf.truncated_normal([fc_dim,label_count], stddev=1.0 / math.sqrt(fc_dim)))
nce_biases = tf.Variable(tf.zeros([label_count]))
logits = tf.nn.softmax(tf.matmul(pred, nce_weights)+nce_biases,name="score")
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch,logits=logits))
#### optimization 优化 ####
# optimizer = tf.train.FtrlOptimizer(learning_rate, l1_regularization_strength=0.01, l2_regularization_strength=0.01).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#### accuracy 准确率 ####
with tf.name_scope("accuracy"):
    # y_label = tf.reshape(y_batch,(batch_size,1))
    correct_pred = tf.equal(tf.argmax(logits, 1), y_batch)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

##################### train and evaluate model ##########################
########## initialize variables ##########
init = tf.global_variables_initializer()
config = tf.ConfigProto(device_count={"CPU": 16}, # limit to num_cpu_core CPU usage
                inter_op_parallelism_threads = 16,
                intra_op_parallelism_threads = 16,
                log_device_placement=False)
with tf.Session(config=config) as sess:
    sess.run(init)

    test_df = pd.read_csv(test_file, delimiter="\t", names=["label", "user_key", "user_id","goods_id"],dtype=str)
    test_total_batch = int(len(test_df) / batch_size)
    all_accuracy = 0
    for i in range(test_total_batch):
        emb,y = read_data(i * batch_size, batch_size, test_df)
        print(np.shape(emb), np.shape(y))
        accuracy_val = sess.run(accuracy, feed_dict={x_emb: emb,y_batch: y})
        all_accuracy = all_accuracy + accuracy_val
    print("Testing Accuracy:", all_accuracy / test_total_batch)

    df = pd.read_csv(train_file, delimiter="\t", names=["label", "user_key", "user_id", "goods_id"],low_memory=False,dtype=str)
    df = shuffle(df)
    print(df["label"].value_counts())
    #### epoch 世代循环 ####
    total_batch = int(len(df) / batch_size)
    for epoch in range(training_epochs):

        all_loss=0
        all_acc=0
        for i in range(total_batch):

            emb,y = read_data(i * batch_size, batch_size, df)
            _, loss_result, acc_result = sess.run([optimizer,cost,accuracy],feed_dict={x_emb: emb, y_batch: y})
            # print(np.shape(logits_result))
            all_loss=all_loss+loss_result
            all_acc=all_acc+acc_result

        if(epoch%display_epoch==0):
            all_accuracy = 0
            for i in range(test_total_batch):
                emb,y = read_data(i * batch_size, batch_size, test_df)
                accuracy_val = sess.run(accuracy, feed_dict={x_emb: emb,y_batch: y})
                all_accuracy = all_accuracy + accuracy_val
            print("Epoch " + str(epoch) + ", Epoche Loss=" + "{:.6f}".format(all_loss/total_batch) + ", Training Accuracy= " + "{:.5f}".format(all_acc/total_batch)+",Testing Accuracy="+"{:.5f}".format(all_accuracy / test_total_batch))


    builder = tf.saved_model.builder.SavedModelBuilder("model_v4_2/")
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING]
    )
    builder.save()
