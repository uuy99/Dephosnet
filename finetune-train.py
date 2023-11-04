from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, AveragePooling1D
from keras.layers.pooling import GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.layers import Input, merge, Flatten,Reshape
import scipy.io as sio
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

import tensorflow.python.keras.engine

from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.regularizers import l2
import keras.backend as K

from tensorflow.keras.layers import Layer
from keras.layers import Conv1D,Conv2D, MaxPooling2D,concatenate
from tensorflow.keras.layers import * 
from tensorflow.keras.layers import Layer
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.optimizers import Adam,SGD
from cProfile import label
import csv
import numpy as np
import tensorflow.keras.utils as kutils
from tensorflow.python.keras.initializers import get
from tensorflow import keras


from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score





def getMatrixLabel(positive_position_file_name,sites, window_size=49  , empty_aa = '*'):
    prot = []  # list of protein name
    pos = []  # list of position with protein name
    rawseq = []
    all_label = []

    short_seqs = []
    half_len = (window_size - 1) / 2

    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
            
        for row in reader:
            # print("row[0]:",int(row[0]))
            position = int(row[2])
            sseq = row[3]
            rawseq.append(row[3])
            center = sseq[position - 1]
            if center in sites:
                all_label.append(int(row[0]))
                # print("length of all_label",len(all_label))
                prot.append(row[1])
                pos.append(row[2])

                
                if position - half_len > 0:
                    start = position - half_len
                    start = int(start)
                    position = int(position)
                    left_seq = sseq[start - 1:position - 1]
                else:
                    #如果在左侧,left_seq中最右侧为位点
                    left_seq = sseq[0:position - 1]

                end = len(sseq)
                if position + half_len < end:
                    end = position + half_len
                    end = int(end)
                right_seq = sseq[position:end]

                if len(left_seq) < half_len:
                    nb_lack = half_len - len(left_seq)
                    nb_lack = int(nb_lack)
                    left_seq = ''.join([empty_aa for count in range(nb_lack)]) + left_seq

                if len(right_seq) < half_len:
                    nb_lack = half_len - len(right_seq)
                    nb_lack = int(nb_lack)
                    right_seq = right_seq + ''.join([empty_aa for count in range(nb_lack)])
                shortseq = left_seq + center + right_seq
                short_seqs.append(shortseq)
        targetY = kutils.to_categorical(all_label)
        letterDict = {}
        letterDict["A"] = 0
        letterDict["C"] = 1
        letterDict["D"] = 2
        letterDict["E"] = 3
        letterDict["F"] = 4
        letterDict["G"] = 5
        letterDict["H"] = 6
        letterDict["I"] = 7
        letterDict["K"] = 8
        letterDict["L"] = 9
        letterDict["M"] = 10
        letterDict["N"] = 11
        letterDict["P"] = 12
        letterDict["Q"] = 13
        letterDict["R"] = 14
        letterDict["S"] = 15
        letterDict["T"] = 16
        letterDict["V"] = 17
        letterDict["W"] = 18
        letterDict["Y"] = 19
        letterDict["*"] = 20
        # letterDict["?"] = 21
        Matr = np.zeros((len(short_seqs), window_size))
        samplenumber = 0
        for seq in short_seqs:
            AANo = 0
            for AA in seq:
                if AA not in  letterDict:
                    # AANo += 1
                    continue
                Matr[samplenumber][AANo] = letterDict[AA]
                AANo = AANo+1
            samplenumber = samplenumber + 1
    # print('data process finished')
    print("matr.shape",Matr.shape)
    return Matr, targetY ,all_label




import csv
train_file_name="/storage/yq/DephosSite/dephos_data/train_Y .csv"
max_features=21
win1=31
win2=51
x_train1,y_train,z1 = getMatrixLabel(train_file_name, ('Y'), win1)
x_train2, _,z1 = getMatrixLabel(train_file_name, ('Y'), win2)
perm = np.random.permutation(len(x_train1))
x_train1 = x_train1[perm]
x_train2 = x_train2[perm]
y_train = y_train[perm]
# z1 = z1[perm]






from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
tensorboard = TensorBoard(log_dir='log')
print(tensorboard)
checkpoint = ModelCheckpoint(filepath='/storage/yq/DephosSite/model_multi_713_y_retrainST.h5',monitor='val_loss',mode='min' ,save_best_only='True')
# ROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=0, mode='auto', cooldown=0, min_lr=0)
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
# ROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=0, mode='auto', cooldown=0, min_lr=0)

# 将所有回调函数放入一个列表中
callback_lists = [tensorboard, checkpoint]
# 加载已经训练好的模型
model1 = load_model("/storage/yq/DephosSite/model/mulit_31_51_hn.h5")

# 修改输出层
# 解冻所有层
for layer in model1.layers:
    layer.trainable = True

    
optimizer= Adam(learning_rate=0.000001)
# 编译模型
model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 训练模型
model1.fit([x_train1,x_train2],y_train,batch_size=16,epochs=200,validation_split=0.2,callbacks=callback_lists)

