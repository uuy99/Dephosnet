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
from tensorflow.keras.models import load_model
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



def calc(TN, FP, FN, TP):
    SN = TP / (TP + FN)  # recall
    SP = TN / (TN + FP)
    # Precision = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    F1 = (2 * TP) / (2 * TP + FP + FN)
    fz = TP * TN - FP * FN
    fm = (TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)
    MCC = fz / pow(fm, 0.5)
    PRE=TP/(TP+FP)
    return SN, SP, ACC, MCC,F1,PRE





test_file_name="/storage/yq/DephosSite/dephos_data/test_Y.csv"
win1=31
win2=51
win3=71
[x_test1,y_test,z] = getMatrixLabel(test_file_name, ('Y'), win1)
[x_test2,_,_] = getMatrixLabel(test_file_name, ('Y'), win2)
[x_test3,_,_] = getMatrixLabel(test_file_name, ('Y'), win3)
perm = np.random.permutation(len(x_test1))
x_test1 = x_test1[perm]
x_test2 = x_test2[perm]
y_test = y_test[perm]



model = load_model("/storage/yq/DephosSite/model/mulit_31_51_hn.h5")
y_pre_test = model.predict([x_test1,x_test2])
tn, fp, fn, tp  = confusion_matrix(y_test.argmax(axis=1),y_pre_test.argmax(axis=1)).ravel()
sn, sp, acc, mcc,f1,Pre= calc(tn, fp, fn, tp)
from sklearn.metrics import roc_auc_score
AUC=roc_auc_score(y_test[:,1], y_pre_test[:,1])
sn, sp, acc, mcc,f1,Pre,AUC