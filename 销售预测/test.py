# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:54:08 2019

@author: wcjb
"""
import multiprocessing as mp
import os
import time
import uuid
# 忽略Warning级警告
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
from tqdm import tqdm

warnings.filterwarnings('ignore')
# 正确显示图片中的中文字体与负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.figsize']=(12,7)
plt.style.use('ggplot')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# 渲染矢量图
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'

def Scalerone(path):
    """数据导入、清洗，并构造监督学习数据集
    Params
    --------------------------------------------
    path            字符串，便是数据文件所在路径

    Return
    --------------------------------------------
    scaler          列表[max,min]，存放数据集标准化时的最大值和最小值，方便后续还原
    Set             字典，保存数据集及划分后的训练集和测试集
    time_x          数据集转化为监督型数据后的自变量
    time_y          数据集转化为监督型数据后的因变量
    """
    
    df2018 = pd.read_excel(path,sheet_name='2018年')
    df2017 = pd.read_excel(path,sheet_name='2017年')
    data2017 = df2017['现金总实洋']
    data2018 = df2018['现金总实洋']
    data = data2017.append(data2018)
    data.index = range(len(data))
    
    Data = data.dropna().values
    
    Data = Data.reshape(-1,1)
    
    
    scaler = [Data.max(),Data.min()]
    Data = (Data-scaler[1])/(scaler[0]-scaler[1])
        # 训练集
    Data_train =Data[:-30]  
        # 测试集
    Data_test = Data[-30:]
    Data = Data[:-30]
    Data_test = Data[-30:]
    Set = {'数据集':Data,'训练集':Data_train,'测试集':Data_test}
    
    time_x = []
    time_y = []
    for i in range(len(Data)-30):
        time_x.append(Data[i:i+30,0])
        time_y.append(Data[i+30,0])
    time_x = np.array(time_x) 
    time_y = np.array(time_y)
    

    plt.figure(figsize=(14,6))
    plt.subplot(221)
    plt.plot(data,label='原序列')
    plt.legend()
    
    plt.subplot(222)
    plt.plot(data.diff().dropna(),color='green',label='一阶差分序列')
    plt.legend()
    
    plt.subplot(223)
    data.plot.box()
    
    plt.subplot(224)
    data.plot.kde()
    plt.savefig(PATH+'one.jpg')
    return scaler,Set,time_x,time_y

def DNN(setdata,city):
    """使用4层的深度神经网络进行销售预测
    Params
    ---------------------------------------------------------------------
    time_x          数据集转化为监督型数据后的自变量
    time_y          数据集转化为监督型数据后的因变量
    city            当前模型对应城市
    
    Return
    ----------------------------------------------------------------------
    df             DataFrame，模型输出的预测值及预测误差
    loss           DataFrame，模型在训练过程中的loss
    """
    modelpath1 = PATH+'model/'+PREDICT+'/'
    if os.path.exists(modelpath1) == False:
        os.makedirs(modelpath1)
    modelpath = modelpath1 + +CITYUUID[city]+'DNNmodel.h5'
    Data_train = Set['训练集']
    Data_test = Set['测试集']
    Data = Set['数据集']
    time_x = Set['time_x']
    time_y = Set['time_y']
    #device_count={'CPU':6},inter_op_parallelism_threads=12,intra_op_parallelism_threads=12,
    config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    graph = tf.Graph()
    with graph.as_default() as g:
        with tf.Session(config=config,graph=g) as sess:
            #tf.keras.backend.set_session(sess)
            # 查看当前使用的计算图
            #tf.get_default_graph()
            if os.path.exists(modelpath)==True:
                with tf.device(DEVICE):
                    try:
                        model = load_model(modelpath)
                        model.save(modelpath,overwrite=True,include_optimizer=True)
                        history = model.fit(time_x,time_y,epochs=20,batch_size=100,verbose=1)
                    except:
                        print('执行异常！请删除异常退出时所执行模块，当前执行模块为：',CITYUUID[city])
            else:
                with tf.device(DEVICE):
                    try:
                        model = Sequential()
                        model.add(Dense(units=100,activation='relu',input_shape=(30,)))
                        model.add(Dense(units=200,activation='tanh'))
                        model.add(Dense(units=100,activation='relu'))
                        model.add(Dense(units=1))
                        model.compile(loss='mean_squared_logarithmic_error',optimizer='adam')
                        history = model.fit(time_x,time_y,epochs=5000,batch_size=100,verbose=1)
                        model.save(modelpath,overwrite=True,include_optimizer=True)
                    except BaseException as e:
                        print('\n执行异常！请删除异常退出时所执行模块，当前执行模块为：',CITYUUID[city])
            if ISTRAIN == 'T':
                x = Data[-60:-30]
                end = 30
            elif ISTRAIN == 'P':
                x = Data[-30:]
                end = 40
            predict = []
            for i in range(end):
                y_ = model.predict(x.reshape(1,-1))
                predict.append(y_)
                x = np.append(x,y_)
                x = np.delete(x,0)
    df = pd.DataFrame()       
    if ISTRAIN == 'T':
        y1 = [abs(i[0][0]) for i in predict]
        df['城市'] = [city]*30
        df['预测值'] = np.array(list(map(lambda x: x*SCALER[city]['scaler'][0]+(1-x)*SCALER[city]['scaler'][1],np.array(y1))))
        df['实际值'] = np.array(list(map(lambda x: x*SCALER[city]['scaler'][0]+(1-x)*SCALER[city]['scaler'][1],np.array(Data_test))))
        df['预测误差'] = 1-df.预测值/df.实际值
    elif ISTRAIN == 'P':
        y1 = [abs(i[0][0]) for i in predict]
        df['城市'] = [city]*40
        df['预测值'] = np.array(list(map(lambda x: x*SCALER[city]['scaler'][0]+(1-x)*SCALER[city]['scaler'][1],np.array(y1))))
    loss = pd.DataFrame()

    loss['城市'] = [city]*len(history.history['loss'])
    loss['loss'] = history.history['loss']

    return df,loss

def LNN(setdata,city):

    """使用4层的LSTM网络进行销售预测
    Params
    ---------------------------------------------------------------------
    time_x          数据集转化为监督型数据后的自变量
    time_y          数据集转化为监督型数据后的因变量
    city            当前模型对应城市
    
    Return
    ---------------------------------------------------------------------
    df             DataFrame，模型输出的预测值及预测误差
    loss           DataFrame，模型在训练过程中的loss
    """
    modelpath1 = PATH+'model/'+PREDICT+'/'
    if os.path.exists(modelpath1) == False:
        os.makedirs(modelpath1)
    modelpath = modelpath1 +CITYUUID[city]+'LNNmodel.h5'
    # 重新构建图层
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True) 
    Data_train = Set['训练集']
    Data_test = Set['测试集']
    Data = Set['数据集']
    time_x = Set['time_x']
    time_y = Set['time_y']
    graph = tf.Graph()
    with graph.as_default() as g:
        with tf.Session(graph=g,config=config) as sess:
            #tf.keras.backend.set_session(sess) 
            if os.path.exists(modelpath)==True:
                with tf.device(DEVICE):
                    try:
                        model = load_model(modelpath)
                        history = model.fit(time_x.reshape(-1,1,30),time_y.reshape(-1,1),epochs=1,batch_size=100,verbose=0)
                        model.save(modelpath,overwrite=True,include_optimizer=True)
                    except BaseException as e:
                        print('执行异常！请删除异常退出时所执行模块，当前执行模块为：',CITYUUID[city])
            else:
                with tf.device(DEVICE):
                    try:
                        model = Sequential()
                        model.add(LSTM(30,input_shape=(1,30)))
                        model.add(LSTM(40,input_shape=(30,30))
                        model.add(Dense(units=60,activation='relu'))
                        model.add(Dense(units=30,activation='tanh'))
                        model.add(Dense(units=1))
                        model.compile(loss='mse',optimizer='adam')
                        history = model.fit(time_x.reshape(-1,1,30),time_y.reshape(-1,1),epochs=1,batch_size=100,verbose=1)
                        model.save(modelpath,overwrite=True,include_optimizer=True)
                    except BaseException as e:
                        print('执行异常！请删除异常退出时所执行模块，当前执行模块为：',CITYUUID[city])
            if ISTRAIN == 'T':
                x = Data[-60:-30]
                end = 30
            elif ISTRAIN == 'P':
                x = Data[-30:]
                end = 40
            predict = []
            for i in range(end):
                    y_ = model.predict(x.reshape(1,1,30))
                    predict.append(y_)
                    x = np.append(x,y_)
                    x = np.delete(x,0)
    df = pd.DataFrame()       
    if ISTRAIN == 'T':
        y1 = [abs(i[0][0]) for i in predict]
        df['城市'] = [city]*30
        df['预测值'] = np.array(list(map(lambda x: x*SCALER[city]['scaler'][0]+(1-x)*SCALER[city]['scaler'][1],np.array(y1))))        
        df['实际值'] = np.array(list(map(lambda x: x*SCALER[city]['scaler'][0]+(1-x)*SCALER[city]['scaler'][1],np.array(Data_test))))
        df['预测误差'] = 1-df.预测值/df.实际值
    elif ISTRAIN == 'P':
        y1 = [abs(i[0][0]) for i in predict]
        df['城市'] = [city]*40
        df['预测值'] = np.array(list(map(lambda x: x*SCALER[city]['scaler'][0]+(1-x)*SCALER[city]['scaler'][1],np.array(y1))))
    loss = pd.DataFrame()
    loss['城市'] = [city]*len(history.history['loss'])
    loss['loss'] = history.history['loss']
    return df,loss

def log(city):
    with open(PATH+'log.txt','a') as file:
        file.writelines('训练时间：'+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n')
        file.writelines('使用模型：'+ TYPE+'\n')
        file.writelines('使用设备：'+DEVICE+'\n')
        file.writelines('当前执行城市：'+ CITYUUID[city]+'\n')
    return 1

def Scaler(city,params):
    """清洗数据，将每个城市的数据集整理为可供网络学习的监督型数据集
    Params
    ----------------------------------------------------------------------
    city            当前模型对应城市
    params          当前城市对应数据集
    
    Return
    ----------------------------------------------------------------------
    time_x          数据集转化为监督型数据后的自变量
    time_y          数据集转化为监督型数据后的因变量
    city            当前模型对应城市
    """
    Data = params
    Data = Data.reshape(-1,1)
    scaler = [Data.max(),Data.min()]
    Data = (Data-scaler[1])/(scaler[0]-scaler[1])
        # 训练集
    Data_train =Data[:-30]
        # 测试集
    Data_test = Data[-30:]
    Data = Data[:-30]
    Data_test = Data[-30:]
    
    time_x = []
    time_y = []
    for i in range(len(Data)-30):
        time_x.append(Data[i:i+30,0])
        time_y.append(Data[i+30,0])
    time_x = np.array(time_x) 
    time_y = np.array(time_y)
    Set = {'数据集':Data,'训练集':Data_train,'测试集':Data_test,'time_x':time_x,'time_y':time_y}

    SCALER[city] = {}
    SCALER[city]['scaler'] = scaler
    SCALER[city]['set'] = Set
    SCALER[city]['x'] = time_x
    SCALER[city]['y'] = time_y
    
    return Set

def tosave(ts,data):
    count = 1
    path = PATH+"TimeSeries"+'/'+PREDICT+TYPE
    if os.path.exists(path) == False:
        os.makedirs(path)

    ts.index = range(ts.shape[0])
    total = pd.DataFrame(index=ts.index,columns=['日期','站点','营收预测','营收预测修正','营收实际','营收预测误差','提货目标','提货预测','提货预测修正','提货实际','售卡目标','售卡预测','售卡预测修正','售卡实际','售卡预测误差'])
    for i in tqdm(ts.index):
        if  count <=  40:
                total.at[i,'日期'] = (data[data.店名==ts.at[i,'城市']][-1:].日期.values[0])+np.timedelta64(count, 'D')
                total.at[i,'站点'] = ts.at[i,'城市']
                if PREDICT=='现金总实洋':
                    total.at[i,'营收预测'] = ts.at[i,'预测值']
                elif PREDICT=='纯零售+提货实洋':
                    total.at[i,'提货预测'] = ts.at[i,'预测值']
                elif PREDICT=='阅读卡售卡实洋':
                    total.at[i,'售卡预测'] = ts.at[i,'预测值']
                count = count+1
        else:
                count =1
        total.to_excel(path+'Timeseries.xlsx')
    return total

def Graphics(t,h):
    """绘制模型的误差曲线以及模型的预测值与实际值
    Params
    ---------------------------------------------------------------------------------
    t                 DataFrame,模型预测后返回的数据集，包含实际值，模型的预测值，预测误差
    h                 DataFrame,模型训练过程中的loss

    Return
    ----------------------------------------------------------------------------------
    num               int，无异常则返回1反之则执行异常
    """
    savefile = PATH+'Graphics/'+ PREDICT+'/'+TYPE
    cl = list(set(Citylist).difference(set(LESS)))
    if os.path.exists(savefile) == False:
        os.makedirs(savefile)
    if ISTRAIN == 'T':
        for i in tqdm(cl,desc='Graphics保存进度:'):
            time = t[t['城市']==i]
            his = h[h['城市']==i]
            mse = mean_squared_error(time['实际值'].values,time['预测值'].values)
            plt.title(str(mse))
            plt.figure(figsize=(20,9),dpi=100)
            plt.subplot(311)
            plt.plot(time['预测值'],label='预测值')
            plt.plot(time['实际值'],label='实际值')
            plt.legend()

            plt.subplot(312)
            plt.scatter(range(time.shape[0]),time['预测误差'],label='预测误差')
            plt.legend()

            plt.subplot(313)
            plt.plot(his['loss'],label='loss')
            plt.legend()    
            plt.savefig(savefile+'/'+i+'.jpg',dpi=100,quality=95)
    elif ISTRAIN == 'P':
        for j in tqdm(cl,desc='Graphics保存进度：'):
            time = t[t['城市']==j]
            his = h[h['城市']==j]
            predict = time['预测值'].values
            predict = (predict-min(predict))/(max(predict)-min(predict))
            plt.figure(figsize=(20,8))
            plt.plot(range(len(SCALER[j]['set']['数据集'])),SCALER[j]['set']['数据集'],label='历史数据')
            plt.plot(np.arange(len(SCALER[j]['set']['数据集']),len(SCALER[j]['set']['数据集'])+len(predict),1),predict,label='预测数据')
            plt.legend()
            plt.savefig(savefile+'/'+j+'.jpg',dpi=100,quality=95)
    return 0


# 主程序
if __name__ =='__main__':
    # 设置当前执行模式：训练模式(T)及预测模式(P)
    ISTRAIN = 'P'
    # 选择不同模型进行建模
    TYPE = 'LNN'
    # 三种预测对象：现金总实洋，纯零售+提货实洋，阅读卡售卡实洋
    PREDICT = '现金总实洋'
    PATH = 'E:/Learning/销售预测/'
    # 检测本机的可用设备，有GPU则使用GPU进行性训练，反之，使用CPU
    LESS = []
    
    if tf.test.is_gpu_available() == False:
        DEVICE = '/gpu:0'
    else:
        DEVICE = '/cpu:0'
    #Scalerone(PATH+'安顺-增加天气.xlsx')
    # 存放所有门店的标准化指标及相应数据集
    SCALER = {}
    # 存放所有门店训练阶段的误差信息，方便或许调试
    HISTORY = pd.DataFrame()

    data2017 = pd.read_excel(PATH+'输出格式及源数据.xlsx',sheet_name='2017年')
    data2018 = pd.read_excel(PATH+'输出格式及源数据.xlsx',sheet_name='2018年')
    data2019 = pd.read_excel(PATH+'输出格式及源数据.xlsx',sheet_name='2019年')
    
    # 按行拼接三个数据帧
    data = pd.concat([data2017,data2018,data2019],axis=0)
    data.index = range(data.shape[0])
    data.drop(columns=['地区'],axis=1)
 
    data.index = range(data.shape[0])

    Citylist = list(set(data['店名'].values))
    CITYUUID = {}
    ID = 8738249184082438861234567891011
    for j in Citylist:
        CITYUUID[j] = hex(ID)
        ID+=1
    for k in Citylist:
        if len(data[data['店名']==k][PREDICT].values) < 100:
            LESS.append(k)
            [data.drop(index=i,inplace=True) for i in data.index if data.at[i,'店名'] == k]
    timeseries = pd.DataFrame()
    cl = list(set(Citylist).difference(set(LESS)))
    if TYPE == 'DNN':
        for i in tqdm(cl,desc='训练进度'):
            step = data[data['店名']==i][PREDICT].values
            log(i)
            Set = Scaler(i,step)
            df,loss = DNN(Set,i,)
            timeseries = pd.concat([timeseries,df])
            HISTORY = pd.concat([HISTORY,loss])
        tosave(timeseries,data)
    elif TYPE=='LNN':
        for i in tqdm(cl,desc='训练进度'):
            step = data[data['店名']==i][PREDICT].values
            Set = Scaler(i,step)
            log(i)
            df,loss = LNN(Set,i)
            timeseries = pd.concat([timeseries,df])
            HISTORY = pd.concat([HISTORY,loss])
        tosave(timeseries,data)
    Graphics(timeseries,HISTORY)
