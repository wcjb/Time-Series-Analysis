import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.models import Sequential,load_model
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras import backend 
from sklearn.externals import joblib 
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
from datetime import datetime
from tensorflow.keras import regularizers
# import warnings
# warnings.filterwarnings('ignore')
class AdaBoost:

    CITY,CITY_MORE,CITY_LESS,SCALER = {},{},{},{}
    def __init__(self):
        self.TYPE = ['LSTM','DNN','RandomForst','GM']
        self.DEVICE = '/cpu:0'
        self.FIRSTEPOCH = 200
        self.GOONEPOCH = 100
        self.BATCH_SIZE = 45
        self.PATH = 'E:/Learning/销售预测/New/'
        self.VERBOSE = 1
        self.DIM = 42
        #三种预测对象：现金总实洋，纯零售+提货实洋，阅读卡售卡实洋
        self.PREDICT = '纯零售+提货实洋'
        self.ISTRAIN = True
        #模型复杂程度在模型损失中所占比重
        self.RATE = 1e-3
       
        
    def Parser(self):
        global CITY,CITY_MORE,CITY_LESS,SCALER
        df2017 = pd.read_excel(self.PATH+'输出格式及源数据.xlsx',sheet_name='2017年')
        df2018 = pd.read_excel(self.PATH+'输出格式及源数据.xlsx',sheet_name='2018年')
        df2019 = pd.read_excel(self.PATH+'输出格式及源数据.xlsx',sheet_name='2019年')
        data = pd.concat([df2017,df2018,df2019],axis=0)
        data.index = range(data.shape[0])
        CITY_MORE = list(set(data['店名'].values))
        CITY_LESS = []
        for k in CITY_MORE:
            if len(data[data['店名']==k][self.PREDICT].values) < 126:
                CITY_LESS.append(k)
                [data.drop(index=i,inplace=True) for i in data.index if data.at[i,'店名'] == k]
        CITY = list(set(CITY_MORE).difference(set(CITY_LESS)))
        SCALER = {}
        for  i in CITY:
            Data = data[data['店名']==i][self.PREDICT]
            Data = Data.values.reshape(-1,1)
            scaler = [Data.max(),Data.min()]
            Set = {}
            Set['标准化'] = scaler
            Data = (Data-scaler[1])/(scaler[0]-scaler[1])
            
            Set['首项'] = Data[0]
            #Data = pd.Series(Data.squeeze()).diff().dropna().values
           
            Data_train = Data[:-2*self.DIM]
            Data_validation = Data[-2*self.DIM:self.DIM]
            Data_test = Data[-self.DIM:]
            time_x = []
            time_y = []
            if self.ISTRAIN == True:
                for j in range(Data.shape[0]-2*self.DIM):
                    time_x.append(Data[j:j+self.DIM,0])
                    time_y.append(Data[j+self.DIM:j+2*self.DIM,0])
                time_x = np.array(time_x)
                time_y = np.array(time_y)
                train_x = time_x[:-2]
                train_y = time_y[:-2]
                validation_x = time_x[-2:-1]
                validation_y = time_y[-2:-1]
                
                Set['原数据(array)'] = data[data['店名']==i][self.PREDICT]
                Set['time_x'] = train_x
                Set['time_y'] = train_y
                Set['validation_x'] = validation_x
                Set['validation_y'] = validation_y
                Set['测试集'] = Data_test
                Set['原数据(DataFrame)'] = data
            else:
                for j in range(Data.shape[0]-2*self.DIM):
                    time_x.append(Data[j:j+self.DIM,0])
                    time_y.append(Data[j+self.DIM:j+2*self.DIM,0])
                time_x = np.array(time_x) 
                time_y = np.array(time_y)
                
                validation_x = time_x[-1:]
                validation_y = time_y[-1:]
                
                Set['原数据(array)'] = data[data['店名']==i][self.PREDICT]
                Set['time_x'] = time_x
                Set['time_y'] = time_y
                Set['validation_x'] = validation_x
                Set['validation_y'] = validation_y
                Set['测试集'] = Data_test
                Set['原数据(DataFrame)'] = data
            SCALER[i] = Set
        print('数据提取完毕！')
        return SCALER

    def Lstm(self,mold,timeseries):
        '''实现弱分类器LSTM
        ===========================================================================
        @mold:string,模型所对应门店
        @timesseries:dict，包含训练集和测试集的字典
        ---------------------------------------------------------------------------
        @return:返回生成的模型
        ============================================================================
        '''
        lstm_path_01 = self.PATH + 'LSTM/'
        if os.path.exists(lstm_path_01)==False:
            os.makedirs(lstm_path_01)
        lstm_path_02 = lstm_path_01 + mold + '.h5'
        regularizer = regularizers.l2(self.RATE)
        validation_data = (timeseries['validation_x'].reshape(-1,1,self.DIM),timeseries['validation_y'].reshape(-1,self.DIM))
        tf.reset_default_graph()
        graph = tf.Graph()
        config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=False)
        with graph.as_default() as G:
            with tf.Session(graph=G,config=config) as sess:
                backend.set_session(sess)
                if os.path.exists(lstm_path_02):
                    with tf.device(self.DEVICE):
                        try:
                            lstm_model = load_model(lstm_path_02,custom_objects=None,compile=True)
                            lstm_history = lstm_model.fit(timeseries['time_x'].reshape(-1,1,self.DIM),timeseries['time_y'].reshape(-1,self.DIM),shuffle=False,validation_data=validation_data,epochs=self.GOONEPOCH,batch_size=self.BATCH_SIZE,verbose=self.VERBOSE)
                            lstm_model.save(lstm_path_02,overwrite=True,include_optimizer=True)
                            return lstm_model
                        except BaseException as exception:
                            print(exception,mold)
                else:
                    with tf.device(self.DEVICE):
                        try:
                            lstm_model = Sequential()
                            lstm_model.add(LSTM(210,kernel_regularizer=regularizer,bias_regularizer=regularizer,input_shape=(1,self.DIM)))
                            lstm_model.add(Dense(units=77,kernel_regularizer=regularizers.l2(self.RATE),bias_regularizer=regularizers.l2(self.RATE),activation='relu'))
                            lstm_model.add(Dense(units=self.DIM,kernel_regularizer=regularizer,bias_regularizer=regularizer,activation='tanh'))
                            lstm_model.compile(loss='mean_squared_error',optimizer='adam')
                            lstm_history= lstm_model.fit(timeseries['time_x'].reshape(-1,1,self.DIM),timeseries['time_y'].reshape(-1,self.DIM),shuffle=False,validation_data=validation_data,epochs=self.FIRSTEPOCH,batch_size=self.BATCH_SIZE,verbose=self.VERBOSE)
                            lstm_model.save(lstm_path_02,overwrite=True,include_optimizer=True)
                            return lstm_model
                        except BaseException as exception:
                            print(exception,mold)
        
    def DNN(self,mold,timeseries):
        '''实现弱分类器:DNN
        ===========================================================================
        @mold:string,模型所对应门店
        @timesseries:dict，包含训练集和测试集的字典
        ---------------------------------------------------------------------------
        @return:返回生成的模型
        ============================================================================
        '''
        dnn_path_01 = self.PATH + 'DNN/'
        if os.path.exists(dnn_path_01)==False:
            os.makedirs(dnn_path_01)
        dnn_path_02 = dnn_path_01 + mold + '.h5'
        validation_data = (timeseries['validation_x'].reshape(-1,self.DIM),timeseries['validation_y'].reshape(-1,self.DIM))
        regularizer = regularizers.l2(self.RATE)
        tf.reset_default_graph()                              
        graph = tf.Graph()
        config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
        with graph.as_default() as G:
            with tf.Session(graph=G,config=config) as sess:
                backend.set_session(sess)
                if os.path.exists(dnn_path_02)==True:
                    with tf.device(self.DEVICE):
                        try:
                            dnn_model = load_model(dnn_path_02,custom_objects=None,compile=True)
                            dnn_history = dnn_model.fit(timeseries['time_x'].reshape(-1,self.DIM),timeseries['time_y'].reshape(-1,self.DIM),shuffle=False,validation_data=validation_data,epochs=self.GOONEPOCH,batch_size=self.BATCH_SIZE,verbose=self.VERBOSE)
                            dnn_model.save(dnn_path_02,overwrite=True,include_optimizer=True)
                            return dnn_model
                        except BaseException as exception:
                            print(exception,mold)
                else:
                    with tf.device(self.DEVICE):
                        try:
                            dnn_model = Sequential()
                            dnn_model.add(Dense(units=119,kernel_regularizer=regularizer,bias_regularizer=regularizer,input_shape=(self.DIM,)))
                            dnn_model.add(Dense(units=77,kernel_regularizer=regularizer,bias_regularizer=regularizer,activation='relu'))
                            dnn_model.add(Dense(units=self.DIM,kernel_regularizer=regularizer,bias_regularizer=regularizer,activation='tanh'))
                            dnn_model.compile(loss='mean_squared_error',optimizer='adam')
                            dnn_history = dnn_model.fit(timeseries['time_x'].reshape(-1,self.DIM),timeseries['time_y'].reshape(-1,self.DIM),shuffle=False,validation_data=validation_data,epochs=self.FIRSTEPOCH,batch_size=self.BATCH_SIZE,verbose=self.VERBOSE)
                            dnn_model.save(dnn_path_02,overwrite=True,include_optimizer=True)
                            return dnn_model
                        except BaseException as exception:
                            print(exception,mold)
        
    def RandomForest(self,mold,timeseries):
        '''实现弱分类器:随机森林
        ===========================================================================
        @mold:string,模型所对应门店
        @timesseries:dict，包含训练集和测试集的字典
        @bestparams:dict,网格搜索的得到最佳参数集合
        ---------------------------------------------------------------------------
        @return:返回生成的模型
        ============================================================================
        '''
        rf_path_01 = self.PATH + 'RANDOMFOREST/'
        if os.path.exists(rf_path_01)==False:
            os.makedirs(rf_path_01)
        rf_path_02 = rf_path_01 + mold + '.pkl'

        if os.path.exists(rf_path_02)==True:
            rf_model = joblib.load(rf_path_02)
            return rf_model
        else:
            rf_model = RandomForestRegressor(n_estimators=128,max_depth=14,
            min_samples_split=2,min_samples_leaf=1,n_jobs=-1,verbose=self.VERBOSE)
            rf_model.fit(timeseries['time_x'].reshape(-1,self.DIM),timeseries['time_y'].reshape(-1,self.DIM))
            joblib.dump(rf_model,rf_path_02)
            return rf_model
    
    def Gm(self,mold,timeseries,params):
        '''实现弱分类器:灰色预测
        ===========================================================================
        @mold:string,模型所对应门店
        @timesseries:dict，包含训练集和测试集的字典
        @params:int，所使用的灰色预测模型
        ---------------------------------------------------------------------------
        @return:返回生成的模型
        ============================================================================
        '''
        return 0

    def adaboost(self,mold,timeseries):
        '''使用几个弱分类器训练成一个强学习器
        '''
        adaboost_path = self.PATH+'adaboost/'
        if os.path.exists(adaboost_path)==False:
            os.makedirs(adaboost_path)
        adaboost_path_02 = adaboost_path+mold+'.h5'
        regularizer = regularizers.l2(self.RATE)
        self.Lstm(mold,timeseries)
        self.DNN(mold,timeseries)
        self.RandomForest(mold,timeseries)
        tf.reset_default_graph()
        graph = tf.Graph()
        config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
        with graph.as_default() as G:
            with tf.Session(graph=G,config=config) as sess:
                backend.set_session(sess)
                
                lstm = load_model(self.PATH + 'LSTM/'+mold+'.h5',custom_objects=None,compile=True)
                dnn = load_model(self.PATH + 'DNN/'+mold+'.h5',custom_objects=None,compile=True)
                rf = joblib.load(self.PATH + 'RANDOMFOREST/'+mold+'.pkl')
                
                lstm_output_01 = lstm.predict(timeseries['time_x'].reshape(-1,1,self.DIM))
                dnn_output_01 = dnn.predict(timeseries['time_x'].reshape(-1,self.DIM))
                rf_output_01 = rf.predict(timeseries['time_x'].reshape(-1,self.DIM))
                adaboost_input_01 = np.concatenate((lstm_output_01,dnn_output_01,rf_output_01),axis=1)
                
                lstm_output_02 = lstm.predict(timeseries['测试集'].reshape(-1,1,self.DIM))
                dnn_output_02 = dnn.predict(timeseries['测试集'].reshape(-1,self.DIM))
                rf_output_02= rf.predict(timeseries['测试集'].reshape(-1,self.DIM))
                adaboost_input_02= np.concatenate((lstm_output_02,dnn_output_02,rf_output_02),axis=1)
                
                
                adaboost_validation_x = np.array(timeseries['validation_x'].tolist()[0]*3).reshape(-1,3*self.DIM)
                adaboost_validation_y = np.array(timeseries['validation_y'].tolist()[0]).reshape(-1,self.DIM)
          
                validation_data = (adaboost_validation_x,adaboost_validation_y)
                if os.path.exists(adaboost_path_02)==True:
                    with tf.device(self.DEVICE):
                        try:
                            adaboost_model = load_model(adaboost_path_02,custom_objects=None,compile=True)
                            adaboost_history = adaboost_model.fit(adaboost_input_01.reshape(-1,3*self.DIM),timeseries['time_y'].reshape(-1,self.DIM),shuffle=False,validation_data=validation_data,epochs=self.GOONEPOCH,batch_size=self.BATCH_SIZE,verbose=self.VERBOSE)
                            adaboost_model.save(adaboost_path_02,overwrite=True,include_optimizer=True)
                        except BaseException as exception:
                            print(exception,mold)
                else:
                    with tf.device(self.DEVICE):
                        try:
                            adaboost_model = Sequential()
                            adaboost_model.add(Dense(units=84,kernel_regularizer=regularizer,bias_regularizer=regularizer,input_shape=(3*self.DIM,),activation='relu'))
                            adaboost_model.add(Dense(units=self.DIM,kernel_regularizer=regularizer,bias_regularizer=regularizer,activation='tanh'))
                            adaboost_model.compile(loss='mean_squared_error',optimizer='adam')
                            adaboost_history =adaboost_model.fit(adaboost_input_01.reshape(-1,3*self.DIM),timeseries['time_y'].reshape(-1,self.DIM),shuffle=True,validation_data=validation_data,epochs=self.FIRSTEPOCH,batch_size=self.BATCH_SIZE,verbose=self.VERBOSE)
                            adaboost_model.save(adaboost_path_02,overwrite=True,include_optimizer=True)
                        except BaseException as exception:
                            print(exception,mold)
                adaboost_output = adaboost_model.predict(adaboost_input_02)
                
                if self.ISTRAIN == True:
                    error = {}
                    error['4周误差'] = sum(adaboost_output)/sum(timeseries['测试集'].squeeze())
                    error['日误差'] = adaboost_output.squeeze()/(timeseries['测试集'].squeeze())-1
                    error['Loss'] = adaboost_history.history['loss']
                    error['val_loss'] = adaboost_history.history['val_loss']
                else:
                    error = {}
                    error['Loss'] = adaboost_history.history['loss']
                    error['val_loss'] = adaboost_history.history['val_loss']
        return adaboost_output,error

    def Reduction(self,mold,timeseries):
        if False:
            np.insert(timeseries,0,SCALER[mold]['首项'].squeeze())
            reduction_output_01 = pd.Series(np.array(timeseries[0])).cumsum(axis=0)
            reduction_output = (reduction_output_01)*SCALER[mold]['标准化'][0]+(1-reduction_output_01)*SCALER[mold]['标准化'][1]
        else:
            reduction_output = (timeseries)*SCALER[mold]['标准化'][0]+(1-timeseries)*SCALER[mold]['标准化'][1]
        return np.array(reduction_output)

    def Graphics(self,mold,lis):

        if self.ISTRAIN == True:
            graphics_path = self.PATH+'训练/'
        else:
            graphics_path = self.PATH+'训练/'
        if os.path.exists(graphics_path) == False:
            os.makedirs(graphics_path)
        
        if self.ISTRAIN == True:
            plt.figure(figsize=(12,5),dpi=100)
            plt.title(lis[1]['日误差'])
            plt.subplot(411)
            plt.plot(lis[0].squeeze(),label='预测值')
            plt.plot(SCALER[mold]['原数据(array)'][-42:].squeeze().values,label='实际值')
            plt.legend()
            plt.subplot(412)
            plt.plot(lis[0].squeeze()-SCALER[mold]['原数据(array)'][-42:].squeeze().values,label='残差')
            plt.legend()
            plt.subplot(413)
            plt.scatter(range(self.DIM),lis[1]['日误差'],label='单日误差')
            plt.legend()
            plt.subplot(414)
            plt.plot(lis[1]['Loss'],label='训练误差')
            plt.plot(lis[1]['val_loss'],label='验证误差')
            plt.legend()
            plt.savefig(graphics_path+mold+'png',dpi=100)
        else:
            plt.figure(figsize=(12,5),dpi=100)
            plt.subplot(211)
            plt.plot(lis[0].squeeze(),label='预测值')
            plt.legend()
            plt.subplot(212)
            plt.plot(lis[1]['Loss'],label='损失函数')
            plt.plot(lis[1]['val_loss'],label='验证误差')
            plt.legend()
            plt.savefig(graphics_path+mold+'png',dpi=100)
   
    def Save(self,mold,timeseries):
        count = 0
        
        #columns=['日期','站点','营收预测','营收预测修正','营收实际','营收预测误差','提货目标','提货预测','提货预测修正','提货实际','售卡目标','售卡预测','售卡预测修正','售卡实际','售卡预测误差']
       
        total = pd.DataFrame()
        data = SCALER[mold]['原数据(DataFrame)']
        if self.ISTRAIN == False:
            for i,j in enumerate(timeseries[0]):
                
                total.at[i,'日期'] = (data[data.店名==mold][-1:].日期.values[0])+np.timedelta64(1+(count%42), 'D')
            
                count = count+1
                total.at[i,'站点'] = mold
                if self.PREDICT=='现金总实洋':
                    total.at[i,'营收预测'] = j
                elif self.PREDICT=='纯零售+提货实洋':
                    total.at[i,'提货预测'] = j
                elif self.PREDICT=='阅读卡售卡实洋':
                    total.at[i,'售卡预测'] = j
            
        else:
            for i,j in enumerate(timeseries[0]):
                total.at[i,'日期'] = (data[data.店名==mold][-1:].日期.values[0])+np.timedelta64(1+(count%42), 'D')
                count = count+1
                total.at[i,'站点'] = mold
                total.at[i,'预测值'] = j
            total['实际值'] = SCALER[mold]['原数据(array)'][-42:].values
            total.index = range(total.shape[0])
            
        return total


if __name__ == '__main__':
    ada = AdaBoost()
    Data = ada.Parser()
    df = pd.DataFrame()
    for city in tqdm(CITY[:101]):
        adboost_output,error = ada.adaboost(city,Data[city])
        reduction_output= ada.Reduction(city,adboost_output)
        ada.Graphics(city,[reduction_output,error])
        df = pd.concat([df,ada.Save(city,reduction_output)],axis=0)
        save_path = ada.PATH+"TimeSeries"+'/'+ada.PREDICT+'/'
        if os.path.exists(save_path) == False:
             os.makedirs(save_path)
        if ada.ISTRAIN == True:
             kind = '训练'
        else:
             kind = '预测'
        df.to_excel(save_path+kind+'timeseries.xlsx')
