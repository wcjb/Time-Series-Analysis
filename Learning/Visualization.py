#%%
import plotly_express as px
import pandas as pd
%config InlineBackend.figure_format = 'svg'
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']  = False
plt.style.use('ggplot')
df = pd.read_excel('C:/Users/wcjb/Desktop/进销存退总表（新）.xlsx',header=1)
#%%
data = pd.pivot_table(df,index='一级分类',columns=['年度','零售财务月'])['零售码洋']

data2017 = data['2017年'][[ '1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月','9月','10月', '11月', '12月']]
data2018 = data['2018年'][[ '1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月','9月','10月', '11月', '12月']]
data2019 = data['2019年'][[ '1月', '2月', '3月']]

#%%
def scale(data,kind):
    data = data.T
    data = (data-data.mean())/data.std()
    return data[kind]
k = 1
plt.figure(figsize=(10,6))
plt.plot(scale(data2017,k),label='2017年')
plt.plot(scale(data2018,k),label='2018年')
plt.plot(scale(data2019,k),label='2019年')
plt.legend()
# 进行本征模态分解
#%%
from PyEMD import EMD
emd = EMD()
imfs = emd(scale(data2017,1))
#%%
for i in range(imfs.shape[0]):
    plt.subplot(4,1,i+1)
    plt.plot(imfs[i],label='IMF')
    plt.plot(scale(data2017,k),label='原始曲线')
    if i==1:
        plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    if i!=imfs.shape[0]-1:
        plt.xticks([])
plt.figure()
plt.plot(imfs[0]+imfs[1],label='分量')
plt.plot(scale(data2017,k),label='原始曲线')
plt.legend()
#%%
imfs = emd(scale(data2018,1))
for i in range(imfs.shape[0]):
    plt.subplot(4,1,i+1)
    plt.plot(imfs[i],label='IMF')
    plt.plot(scale(data2018,k),label='原始曲线')
    if i==1:
        plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    if i!=imfs.shape[0]-1:
        plt.xticks([])
plt.figure()
plt.plot(imfs[0]+imfs[1],label='分量')
plt.plot(scale(data2018,k),label='原始曲线')
plt.legend()

#%%
#random.choice 
emd.FIXE = 10
imfs = emd(scale(data2018,1))
