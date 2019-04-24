#%%
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
#%%
df = pd.read_clipboard()
#%%
df[' 零售码洋 '] = df['零售码洋'].astype('float')
data = df.groupby(['年度', '一级分类','零售财务月' ])['零售码洋'].sum()
#%%
def function(dd,year):
    d = pd.Series()
    if year!='2019年':
        month = [str(i+1)+'月' for i in range(12)]
    else:
        month = [str(i+1)+'月' for i in range(3)]
    for i in month:
        d[i] = dd[i]
    return d
kind = 6
years = ['2017年','2018年','2019年']
plt.figure(figsize=(12,5))
for i in years:
    plt.plot(function(data[i][kind],i),label=i)
plt.legend()
