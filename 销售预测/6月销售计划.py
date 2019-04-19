#%%
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD
from tqdm import tqdm
import dask.dataframe as dd
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
import multiprocessing 
#%%
# 使用多进程读取文件并进行数据清洗
# 读取2017和2018年的历史销售数据
Path = ["E:/历史销售数据/2017.txt","E:/历史销售数据/2018.txt"]
data = []
l = 0
pool = multiprocessing.Pool()
def function(fileline):
      l += 1
      if fileline.find('\\;')!=-1:
            return fileline.split(';',11)
      return fileline


if __name__ == '__main__':
      step = []   
      columns = ['零售单号','唯一码','书号','码洋','零售数量','四级分类','零售日期','站点','出版社','供应商','零售实洋','未知']
      df = pd.DataFrame(columns=columns)    
      with open(Path[0],'r+',encoding='utf-8') as files:
            print(l)
            for file in tqdm(files):
                  step.append(pool.apply_async(function,args=(file)))
            pool.close()
            pool.join()
            for job in tqdm(step,"数据提取进度:"):
                  df =  pd.concat([df,pd.Series(job.get())],axis=1)


