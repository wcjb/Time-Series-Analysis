#%%
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD
from tqdm import tqdm
import os
import dask.dataframe as dd
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
from multiprocessing import Pool
Path = ["E:/历史销售数据/2017.txt", "E:/历史销售数据/2018.txt", 'E:/历史销售数据/2019']
columns = ['零售编号', '唯一码', '书号', '零售码洋', '零售数量', '四级分类',
           '零售日期', '站点编号', '出版社', '供应商', '零售实洋', '未知列']
lines_in_chunk = 5e5
ftype = {
    columns[0]: 'category',
    columns[1]: 'category',
    columns[2]: 'category',
    columns[3]: 'float32',
    columns[4]: 'float32',
    columns[5]: 'category',
    columns[6]:'datetime64',
    columns[7]: 'category',
    columns[8]: 'category',
    columns[9]: 'category',
    columns[10]: 'float32'
}
#%%
def function(path):
      data = pd.DataFrame(columns=columns)
      reader = pd.read_csv(
            path, 
            sep=';',
            encoding='utf-8',
            engine='python', 
            error_bad_lines=False, 
            warn_bad_lines=False,
            memory_map=True,
            chunksize=lines_in_chunk)
      for chunk in tqdm(reader):
            chunk.columns = columns
            data = pd.concat([data,chunk])
      data['零售日期'] = pd.to_datetime(data['零售日期'], format='%Y/%m/%d')
      data.drop(columns='未知列',inplace=True)
      return data

def Memory(df):
      # 输出每种数据类型的平均内存占用
      for dtype in ['float', 'int', 'category','datetime64']:
            # 将数据帧按数据类型分类
            selected_dtype = df.select_dtypes(include=[dtype])
            mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
            mean_usage_mb = mean_usage_b / 1024 ** 2
            print("平均内存占用 {} 类型: {:03.2f} MB".format(
                  dtype, mean_usage_mb))
      return mean_usage_mb

def function_drop(path):
      drop = []
      with open(path,'r',encoding='utf-8') as file:
            for line in file:
                  if len(line.split(';'))==13:
                        drop.append(line.split(';',12))
      dropdf = pd.DataFrame(drop)
      dropdf.drop(columns=10,inplace=True)
      dropdf.columns = columns
      dropdf.drop(columns='未知列', inplace=True)
      return dropdf

def Merge(path):
      data = function(path)
      print('格式正确数据已提取完毕！')
      dropdf = function_drop(path)
      print('格式错误数据已清洗完毕！')
      datastep = pd.concat([data,dropdf])
      # for i in tqdm(ftype):
      #       datastep[i] = datastep[i].astype(ftype[i])
      print('正在将数据保存到本地。。。。。')
      if path[-8:-4] == '2017':
            outpath = path[:-8]+'2017.csv'
      else:
            outpath = path[:-8]+'2018.csv'
      datastep.to_csv(outpath,index=False)
      datastep.info(memory_usage='deep')
      print('已合并数据集，并保存！')
      return datastep  
if __name__ == '__main__':
      for i in range(2):    
            if i==0:
                  print('正在处理2017年数据:') 
            else:
                  print('正在处理2018年数据:')
            data = Merge(Path[i])
