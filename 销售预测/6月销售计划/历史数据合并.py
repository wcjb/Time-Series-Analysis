#%%
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD
from tqdm import tqdm
import os
import dask.dataframe as dd
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
Path = ["E:/历史销售数据/2017.csv", "E:/历史销售数据/2018.csv", 'E:/历史销售数据/2019截至0422.csv']
columns = ['零售编号', '唯一码', '书号', '零售码洋', '零售数量', '四级分类',
           '零售日期', '站点编号', '出版社', '供应商', '零售实洋']
lines_in_chunk = 5e5
ftype = {
    columns[0]: 'category',
    columns[1]: 'category',
    columns[2]: 'category',
    columns[3]: 'float32',
    columns[4]: 'float32',
    columns[5]: 'category',
    columns[6]: 'datetime64',
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
          encoding='utf-8',
          engine='python',
          memory_map=True,
          chunksize=lines_in_chunk)
      for chunk in tqdm(reader):
          # 需要注意的是，这里的sort仅仅只是对合并的表排序而不是对各个表内的数据排序
            data = pd.concat([data, chunk],axis=0,sort=True)
      return data
#%%
df = pd.DataFrame()
for i in Path:
    df = pd.concat([df,function(i)])
for key,value in tqdm(ftype.items()):
    df[key] = df[key].astype(value)
df.info(memory_usage='deep')
print('数据文件保存中。。。。。')
df.to_csv('E:/历史销售数据/历史销售数据.csv')

