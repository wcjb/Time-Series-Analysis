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
chunksize = 1e6
ftype = {
    '零售编号': 'category',
    '唯一码': 'category',
    '书号': 'category',
    '零售码洋': 'float32',
    '零售数量': 'float32',
    '四级分类': 'category',
    '零售日期': 'datetime64',
    '站点编号': 'category',
    '出版社': 'category',
    '供应商': 'category',
    '零售实洋': 'float32'}
data = pd.DataFrame()
def function(chunck):
    for key,value in tqdm(ftype.items()):
        chunck[key] = chunck[key].astype(value)
    return chunck

if __name__ == '__main__':
    pool = Pool()
    jobs = []
    
    reader = pd.read_csv(
            'E:/历史销售数据/历史销售数据.csv',
            encoding='utf-8',
            engine='python',
            memory_map=True,
            chunksize=chunksize)
    for chunk in tqdm(reader):
        jobs.append(pool.apply_async(function,args=(chunk,)))
    pool.close()
    pool.join()
    for job in tqdm(jobs):
        # 需要注意的是，这里的sort仅仅只是对合并的表排序而不是对各个表内的数据排序
        data = pd.concat([data, job.get()], axis=0, sort=True)

