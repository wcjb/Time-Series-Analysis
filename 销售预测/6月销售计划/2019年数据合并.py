from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
import os

Path = ["E:/历史销售数据/2017.txt", "E:/历史销售数据/2018.txt", 'E:/历史销售数据/2019']
columns = ['零售编号', '唯一码', '书号', '零售码洋', '零售数量', '四级分类',
           '零售日期', '站点编号', '出版社', '供应商', '零售实洋', '未知列']
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

# 2019年数据合并
#%%
path = []
for root, _, files in os.walk(Path[2]):
      for file in files:
            path.append(os.path.join(root, file))
path = [i.replace('\\', '/') for i in path]

#%%
Code_name = {
    'LS_ID': '零售编号',
    'H_ISBN': '书号',
    'H_AMOUNT': '零售数量',
    'H_PRICE': '零售码洋',
    'PUB_NAME': '出版社',
    'STATION_ID': '站点编号',
    'LS_DATETIM': '零售日期',
    'H_ID': '唯一码',
    'FACTORY_NA': '供应商',
    'H_TYPE': '四级分类',
    'NOTAX_REAL': '零售实洋'}
# 互换字典的键值
Code_name = dict([values, key] for key, values in Code_name.items())
#%%


def Read_excel(path):
      df = pd.DataFrame()
      data = pd.read_excel(path)
      for i in columns[:-1]:
            df[i] = data[Code_name[i]]
      return df


#%%
if __name__ == '__main__':
      pool = Pool()
      step = []
      df = pd.DataFrame()
      for i in path:
            step.append(pool.apply_async(Read_excel, args=(i,)))
      pool.close()
      pool.join()
      for job in tqdm(step, desc='数据读取进度:'):
            df = pd.concat([df, job.get()])
      print('文件保存中。。。。。')
      df.to_csv('E:/历史销售数据/2019截至0422.csv',index=False)
