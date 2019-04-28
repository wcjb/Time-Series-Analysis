#%%
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pysnooper
import time
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
Path = "E:/历史销售数据/HistoryData.csv"
columns = ['零售编号', '唯一码', '书号', '零售码洋', '零售数量', '四级分类',
           '零售日期', '站点编号', '出版社', '供应商', '零售实洋']
lines_in_chunk = 5e6
ftype = {
    columns[0]: 'category',
    columns[1]: 'category',
    columns[2]: 'category',
    columns[3]: 'float32',
    columns[4]: 'float32',
    columns[5]: 'int64',
    columns[6]: 'datetime64',
    columns[7]: 'category',
    columns[8]: 'category',
    columns[9]: 'category',
    columns[10]: 'float32'}

reader = pd.read_csv(
    Path,
    encoding='utf-8',
    engine='python',
    memory_map=True,
    iterator=True
    #chunksize=lines_in_chunk
    )
#%%

def function(data):
    for key, value in ftype.items():
        data[key] = data[key].astype(value)
    return data


if __name__ == '__main__':
    df = pd.DataFrame(columns=columns)
    pool = Pool()
    jobs = []
    for chunk in tqdm(reader):
        jobs.append(pool.apply_async(function, args=(chunk,)))
    pool.close()
    pool.join()
    for job in tqdm(jobs):
        #需要注意的是，这里的sort仅仅只是对合并的表排序而不是对各个表内的数据排序
        df = pd.concat([df, job.get()], axis=0, sort=True)
    df.info(memory_usage='deep')


start = time.time()
loop = True
chunks = []
while loop:
    try:
        chunk = reader.get_chunk(lines_in_chunk)
        chunks.append(chunk)
    except StopIteration:
        print('文件读取结束')
        loop = False
df = pd.concat(chunks, ignore_index=True)
end = time.time()-start
