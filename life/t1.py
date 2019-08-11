
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import itertools
import warnings
from matplotlib import  font_manager
import seaborn as sns
import sys
import os


import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties


import matplotlib as mpl
# mpl.rcParams['font.sans-serif'] = ['Yahei Mono']
# mpl.rcParams['font.serif'] = ['Yahei Mono']
#
#
#
#
# pd.set_option('display.float_format', lambda x: '%.5f' % x) # pandas
# np.set_printoptions(precision=10, suppress=True) # numpy
#
# pd.set_option('display.max_columns', 100)
# pd.set_option('display.max_rows', 100)

my_font=font_manager.FontProperties(fname=r"c:\windows\fonts\simkai.ttf")




work_day = 'get_data/t1.csv'
w = pd.read_csv(work_day, index_col=0)

# print(work_day.head(10))



plt.figure(figsize=(20,8),dpi=150)
# plt.title("剩余寿命", loc="center", fontsize=20)
plt.xlabel('管道区域标号', fontproperties=my_font)
plt.ylabel('年',fontproperties=my_font)#设置标签
# plt.ylabel('出口压力')
plt.plot(w ,label="剩余寿命")
plt.legend(loc=1,prop=my_font)
plt.show()



