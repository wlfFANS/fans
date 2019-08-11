from matplotlib import pyplot as plt
from matplotlib import font_manager

my_font=font_manager.FontProperties(fname=r"c:\windows\fonts\simkai.ttf")


a = ["DBN+SVM","SVM","LightGBM","Keras模型"]

b=[0.97, 0.94, 0.63,0.90]

plt.figure(dpi=100)

plt.bar(range(len(a)),b,width=0.3)
# plt.xlabel("模型类型")
plt.ylabel("F1-SCORE")
plt.xticks(range(len(a)),a,fontproperties=my_font,rotation=0)

plt.show()