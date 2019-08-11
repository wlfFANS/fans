import pandas as pd



data = pd.read_csv('get_data/newdf1-1A.csv' ,header=0,usecols=[0 ,1 ,2 , 3, 4,5 , 6, 7, 8] ,engine='python' ,index_col=0, parse_dates=[0])
print(data.head())
print(data.info())
print((len(data)//10)*10+7)
