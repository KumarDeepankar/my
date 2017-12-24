import csv
import pandas as pd
from pandas import DataFrame
import re



#"id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"

pd.options.display.float_format = '{:,.0f}'.format
regex = re.compile('[^a-zA-Z\']')

Trainfile = "C:\\Kaggle\\train.txt"

df = DataFrame.from_csv(Trainfile, sep='\t', header=0,index_col=None)
comments = df.iloc[:,:3]
#print(df.info())
linecount = 0
for index, row in comments.iterrows():
    if row['toxic'] == 1:
        #linecount +=1
        #print("-----------New Comment----------------")
        line = re.findall(r'[^"]*', row['comment_text'])
        #print(linecount)
        line = "".join(str(line).split("\\n"))
        line = regex.sub(' ', line)
        line = line.replace("'", "")
        print(line)
        #print(row['comment_text'])
