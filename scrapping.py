# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:52:29 2019

@author: Sushom-Dell
"""
import pandas as pd
import numpy as np
import bs4 as bs
import urllib.request

#Read the data from .csv file for IMDB Data set
imdbData = pd.read_csv("/Users/Sushom-Dell/Desktop/Ananya/New folder/smai/PROJECT/imdb.csv")
#***********************************************************************
#Converting data into data frame
df = pd.DataFrame(imdbData,columns=imdbData.columns)
#********************************************************************
#for i in range(len(df['tid'])):
for i in range(100):
    url = 'https://www.imdb.com/title/'+df['tid'][i]+'/reviews'
    source = urllib.request.urlopen(url).read()    
    soup = bs.BeautifulSoup(source,'lxml')
    comments = []
    
    for p in soup.find_all('div', class_ = 'text show-more__control'):
        #print(p)
        comments.append(p.text)
       
    newfile = df['tid'][i]+".txt"       
    with open(newfile, "w", encoding='utf-8') as output:
        output.write(str(comments))
