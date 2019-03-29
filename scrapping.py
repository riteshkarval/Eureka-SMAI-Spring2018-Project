# -*- coding: utf-8 -*-


"""
@author: Ananya Mukherjee
"""

#Program to webscrape the imdb site of movie reviews.
#Retrieve the movie title id from the imdb data set that has been downloaded from kaggle.

import pandas as pd
import numpy as np
import bs4 as bs
import urllib.request
import re

#******************************************************************************
#Code to clean the comment by splitting it into sentences and then passing it to
#the cleaning function
def CommentCleaner(para):
    data = para.split(u"\n")
    cleanPara = []
    for sentance in data:
        clean_sen = TextCleaner(sentance)
        cleanPara.append(clean_sen)
    return cleanPara

#Code to clean the text by replacing the punctuations, numbers, other html tags
def TextCleaner(text):   
    text=re.sub(r'(\d+)',r'',text)
    text=text.replace(u'<div class="text show-more__control">','')
    text=text.replace(u'%','')
    text=text.replace(u'</div>','')
    text=text.replace(u',','')
    text=text.replace(u'"','')
    text=text.replace(u'(','')
    text=text.replace(u')','')
    text=text.replace(u'"','')
    text=text.replace(u':','')
    text=text.replace(u"'",'')
    text=text.replace(u"‘‘",'')
    text=text.replace(u"’’",'')
    text=text.replace(u"''",'')
    text=text.replace(u".",'')
    text=text.replace(u"-",'')
    text=text.replace(u"<br/>",'')
    
    return text
 
#***********************************************************************
#Read the data from .csv file for IMDB Data set
imdbData = pd.read_csv("/Users/Sushom-Dell/Desktop/Ananya/New folder/smai/PROJECT/imdb.csv")
#***********************************************************************
#Converting data into data frame
df = pd.DataFrame(imdbData,columns=imdbData.columns)
#***********************************************************************
#for i in range(len(df['tid'])):
for i in range(100):
    #Frame the url by inserting title id 
    url = 'https://www.imdb.com/title/'+df['tid'][i]+'/reviews'
    source = urllib.request.urlopen(url).read()    
    soup = bs.BeautifulSoup(source,'lxml')
    comments = [] #Array to hold the list of comments for a given movie
    
    for p in soup.find_all('div', class_ = 'text show-more__control'):  
        cleanP = CommentCleaner(str(p)) #Clean the data
        comments.append([cleanP[0]+'/n']) #Store each user comment into an array 'comments'
       
    newfile = df['tid'][i]+".txt"        #File name is titleid.txt
    with open(newfile, "w", encoding='utf-8') as output:  #Creating a new file for each movie
        output.write(str(comments))      #writing the comments into the file.