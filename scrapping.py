
#Program to webscrape the imdb site of movie reviews.
#Retrieve the movie title id from the imdb data set that has been downloaded from kaggle.

import pandas as pd
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

def ratingCleaner(rating):
    text=re.sub(r'\n',r'',rating)
    text = text.split(u"/")
    return text[0]

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

imdbData=pd.read_csv("./imd.csv")

#***********************************************************************
#Converting data into data frame
df = pd.DataFrame(imdbData, columns=imdbData.columns)
#***********************************************************************
comments = []  # Array to hold the list of comments for a given movie
ratings = []
tids = []

for i in range(250):
    print(i)
    #Frame the url by inserting title id 
    url = 'https://www.imdb.com/title/'+df['tid'][i]+'/reviews'
    source = urllib.request.urlopen(url).read()    
    soup = bs.BeautifulSoup(source, 'lxml')

    for p in soup.find_all('div', class_ = 'lister-item-content'):
        comment = p.find('div', class_='text show-more__control')
        cleanP = CommentCleaner(str(comment)) #Clean the data
        comments.append(cleanP[0].lower())
        rating = p.find('span', class_='rating-other-user-rating')
        userRating = '0'
        tids.append(df['tid'][i])
        if rating != None:
            userRating = ratingCleaner(rating.text)
        ratings.append(userRating)
dfm = pd.DataFrame({'TID':tids , 'Comments': comments , 'Rating' : ratings})
newfile = "./dataset.csv"
dfm.to_csv(newfile,index=False)
