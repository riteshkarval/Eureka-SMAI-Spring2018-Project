#Program to webscrape the imdb site of movie reviews.
#Retrieve the movie title id from the imdb data set that has been downloaded from kaggle.
#Based on the movie title id, frame url and retrieve the user reviews and ratings for each movie.

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

#Code to retrieve the rating of each comment
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
imdbData = pd.read_csv("imdb.csv")
#***********************************************************************
#Converting data into data frame
df = pd.DataFrame(imdbData,columns=imdbData.columns)
#***********************************************************************
#for i in range(len(df['tid'])): #Uncomment this to make the code run for all the movie in the imdb dataset.
for i in range(1): #Code runs for 1 movie. 
    #Frame the url by inserting title id 
    url = 'https://www.imdb.com/title/'+df['tid'][i]+'/reviews'
    source = urllib.request.urlopen(url).read()    
    soup = bs.BeautifulSoup(source,'lxml')
    comments = [] #Array to hold the list of comments for a given movie
    #Get the data in the div of class 'lister-item-content'
    #It contains the review details
    for p in soup.find_all('div', class_ = 'lister-item-content'):
        
        #Extract the comment from the review details
        comment = p.find('div', class_='text show-more__control')
        cleanP = CommentCleaner(str(comment)) #Clean the comment data
        
        #Extract the rating given against each comment by the user.
        rating =  p.find('span', class_='rating-other-user-rating')
        userRating = '0' #Initialize rating to zero
        
        if rating != None: #If user has given the rating then retrieve else leave it as 0
            userRating = ratingCleaner(rating.text)
        #Append the comment and rating pair and store it into an array
        comments.append([cleanP[0],userRating]) 
    review[df['tid'][i]] = comments
    
# Store data (serialize)
with open('dataset/movieReviews.pkl','wb') as handle:  
    pickle.dump(review, handle, protocol=pickle.HIGHEST_PROTOCOL)
