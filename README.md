Topic of project: Predicting Movie Ratings Based on Reviews

Literature Survey:
 http://aclweb.org/anthology/Y07-1050 : In this paper they have conducted various experiments on different classifiers like SVM, Maximum Entropy and Scoring which resulted in poor performance.  Then they have collaborated all the three models which significantly increased the performance.
 
 
Data Extraction: 

•	Download the IMDB dataset from Kaggle (https://www.kaggle.com/orgesleka/imdbmovies/version/1) <br>
•	Perform Data Cleaning (missing attributes/ misplaced data) <br>
•	From the CSV file we can extract the movie title id for each given movie.<br>
•	Frame a url https://www.imdb.com/title/tt0032976/reviews. 
For example, title id of movie Rebecca is tt0032976. <br>
•	Using web scrapping, we can retrieve the User Reviews section from the above mentioned url. <br>

Problem Categorization:  It is a classification problem. + Regression problem 

Success Metric: Accuracy, Recall, Precision, F1 Score

Feature Extraction: Sentence to Vector/ Paragraph to Vector

Model Selection/ Choice: Based on empirical experiments

Validation and Testing: We can tune the parameters based on the validation set and calculate accuracy on the test data

Expectations
1) Need to implement the research paper (SVM, Maximum Entropy,Scoring) <br>
2) Need to implement using Deep Learning Techniques
3) Need to extend it to Indian Movies
4) Categorize the rating (screenplay, music, action etc..)

Scrapping.py 
This code read the imdb.csv dataset.
It extracts the comments using webscrapping ( using BeautifulSoup, urllib) from the imdb reviews page for each movie.
Stores all the user comments for a given movie into a txt file.
For 14K movies in the imdb data set, it creates 14K text files. 
The sample output is uploaded tt0012349.txt 

