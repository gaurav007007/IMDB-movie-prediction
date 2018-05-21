# IMDB-movie-prediction


# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


pd.set_option('display.max_rows', 5000)
dataframe_movies = pd.read_csv("E:\gaurav_back up\Python_Rang\IMDB rating of a movie\movie_metadata.csv")
dataframe_movies.head()
# We have NAN values. We have to handle them because our mathematical model do not work with NAN values,blank values and duplicated values


# In[3]:


# data cleaning
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
#Checking  duplicate rows
dataframe_movies.duplicated()
#Deleting duplicate rows
dataframe_movies_without_duplicates=dataframe_movies.drop_duplicates()
# removing NAN
df_movies_final=dataframe_movies_without_duplicates.dropna(axis=0)


# In[4]:


# Lets start with quantitative data first by using describe function
df_movies_final.describe(
)
# Quite messy. Lets visualize it by observing the frequency distribution.


# In[5]:


# ratings is important for any movie. So lets start with rating fequency distribution.
# Learn PEP8 
rating_Disribution=df_movies_final["imdb_score"]


# In[6]:


plt.figure()
rating_Disribution.plot.hist()
#So maximum ratings are in the range of 6 to 8


# In[7]:


# Now we will do feature engineering
# The genres has so many terms, we can extract the first one from to determine the movie genre and add a new column.
x=df_movies_final['genres']
# the first word is the genre of the movie

import re
df =pd.DataFrame(x,columns=['genres'])
df['genre']=df['genres'].str.extract('([A-Z]\w{0,})',expand =True)
#df['genre']
#adding new feature to the dataframe.
df_movies_final['genres']=df['genre']
#df_movies_final['genres']


# In[8]:


#Approach ahead
#Step 1
#a there ae many questions.
#b  Maximum genere of which movies belong to,.pie chart
#c Movies vs genre, scatter plot
#d  Is director facebook rating impact movie?
#e Is actor's facebook like imapct movie?
#f is number of reviews impact ratings?
#g Is running time of movie impact ratings
#h Does language of movie impact ratings and revenue
#i Is there a social media impact of likes on movie?
#j  Which country make better movie( or higher IMDB rating)?
#k Which country earn maximum revenue/profit from movies?

#k Do people give rating/likes/revenue based on aspect ratio of movie?  etc.
#m IMDB score VS country
#n IMDB score VS movie year
#o  Does facenumber_in_poster  impact ratings?


# Lets analyze and visualize data.

#Step 2
#Correlation and PCA to reduce the dimensions of data beause may be not dimensions are so important.

#Step 3
# Model building
#Step 4
#Model evaluation -( By dividing data into test and train)
#Step 5
#Applying cross validation to increase our model perforamance.
#Step 6
#Insights



# In[9]:


#df_movies_final.groupby(['genres']).count()
df_movies_final['genres'].value_counts()
# Genre of Comedy is the highest
#!pip install plotly
import plotly.plotly as py
Bar_plot_genre=df_movies_final['genres'].value_counts()
Bar_plot_genre.plot.bar(legend=True,figsize=(10,10))
#py.iplot(df_movies_final['genres'].value_counts(),filename='basic-bar')


# In[10]:


#Genre vs IMDB rating
# Wow, Biography has the largest average IMDB rating among all and it has the double facebook like for director. People are connecting
# to Biography more than fictional series.
# Romantic actors gets more facebook likes but the director in romantic movie is insignificant.
# Family movie has more gross than big budget movies like action,adventure and animation.( If any one wants to invest in movies ,family movie is better choice)
# etc .There are many insights here.
g1=df_movies_final.groupby(['genres']).mean()
type(g1)
g1.index
g1.add_suffix('_Count').reset_index()
# Lets plot Genre and IMDB.(Genres vs imdb_score_Count)

#df_movies_final.plot(x='genres', y='imdb_score',kind='bar')
#Director facebooklikes have a positive impact on the IMDb score as seen in genre(Western) 


# In[11]:


#df_movies_final.pivot_table(columns='color') #nothing much.


# In[12]:


g2=df_movies_final.groupby(['color']).mean()
g2
# Black and White color has more facebook like for director.
#Gross revenue of color movie is more.

# By  using pivot and groupby function, we can bring various relationsshios between data.


# In[13]:


#d  Is director facebook rating impact movie?
plt.figure(figsize=(12,12))
plt.title("IMDB vs Director Facebook Popularity")
plt.xlabel("IMDB Score")
plt.ylabel("Facebook Popularity")
tmp=plt.scatter(df_movies_final.imdb_score,df_movies_final.director_facebook_likes,c=df_movies_final.imdb_score,vmin=1,vmax=10)
plt.yticks([i*2500 for  i in range(11)])
plt.colorbar(tmp,fraction=0.25)
plt.show()
# Generally, high facebook popualrity have a  good impact on rating. But some exceptions are always there.


# In[14]:


#e Is actor's facebook like imapct movie?
plt.figure(figsize=(12,12))
plt.title("IMDB vs Cast facebook popularity")
plt.xlabel("IMDB Score")
plt.ylabel("Facebook Popularity")
tmp=plt.scatter(df_movies_final.imdb_score,df_movies_final.cast_total_facebook_likes,c=df_movies_final.imdb_score,vmin=1,vmax=10)
plt.yticks([i*70000 for  i in range(11)])
plt.colorbar(tmp,fraction=0.25)
plt.show()
#No impact of cast
#Normally distributed


# In[15]:


#f is number of reviews impact ratings?
plt.figure(figsize=(12,12))
plt.title("IMDB vs Cast cast_total_facebook_likes")
plt.xlabel("IMDB Score")
plt.ylabel("Facebook Popularity")
tmp=plt.scatter(df_movies_final.imdb_score,df_movies_final.cast_total_facebook_likes,c=df_movies_final.imdb_score,vmin=1,vmax=10)
plt.yticks([i*70000 for  i in range(11)])
plt.colorbar(tmp,fraction=0.25)
plt.show()
#no correlation.
# And we can analyze various other factor 


# In[16]:


#j  Which country make better movie( or higher IMDB rating)?
# so we draw boxplot with higer and lower 
List_of_countries=df_movies_final.country.unique()
country_score=[]
for i in List_of_countries:
    country_score.append(df_movies_final.imdb_score[df_movies_final.country==i])

List_of_countries=np.insert(List_of_countries,0,'')    
    
country_score    
plt.figure(figsize=(10,10))
plt.title("IMDB Score Vs Country")
plt.ylabel("IMDB Score")
plt.xlabel('Country')
plt.boxplot(country_score,widths=.75,)
plt.xticks(range(len(List_of_countries)),List_of_countries,rotation=90,fontsize=8)
plt.show()
# US and UK make maximum movie but Brazil has the highest median rating.


# In[17]:


#Gross vs budget across genre
#Gross-budget=Profit

List_of_genre=df_movies_final.genres.unique()
gross_across_genre=[]
for i in List_of_genre:
    gross_across_genre.append(df_movies_final.gross[df_movies_final.genres==i])

List_of_genre=np.insert(List_of_genre,0,'')    
    
  
plt.figure(figsize=(10,10))
plt.title("Gross Vs genre")
plt.ylabel("gross")
plt.xlabel('genre')
plt.boxplot(gross_across_genre,widths=.75,)
plt.xticks(range(len(List_of_genre)),List_of_genre,rotation=90,fontsize=8)
plt.show()


# In[18]:


# type(gross_across_genre) #sequence
# type(List_of_genre) # array

# gross_across_genre = np.array(gross_across_genre).tolist()
# type(gross_across_genre)


# In[19]:


# List_of_genre
# range(len(List_of_genre))
# type(gross_across_genre)
# gross_across_genre = np.asarray(gross_across_genre)
# gross_across_genre
# np.asarray(gross_across_genre)


# In[20]:


# List_of_genre=df_movies_final.genres.unique()

# gross_across_genre=[]
# for i in List_of_genre:
#     gross_across_genre.append(df_movies_final.gross[df_movies_final.genres==i])

# plt.bar(gross_across_genre,List_of_genre)
    
# #gross_across_genre = np.asarray(gross_across_genre)

# plt.title("Gross Vs genre")
# plt.ylabel("gross")
# plt.xlabel('genre')
# plt.bar(range(len(List_of_genre)),np.asarray(gross_across_genre),align='center',alpha=0.5)
# plt.xticks(range(len(List_of_genre)),List_of_genre,rotation=90,fontsize=8)
# plt.show()


# In[21]:


#Step 2
#Correlation and PCA to reduce the dimensions of data beause may be not dimensions are so important.


# In[22]:


correlation=df_movies_final.corr()
correlation


# In[23]:


range(len(correlation.columns))


# In[24]:


plt.figure(figsize=(10,10))
tmp=plt.matshow(correlation,fignum=1)
plt.xticks(range(len(correlation.columns)),correlation.columns,rotation=90,fontsize=8)
plt.yticks(range(len(correlation.columns)),correlation.columns,fontsize=8)
plt.colorbar(tmp,fraction=0.035)
plt.show()


# In[ ]:


# Analyzing gross and budget


# In[ ]:


# 28 are too many a dimensions for building a model. Too many parameters will make our model complex and increase the variance.
#lets try PCA.

