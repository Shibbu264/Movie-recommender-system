# %%


import modelbit
import numpy as np
import pandas as pd
mb = modelbit.login()


# %%
movies =pd.read_csv('tmdb_5000_movies[1].csv')
credits =pd.read_csv('tmdb_5000_credits[1].csv')

# %%
movies.head(5)

# %%
credits.head(1)

# %%
movies=movies.merge(credits,on='title')
movies.head(1)

# %%
# genre id keywords title overview cast crew

# %%
movies=movies[['movie_id','title','cast','crew','overview','genres','keywords']]
movies.head(1)

# %%
movies.dropna(inplace=True)
movies.isnull().sum()


# %%
movies=movies.drop_duplicates()
movies.duplicated().sum()

# %%
movies.iloc[0].genres

# %%
import ast
def convert(obj):
 L = []
 for i in ast.literal_eval(obj):
  L.append(i['name'])
 return L


# %%
movies['genres']=movies['genres'].apply(convert)

# %%
movies['keywords']=movies['keywords'].apply(convert)

# %%
movies.head(1)

# %%
def convert3(obj):
 L = []
 counter = 0
 for i in ast.literal_eval(obj):
  if counter != 3:
   L.append(i['name'])
   counter+=1
  else:
   break
 return L


# %%
movies['cast']=movies['cast'].apply(convert3)

# %%
def fetch_direc(obj):
 L = []
 for i in ast.literal_eval(obj):
     if i['job']== 'Director':
          L.append(i['name'])
         
     
   
 return L

# %%
movies['title']

# %%
movies['crew']=movies['crew'].apply(fetch_direc)

# %%
movies.head()

# %%
movies ['overview']= movies ['overview'].apply(lambda x:x.split())

# %%
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])

# %%
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])

# %%
movies['tags']=movies['cast']+movies['crew']+movies['keywords']+movies['genres']

# %%
movies['tags']=movies['tags']+movies ['overview']

# %%
movies.head(1)

# %%
new_df=movies[['movie_id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

# %%
new_df['tags'][0]

# %%
from sklearn.feature_extraction.text import CountVectorizer

# %%
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()

# %%
vectors

# %%
cv.get_feature_names_out()

# %%
!pip install nltk

# %%
import nltk
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()


# %%
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)    
        

# %%
new_df['tags'].apply(stem)

# %%
from sklearn.metrics.pairwise import cosine_similarity
similarity =cosine_similarity(vectors)

# %%

def recommend2(movie):
    movie_index = new_df[new_df['title'] == 'Avatar'].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_moviesID = []
    
    for i in movies_list:
        recommended_moviesID.append(new_df.iloc[i[0]].movie_id)
    
    return recommended_moviesID 
    

recommend2('Avatar')
    

# %%
!pip install flask --user
!pip install flask-ngrok --user
!pip install requests
!pip freeze requirements.txt

# %%
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import requests
app = Flask(__name__)
run_with_ngrok(app) 





def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
    
    recommended_movies = []
    
    for i in movies_list:
        recommended_movies.append(new_df.iloc[i[0]].title)
    
    return recommended_movies




def recommend2(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
    
    recommended_moviesID = []
    
    for i in movies_list:
        url = "https://api.themoviedb.org/3/movie/"+str(new_df.iloc[i[0]].movie_id)+"?language=en-US"
        headers = {
                  "accept": "application/json",
                   "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjNTY5M2ZjNzI5NzAyZTRjNmRhZjhmOWVmZWUyMDRlYSIsInN1YiI6IjY1N2I4M2E2ZTkzZTk1MjE4ZjZjZTYzYSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Gl7NxlB8u9d8n76XVDpvwlFQNY6KG0G7b8EoomJwKx8"
                   }

        response = requests.get(url, headers=headers)

        recommended_moviesID.append("https://image.tmdb.org/t/p/w500/"+response.json()['poster_path'])

       
    
    return recommended_moviesID
def recommend3(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
    
    moviespath = []
    
    for i in movies_list:
        url = "https://api.themoviedb.org/3/movie/"+str(new_df.iloc[i[0]].movie_id)+"?language=en-US"
        headers = {
                  "accept": "application/json",
                   "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjNTY5M2ZjNzI5NzAyZTRjNmRhZjhmOWVmZWUyMDRlYSIsInN1YiI6IjY1N2I4M2E2ZTkzZTk1MjE4ZjZjZTYzYSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Gl7NxlB8u9d8n76XVDpvwlFQNY6KG0G7b8EoomJwKx8"
                   }

        response = requests.get(url, headers=headers)

        moviespath.append(response.json()['homepage'])

       
    
    return moviespath
    

   
    
        

# PUT request handler for the /recommend endpoint
@app.route('/recommend', methods=['POST'])
def handle_recommendation():
    try:
        # Get the data from the request
        request_data = request.get_json(force=True)
        print(request_data)

        # Call the recommend function with the request data
        result = recommend(request_data.get('movie',''))
        result2 = recommend2(request_data.get('movie',''))
        result3 = recommend3(request_data.get('movie',''))
       
        print(result2)

        # Return the result as a JSON response
        return jsonify({'result': result,'result2':result2,'result3':result3})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run()

mb.deploy(handle_recommendation)


