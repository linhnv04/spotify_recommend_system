from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np

app = FastAPI()
df = pd.read_csv("perfect.csv")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/")
def homepage(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/recommend')
def recommend(song_name, request: Request):
    songs = get_best_similarity_song(df, song_name)
    return templates.TemplateResponse('song.html', {'request': request, 'songs': songs})


def euclidean_similarity(row, point):
    """
    Compute the Euclidean similarity between a vector created from a row in a DataFrame and a given point.
    
    Parameters:
        row (pandas Series): A row in the DataFrame.
        point (numpy array): A vector.
        
    Returns:
        float: Euclidean similarity between the vector created from the row and the given point.
    """
    temp = np.array(row.iloc[:17])  # Assuming you want to use the first 17 elements of the row
    distance = np.linalg.norm(temp - point)  # Compute the Euclidean distance
    similarity = 1 / (1 + distance)  # Compute the similarity
    return similarity


def get_best_similarity_song(df, name, n_best = 10):
    name = name.lower()
    df["lowername"] = df["name"].apply(str.lower)
    temp = df[df["lowername"] == name]

    # temp = df[df["name"] == name]
    if len(temp) == 0:
        return []
    ret = []
    compare_point = np.array(temp.iloc[0,:17])
    kind = temp.iloc[0,17]
    same_type_df = df[df["pred"] == kind].copy()
    same_type_df.drop_duplicates(subset=["name"], inplace = True)

    # df["simi_score"] = df.apply(euclidean_similarity,args=(compare_point,), axis= 1)
    same_type_df["simi_score"] = same_type_df.apply(euclidean_similarity, args=(compare_point,), axis=1)

    top_n = same_type_df.sort_values(by=["simi_score"], ascending=False).iloc[1:n_best+1,:]
    for index, row in top_n.iterrows():
        ret.append((row["name"],row["id"]))
    # df['distance_to_user_point'] = df.apply(calculate_distance_to_point, args=(user_point,), axis=1)
    return ret