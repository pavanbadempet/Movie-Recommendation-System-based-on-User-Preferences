#Run using py -m streamlit run app.py
import streamlit as st
import base64
import pickle
import pandas as pd
import requests
import time
st.set_page_config(page_title="Movie Recommendation System based on User Preferences", page_icon="🎬",layout="wide")
hide = """<style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
}
            </style>
            """
        
st.markdown(hide,unsafe_allow_html=True) 



def fetch_backdrops(movie_id):
    
    response2 = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key=44726ef95f4d79cb7001a4947fca7f53&language=en-US")
    data2 = response2.json()
    print(data2)
    try:
        return data2["results"][0]["key"]
    except:
        return "n73_6vyq2v4"

def fetch_poster(data):
    try:
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        return "https://st3.depositphotos.com/1322515/35964/v/600/depositphotos_359648638-stock-illustration-image-available-icon.jpg"

def fetch_info(movie_id,data):    
    try:
        return [data["release_date"],data["budget"],data["revenue"],data["popularity"]]
    except:
        return "No Info Available"



def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    recommended_posters = []
    recommended_backdrops = []
    recommended_info = []
    for i in distances[1:11]:
        if i is not None:
            movie_id = movies.iloc[i[0]].movie_id
            recommended_movies.append([movies.iloc[i[0]].title] + [movies.iloc[i[0]].genres] + [movies.iloc[i[0]].cast] + [movies.iloc[i[0]].crew] + [movies.iloc[i[0]].overview])
            # fetching poster via api
            response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=44726ef95f4d79cb7001a4947fca7f53&language=en-US')
            data = response.json()
            recommended_posters.append(fetch_poster(data))
            recommended_info.append(fetch_info(movie_id,data))
            recommended_backdrops.append(fetch_backdrops(movie_id))
    return recommended_movies,recommended_posters,recommended_info,distances,recommended_backdrops


movies_list = pickle.load(open('movie_list.pkl', 'rb'))
movies = pd.DataFrame(movies_list)

similarity = pickle.load(open('similarity.pkl', 'rb'))

st.title("Movie Recommendation System based on User Preferences (MRSBUP)")
selected_movie_name = st.selectbox('Looking for Similar Movies?', movies['title'].values)

def display(fetch):
    print(fetch)
    video_html = """
    
    <style>
    
    .video-container{
  width: 60vw;
  height: 100vh;
    position: absolute;
    min-width: 80%; 
    
   filter: brightness(60%);
}
    
iframe {
  position: absolute;
  top: 52.5%;
  left: 60%;
  width: 100vw;
  height: 100vh;
  transform: translate(-50%, -50%);
}
</style>
""" + f"""
<div class="video-container">
  <iframe src="https://www.youtube.com/embed/{fetch}?controls=0&autoplay=1&mute=1&loop=1"></iframe>
</div>
</body>
        """
    st.markdown(video_html, unsafe_allow_html=True)

if st.button('Recommend'):
    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
    names, posters, info, distances, backdrops = recommend(selected_movie_name)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10= st.tabs([names[0][0], names[1][0], names[2][0], names[3][0], names[4][0],names[5][0], names[6][0], names[7][0], names[8][0], names[9][0]])
    with tab1:
        display(backdrops[0])
        col1, col2 = st.columns(2)
        with col1:
            st.image(posters[0], width=350)
        with col2:
            st.subheader(names[0][0])
            st.caption("Release Date: "+ info[0][0])
            st.caption("Budget: $" + str(info[0][1]))
            st.caption("Collection: $" + str(info[0][2]))
            st.caption("Popularity: " +str(info[0][3]))
            st.caption("Genres: " +str(names[0][1]))
            st.caption("Cast: " +str(names[0][2]))
            st.caption("Director: " +str(names[0][3]))
            st.caption("Similarity: " +str(distances[1][1]))
            st.caption("Overview: ") 
            st.caption(names[0][4])

    with tab2:
        display(backdrops[1])
        col3, col4 = st.columns(2)
        with col3:
            st.image(posters[1], width=350)
        with col4:
            st.subheader(names[1][0])
            st.caption("Release Date: "+ info[1][0])
            st.caption("Budget: $" + str(info[1][1]))
            st.caption("Collection: $" + str(info[1][2]))
            st.caption("Popularity: " +str(info[1][3]))
            st.caption("Genres: " +str(names[1][1]))
            st.caption("Cast: " +str(names[1][2]))
            st.caption("Director: " +str(names[1][3]))
            st.caption("Similarity: " +str(distances[2][1]))
            st.caption("Overview: ") 
            st.caption(names[1][4])

    with tab3:
        display(backdrops[2])
        col5, col6 = st.columns(2)
        with col5:
            st.image(posters[2], width=350)
        with col6:
            st.subheader(names[2][0])
            st.caption("Release Date: "+ info[2][0])
            st.caption("Budget: $" + str(info[2][1]))
            st.caption("Collection: $" + str(info[2][2]))
            st.caption("Popularity: " +str(info[2][3]))
            st.caption("Genres: " +str(names[2][1]))
            st.caption("Cast: " +str(names[2][2]))
            st.caption("Director: " +str(names[2][3]))
            st.caption("Similarity: " +str(distances[3][1]))
            st.caption("Overview: ") 
            st.caption(names[2][4])

    with tab4:
        display(backdrops[3])
        col7, col8 = st.columns(2)
        with col7:
            st.image(posters[3], width=350)
        with col8:
            st.subheader(names[3][0])
            st.caption("Release Date: "+ info[3][0])
            st.caption("Budget: $" + str(info[3][1]))
            st.caption("Collection: $" + str(info[3][2]))
            st.caption("Popularity: " +str(info[3][3]))
            st.caption("Genres: " +str(names[3][1]))
            st.caption("Cast: " +str(names[3][2]))
            st.caption("Director: " +str(names[3][3]))
            st.caption("Similarity: " +str(distances[4][1]))
            st.caption("Overview: ") 
            st.caption(names[3][4])

    with tab5:
        display(backdrops[4])
        col9, col10 = st.columns(2)
        with col9:
            st.image(posters[4], width=350)
        with col10:
            st.subheader(names[4][0])
            st.caption("Release Date: "+ info[4][0])
            st.caption("Budget: $" + str(info[4][1]))
            st.caption("Collection: $" + str(info[4][2]))
            st.caption("Popularity: " +str(info[4][3]))
            st.caption("Genres: " +str(names[4][1]))
            st.caption("Cast: " +str(names[4][2]))
            st.caption("Director: " +str(names[4][3]))
            st.caption("Similarity: " +str(distances[5][1]))
            st.caption("Overview: ") 
            st.caption(names[4][4])

    with tab6:
        display(backdrops[5])
        col11, col12 = st.columns(2)
        with col11:
            st.image(posters[5], width=350)
        with col12:
            st.subheader(names[5][0])
            st.caption("Release Date: "+ info[5][0])
            st.caption("Budget: $" + str(info[5][1]))
            st.caption("Collection: $" + str(info[5][2]))
            st.caption("Popularity: " +str(info[5][3]))
            st.caption("Genres: " +str(names[5][1]))
            st.caption("Cast: " +str(names[5][2]))
            st.caption("Director: " +str(names[5][3]))
            st.caption("Similarity: " +str(distances[6][1]))
            st.caption("Overview: ") 
            st.caption(names[5][4])

    with tab7:
        display(backdrops[6])
        col13, col14 = st.columns(2)
        with col13:
            st.image(posters[6], width=350)
        with col14:
            st.subheader(names[6][0])
            st.caption("Release Date: "+ info[6][0])
            st.caption("Budget: $" + str(info[6][1]))
            st.caption("Collection: $" + str(info[6][2]))
            st.caption("Popularity: " +str(info[6][3]))
            st.caption("Genres: " +str(names[6][1]))
            st.caption("Cast: " +str(names[6][2]))
            st.caption("Director: " +str(names[6][3]))
            st.caption("Similarity: " +str(distances[7][1]))
            st.caption("Overview: ") 
            st.caption(names[6][4])

    with tab8:
        display(backdrops[7])
        col15, col16 = st.columns(2)
        with col15:
            st.image(posters[7], width=350)
        with col16:
            st.subheader(names[7][0])
            st.caption("Release Date: "+ info[7][0])
            st.caption("Budget: $" + str(info[7][1]))
            st.caption("Collection: $" + str(info[7][2]))
            st.caption("Popularity: " +str(info[7][3]))
            st.caption("Genres: " +str(names[7][1]))
            st.caption("Cast: " +str(names[7][2]))
            st.caption("Director: " +str(names[7][3]))
            st.caption("Similarity: " +str(distances[8][1]))
            st.caption("Overview: ") 
            st.caption(names[7][4])

    with tab9:
        display(backdrops[8])
        col17, col18 = st.columns(2)
        with col17:
            st.image(posters[8], width=350)
        with col18:
            st.subheader(names[8][0])
            st.caption("Release Date: "+ info[8][0])
            st.caption("Budget: $" + str(info[8][1]))
            st.caption("Collection: $" + str(info[8][2]))
            st.caption("Popularity: " +str(info[8][3]))
            st.caption("Genres: " +str(names[8][1]))
            st.caption("Cast: " +str(names[8][2]))
            st.caption("Director: " +str(names[8][3]))
            st.caption("Similarity: " +str(distances[9][1]))
            st.caption("Overview: ") 
            st.caption(names[8][4])

    with tab10:
        display(backdrops[9])
        col19, col20 = st.columns(2)
        with col19:
            st.image(posters[9], width=350)
        with col20:
            st.subheader(names[9][0])
            st.caption("Release Date: "+ info[9][0])
            st.caption("Budget: $" + str(info[9][1]))
            st.caption("Collection: $" + str(info[9][2]))
            st.caption("Popularity: " +str(info[9][3]))
            st.caption("Genres: " +str(names[9][1]))
            st.caption("Cast: " +str(names[9][2]))
            st.caption("Director: " +str(names[9][3]))
            st.caption("Similarity: " +str(distances[10][1]))
            st.caption("Overview: ") 
            st.caption(names[9][4])