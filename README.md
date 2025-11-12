# Introduction
A Recommendation System is a filtering program whose primary goal is to predict the “rating” or “preference” of a user towards a domain-specific element.  In our project, this domain-specific element is a movie. Hence the main focus of our recommendation system is to provide a total of ten movie recommendations to users who searched for a movie that they like. These results are based on tags of the movie that has been searched. Content based filtering is a technique that is used to recommend movies. 
Apart from providing recommendations the system also provides posters, trailers/relevant videos of the Movies along with Release Date, Budget, Collection, Popularity, Similarity between selected Movie, Overview of Movie and More. 
The System uses the concept of vectorization based on common features and uses Cosine Similarity with respect to each other vectors to determine the most similar movies.

## Test It Online
Hosted on Azure: ~~http://mrsbup.azurewebsites.net/~~


## Installation Local

1. Download the Repository
2. Create a Virtual Environment for Movie-Recommendation-System-based-on-User-Preferences/Movie Recommendation System based on User Preferences/movie-recommender-system

## Usage

```bash
py -m streamlit run app.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
