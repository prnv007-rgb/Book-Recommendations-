import pickle
import streamlit as st
import numpy as np

st.header("Book Recommender System")

# Load the pre-trained models and data
model = pickle.load(open('artificats/model.pkl', 'rb'))
books_name = pickle.load(open('artificats/bn.pkl', 'rb'))
final_rating = pickle.load(open('artificats/fr.pkl', 'rb'))
books_pivot = pickle.load(open('artificats/bp.pkl', 'rb'))

# Function to fetch poster URLs
def fetch_poster(suggestion):
    book_names = []
    id_index = []
    poster_url = []
    
    for book_id in suggestion[0]:  # suggestion is a 2D array; we need the first list
        book_names.append(books_pivot.index[book_id])
    
    for name in book_names:
        ids = np.where(final_rating['T'] == name)[0][0]
        id_index.append(ids)
    
    for idx in id_index:
        url = final_rating.iloc[idx]['I']
        poster_url.append(url)
    
    return poster_url

# Function to recommend books
def rec_book(book_name):
    book_list = []
    book_id = np.where(books_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(books_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    
    poster_url = fetch_poster(suggestion)
    
    for i in range(len(suggestion[0])):  # suggestion is a 2D array; we need the first list
        books = books_pivot.index[suggestion[0][i]]
        book_list.append(books)
    
    return book_list, poster_url

# Streamlit UI for selecting and showing recommendations
selected_books = st.selectbox("Type or select a book", books_name)

if st.button("Show Recommendations"):
    recommendation_books, poster_url = rec_book(selected_books)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.text(recommendation_books[1])
        st.image(poster_url[1])
        
    with col2:
        st.text(recommendation_books[2])
        st.image(poster_url[2])
        
    with col3:
        st.text(recommendation_books[3])
        st.image(poster_url[3])
        
    with col4:
        st.text(recommendation_books[4])
        st.image(poster_url[4])
        
    with col5:
        st.text(recommendation_books[5])
        st.image(poster_url[5])
