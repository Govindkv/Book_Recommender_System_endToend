'''
Author: Govind Kumar
Email: Govind26663355@gmail.com
Date: 15-09-2024
'''

import pickle
import streamlit as st
import numpy as np

st.header("Smart Recommendations for Avid Readers:")
st.markdown("<h5 style='text-align: right; color: #ff6600;'>Powered by Govind Kumar's System</h5>", unsafe_allow_html=True)

# Load models and data with error handling
try:
    model = pickle.load(open('artifacts\model.pkl', 'rb'))
    book_names = pickle.load(open('artifacts\books_name.pkl', 'rb'))
    final_rating = pickle.load(open('artifacts\final_rating.pkl', 'rb'))
    book_pivot = pickle.load(open('artifacts\book_pivot.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

def fetch_poster(suggestion):
    book_names_list = []
    poster_urls = []

    for book_id in suggestion:
        book_name = book_pivot.index[book_id]
        book_names_list.append(book_name)

    for name in book_names_list:
        idx = np.where(final_rating['title'] == name)[0]
        if idx.size > 0:
            url = final_rating.iloc[idx[0]]['image_url']
            poster_urls.append(url)
        else:
            poster_urls.append('')  # Empty URL if not found

    return poster_urls

def recommend_book(book_name):
    books_list = []
    try:
        book_id = np.where(book_pivot.index == book_name)[0][0]
    except IndexError:
        st.error("Book not found in the dataset.")
        return [], []

    distance, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    # Flatten suggestions to get individual book IDs
    suggested_books_ids = suggestions.flatten()

    poster_urls = fetch_poster(suggested_books_ids)
    
    for book_id in suggested_books_ids:
        books_list.append(book_pivot.index[book_id])
    
    return books_list, poster_urls

selected_books = st.selectbox(
    "Pick a Book You Love, and Let Us Find Your Next Favorite!",
    book_names
)

if st.button('Show Recommendation'):
    recommended_books, poster_urls = recommend_book(selected_books)
    
    # Display recommendations and posters
    cols = st.columns(min(len(recommended_books), 5))  # Adjust columns based on the number of recommendations
    
    for i, col in enumerate(cols):
        if i < len(recommended_books):  # Ensure we don't access out of bounds
            col.text(recommended_books[i])
            if i < len(poster_urls) and poster_urls[i]:
                col.image(poster_urls[i])
            else:
                col.text("No image available")
