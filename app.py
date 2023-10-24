import streamlit as st
import pickle
import pandas as pd
import numpy as np
st.title('Book Recommender System')
model = pickle.load(open('model.pkl','rb'))
book_pivot = pd.read_csv('pivot.csv')
book_img = pd.read_csv('book_img.csv')
book_name_list =book_pivot.title.tolist()
book_name= st.selectbox('Select Book Name',book_name_list)
if st.button('Get Recommendations'):
    st.title('Recommended Books Are')
    book_index = np.where(book_pivot['title']==book_name)[0][0]
    book_array = book_pivot.iloc[book_index,1:].values.reshape(1,-1)
    distances,suggestions=model.kneighbors(book_array,n_neighbors=6)
    suggestions=suggestions.tolist()[0]
    for i in range(len(suggestions)):
        if i != 0:
            title = book_pivot.title[suggestions[i]]
            st.header(title)
            url = book_img[book_img['Book-Title']==title]['Image-URL-L']
            st.image(url.values[0])
            
            
            
            

