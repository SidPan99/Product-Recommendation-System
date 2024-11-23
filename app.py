
import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader

data = pd.read_csv('assignment.csv')

reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(data[['User ID', 'Product ID', 'Ratings']], reader)
trainset = surprise_data.build_full_trainset()
model = SVD()
model.fit(trainset)

def get_recommendations(user_id, top_n=5):
    all_products = data['Product ID'].unique()
    user_rated_products = data[data['User ID'] == user_id]['Product ID']
    unrated_products = [p for p in all_products if p not in user_rated_products.values]

    recommendations = []
    for product_id in unrated_products:
        prediction = model.predict(user_id, product_id).est
        recommendations.append((product_id, prediction))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return [product for product, _ in recommendations[:top_n]]

st.title("Product Recommendation System")

user_id = st.number_input("Enter User ID", min_value=1, step=1)
top_n = st.number_input("Number of Recommendations", min_value=1, max_value=10, step=1, value=5)

if st.button("Get Recommendations"):
    recommendations = get_recommendations(user_id, top_n)
    st.write(f"Recommendations: {recommendations}")
