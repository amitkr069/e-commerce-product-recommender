import os
import streamlit as st
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
# import logging

# --- LangChain / Gemini Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.globals import set_verbose

# --- Configuration & Initialization ---
load_dotenv()
# logging.basicConfig(level=logging.INFO)
set_verbose(True)

# --- MongoDB Configuration ---
MONGO_URI = st.secrets.get("MONGO_URL") or os.getenv("MONGO_URL")
DATABASE_NAME = "Product_recommender"

# --- LLM Key ---
LLM_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# --- Database & Data Initialization ---
@st.cache_resource
def initialize_db_and_data():
    """Initializes MongoDB connection and loads data."""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        db.command('ping')

        users_df = pd.DataFrame(list(db.users.find({}, {'_id': 0})))
        products_df = pd.DataFrame(list(db.products.find({}, {'_id': 0})))
        interactions = pd.DataFrame(list(db.user_interactions.find({}, {'_id': 0})))

        if users_df.empty or products_df.empty or interactions.empty:
            st.error("Some collections are empty (users, products, or user_interactions).")
            return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        interactions['score'] = interactions['interaction_type'].apply(lambda x: 2 if x == 'purchase' else 1)
        interaction_matrix = interactions.pivot_table(
            index='user_id', columns='product_id', values='score'
        ).fillna(0)

        return db, users_df, products_df, interaction_matrix

    except Exception as e:
        st.error(f"Critical Error connecting/loading MongoDB: {e}")
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

DB, USERS_DF, PRODUCTS_DF, INTERACTION_MATRIX = initialize_db_and_data()

# --- LLM Initialization ---
@st.cache_resource
def initialize_llm_chain():
    if not LLM_API_KEY:
        st.warning("GEMINI_API_KEY missing — LLM explanations disabled.")
        return None

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
            max_output_tokens=80,
            google_api_key=LLM_API_KEY
        )

        explainer_template = (
            "You are a concise and persuasive e-commerce product recommendation explainer. "
            "Your only task is to provide a brief, personalized reason for recommending the {product_name}. "
            "The recommendation was generated because: {user_behavior_context}. "
            "Generate the explanation based ONLY on the context provided. "
            "Respond in exactly one or two sentences, no JSON or extra formatting."
        )

        explainer_prompt = PromptTemplate.from_template(explainer_template)
        return LLMChain(llm=llm, prompt=explainer_prompt, verbose=False)

    except Exception as e:
        st.error(f"LLM Initialization Failed: {e}")
        return None

EXPLAINER_CHAIN = initialize_llm_chain()

# --- LLM Explanation Function ---
def call_gemini_service_lc(product_name: str, user_behavior_context: str) -> str:
    if not EXPLAINER_CHAIN:
        return f"The {product_name} is a top match, but the LLM explanation service is unavailable."

    try:
        response = EXPLAINER_CHAIN.run({
            "product_name": product_name,
            "user_behavior_context": user_behavior_context
        })
        return str(response).strip()

    except Exception as e:
        st.error(f"LLM Invocation Failed: {type(e).__name__}: {e}")
        return f"We recommend the {product_name} because it matches your interests and is highly rated."


# --- Hybrid Recommendation Logic ---
def generate_hybrid_recommendations(
    user_id: int,
    selected_product_id: str,
    matrix: pd.DataFrame,
    products_df: pd.DataFrame,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """Generates recommendations giving more weight to content-based filtering."""
    if matrix.empty or user_id not in matrix.index:
        return []

    recommendations = {}
    selected_product = products_df.loc[products_df['product_id'] == selected_product_id].iloc[0]
    selected_category = selected_product['category']
    selected_name = selected_product['name']

    # --- Content-Based Filtering (High weight) ---
    cbf_candidates = products_df[
        (products_df['category'] == selected_category) &
        (products_df['product_id'] != selected_product_id)
    ].copy()

    for _, row in cbf_candidates.head(3).iterrows():  # give more priority to similar products
        pid = row['product_id']
        context = f"You showed interest in '{selected_category}' products like {selected_name}."
        recommendations[pid] = {
            'product_name': row['name'],
            'raw_context': context,
            'score': 2.0  # higher weight
        }

    # --- User-Based Collaborative Filtering (Lower weight) ---
    user_vector = matrix.loc[user_id].values.reshape(1, -1)
    user_sim = cosine_similarity(matrix, user_vector).flatten()
    similar_users = matrix.index[user_sim.argsort()[::-1][1:4]]  # top 3 similar users

    for sim_user in similar_users:
        user_products = matrix.loc[sim_user]
        liked_products = user_products[user_products > 0].index.tolist()

        for pid in liked_products:
            if pid != selected_product_id and pid not in recommendations:
                product_row = products_df.loc[products_df['product_id'] == pid]
                if not product_row.empty:
                    pname = product_row.iloc[0]['name']
                    context = f"Similar users to you also interacted with {pname}."
                    recommendations[pid] = {
                        'product_name': pname,
                        'raw_context': context,
                        'score': 1.0  # lower weight
                    }

    # --- Sort and Return Final Recommendations ---
    final_results = sorted(recommendations.items(), key=lambda x: -x[1]['score'])
    return [
        {
            'product_id': pid,
            'product_name': data['product_name'],
            'raw_context': data['raw_context']
        }
        for pid, data in final_results[:top_n]
    ]


# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Hybrid Product Recommender", layout="wide")
    st.title("🛍️ E-commerce Product Recommender")
    st.markdown("Get personalized product recommendations")
    st.markdown("---")

    if DB is None or INTERACTION_MATRIX.empty:
        st.warning("Cannot proceed — MongoDB connection failed or data missing.")
        return

    # --- Sidebar Inputs ---
    st.sidebar.title("Configuration")
    user_email = st.sidebar.selectbox("Select User Email", options=[None] + USERS_DF['email'].tolist())
    product_name = st.sidebar.selectbox("Select a Product", options=[None] + PRODUCTS_DF['name'].tolist())

    # llm_status_message = "🟢 LLM Ready (Gemini)" if EXPLAINER_CHAIN else "🔴 LLM Fallback Active"
    # st.sidebar.markdown(f"**LLM Status:** {llm_status_message}")

    if user_email and product_name:
        user_data = USERS_DF.loc[USERS_DF['email'] == user_email].iloc[0]
        user_id = user_data['user_id']
        selected_product_id = PRODUCTS_DF.loc[PRODUCTS_DF['name'] == product_name, 'product_id'].iloc[0]

        st.markdown(f"Recommendations for **{user_email}**")
        st.write(f"Selected Product: **{product_name}**")

        with st.spinner("Generating hybrid recommendations..."):
            recommendation_list = generate_hybrid_recommendations(
                user_id, selected_product_id, INTERACTION_MATRIX, PRODUCTS_DF, top_n=5
            )

        if not recommendation_list:
            st.info(f"No recommendations available for {user_email} and product '{product_name}'.")
            return

        # st.success(f"Generated {len(recommendation_list)} recommendations.")
        st.subheader("Recommended Products")

        data = []
        with st.spinner("Fetching Recommendations"):
            for rec in recommendation_list:
                llm_explanation = call_gemini_service_lc(rec['product_name'], rec['raw_context'])
                data.append({
                    "Product Name": rec["product_name"],
                    # "Product ID": rec["product_id"],
                    "Why this product!!": llm_explanation
                })

        results_df = pd.DataFrame(data)
        st.table(results_df)

        st.markdown("---")
        # st.subheader("Debugging Context")
        # st.info("The context used for LLM explanations is shown in code execution above.")

    else:
        st.info("Select both a user email and a product from the sidebar to begin.")


if __name__ == "__main__":
    main()
