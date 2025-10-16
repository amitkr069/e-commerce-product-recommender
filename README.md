# e-commerce-product-recommender

**Video Demo** - https://drive.google.com/file/d/1JlV3RXIDO6DgERLiQ9_PGWbDhFmn7ChW/view?usp=sharing

**Website Link** - https://e-commerce-pro-duct-recommender-amit-kumar-mandal.streamlit.app/
_________
**Overview**

  The E-Commerce Product Recommender System is an intelligent web application that suggests relevant products to users by combining Content-Based Filtering (CBF) and Collaborative Filtering (CF) techniques.
  
  The app uses MongoDB for data storage, Streamlit for the frontend, and Google’s Gemini (via LangChain) to generate concise, natural-language explanations for each recommendation.
  
  This hybrid design ensures recommendations are not only personalized but also interpretable, helping users understand why each product was recommended.

**Key Features**

 Hybrid Recommendation Engine — Combines Content-Based Filtering (product similarity) and Collaborative Filtering (user interaction similarity).
 
 Weighted Recommendation Logic — Prioritizes product similarity (CBF) more heavily than user similarity (CF).
 
 LLM-Powered Explanations — Uses Gemini 2.5 Flash (via LangChain) to generate short, human-readable explanations for each recommendation.
 
 MongoDB Integration — Retrieves users, products, and interactions from MongoDB Atlas.
 
 Streamlit Frontend — Interactive web interface with dropdown selections for user email and product.
 
 Robust Error Handling — Handles database connection errors and missing data gracefully.
 
 Cached Resources — Optimizes performance using Streamlit’s caching for database and model initialization.



**Database Structure**

1. users Collection

Stores user profile information and preferences.

{
  "user_id": 1001,
  "email": "a.smith@example.com",
  "join_date": "2024-01-15T10:00:00Z",
  "preferences": ["Electronics", "Smart Home"]
}

2. products Collection

Stores details of each product.

{
  "product_id": "P1",
  "name": "Noise Cancelling Headphones X7",
  "category": "Electronics",
  "price": 249.99,
  "attributes": {
    "brand": "AudioTech",
    "color": "Black"
  },
  "description": "Premium wireless headphones with industry-leading noise cancellation."
}

3. user-interactions Collection

strores interacion of users - the prodcuts they purchased or viewed

{
  "user_id": 1001,
  "product_id": "P1",
  "interaction_type": "purchase",
  "timestamp": "2024-10-01T15:00:00Z",
  "value": 1
}


**Recommendation Logic**

The system combines two filtering strategies with different weights:

1. Content-Based Filtering (High Weight = 2.0)

  Focuses on the product category of the item selected by the user.
  
  Recommends similar products within the same category.
  
  Excludes the selected product itself from the final recommendations.
  
  Provides contextual reasoning such as:
  
  "You showed interest in ‘Electronics’ products like Noise Cancelling Headphones X7."

2. User-Based Collaborative Filtering (Low Weight = 1.0)

  Computes cosine similarity between users based on their interaction matrix.
  
  Identifies top 3 similar users and suggests products they have interacted with.
  
  Used as a secondary layer of personalization to refine results.
  
  Example reasoning:
  
  "Similar users to you also interacted with Smart Home Speaker Z10."

3. Final Hybrid Output

  Recommendations from both methods are combined.
  
  Each product is assigned a weighted score.
  
  Results are sorted by score (CBF first, CF next).
  
  Limited to top 5 recommendations per user-product combination


**LLM Explanation Generation**

  Each recommended product is passed to Gemini 2.0 Flash via LangChain.
  
  A concise prompt template is used to generate personalized explanations for each item.
  
  The model outputs 1–2 sentence justifications like:
  
  “The Smart Home Camera Pro fits your interest in smart home devices and is highly rated by users similar to you.”


**Algorithm Flow**

1. Initialize MongoDB Connection

  Loads data from the three collections into Pandas DataFrames.
  
  Creates a user-product interaction matrix for collaborative filtering.

2. User Input via Streamlit Sidebar

  User selects their email ID and a product name from dropdowns.

3. Hybrid Recommendation Generation

  CBF suggests products from the same category (excluding selected one).
  
  CF finds similar users using cosine similarity and suggests items they liked.
  
  Combines and weights both results (CBF = 2.0, CF = 1.0).

4. LLM Explanation

  Gemini LLM generates short personalized reasons for each recommendation.

5. Display Results

  Final table shows product name, and the LLM-generated explanation.


**Tech Stack**

| Component                 | Technology Used               |
| ------------------------- | ----------------------------- |
| **Frontend**              | Streamlit                     |
| **Database & Backend API**| MongoDB Atlas                 |
| **Backend Logic**         | Python (Pandas, Scikit-learn) |
| **AI Explanation**        | Google Gemini via LangChain   |
| **Environment Variables** | python-dotenv                 |

  
