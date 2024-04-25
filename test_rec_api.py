<<<<<<< HEAD
from fastapi import FastAPI, HTTPException
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from sklearn.preprocessing import MinMaxScaler
import webbrowser
from fastapi.responses import HTMLResponse

# Initialize FastAPI app
app = FastAPI()

# Load the dataset and pre-process it
df = pd.read_csv('Landmark_Rec.csv')

df['Features'] = df['Category'] + ' ' + df['Tags']
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define sample user history (you can replace this with actual user history data)
sample_user_history = {
    # 'Haji Ali Dargah, Worli': 0,
    'Rizvi College of Engineering - Bandra': 1,
    # 'Mahalakshmi Temple, Mahalaxmi': 1,
    # 'Wadiaji Atash Behram, Grant Road': 1,
    # 'Sanjay Gandhi National Park': 0,
    # 'Gloria Church, Byculla': 1,
    # 'Raheja Towers': 0,
    # 'Wilson College - Chowpatty': 1,
    # 'Teerthdham Mangalayatan, Vasai': 1,
    'Rajiv Gandhi Institute of Technology (RGIT) - Versova': 1,
    'M.H. Saboo Siddik College of Engineering - Byculla' : 0,
    'Elphinstone college': 0,
    'Gateway of India': 1,
    "Victoria Terminus (Chhatrapati Shivaji Maharaj Terminus)": 1,
}

# Function to get the top similar landmarks
def get_recommendations(user_history: Dict[str, int]) -> List[Tuple[str, float, str]]:
    if not user_history:
        raise HTTPException(status_code=400, detail="User history cannot be empty")
    
    total_scores = [0] * len(df)
    popularity_scores = df['PopularityScore'].tolist()
    reasons = [''] * len(df)

    for landmark_name, user_rating in user_history.items():
        closest_match = process.extractOne(landmark_name, df['Name'])[0]
        idx = df[df['Name'] == closest_match].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = [(idx, score + 1) for idx, score in sim_scores]  # Add a small constant to avoid zero scores
        sim_scores = [(idx, score * user_rating * popularity_scores[idx]) for idx, score in sim_scores]
        for idx, score in sim_scores:
            total_scores[idx] += score
            if reasons[idx] == '':
                reasons[idx] = f"Recommended due to your visit to {landmark_name}"
            else:
                # Update the reason if the current landmark has a higher similarity score
                current_landmark = df['Name'].iloc[idx]
                current_score = cosine_similarity(tfidf_matrix[df[df['Name'] == current_landmark].index[0]], tfidf_matrix[idx])
                previous_landmark = reasons[idx].replace("Recommended due to your visit to ", "")
                previous_score = cosine_similarity(tfidf_matrix[df[df['Name'] == previous_landmark].index[0]], tfidf_matrix[idx])
                if current_score > previous_score:
                    reasons[idx] = f"Recommended due to your visit to {landmark_name}"

    top_indices = sorted(range(len(total_scores)), key=lambda i: total_scores[i], reverse=True)
    top_indices = [idx for idx in top_indices if df['Name'].iloc[idx] not in user_history.keys()]
    top_landmarks = df['Name'].iloc[top_indices[:5]]
    top_scores = [total_scores[i] for i in top_indices[:5]]
    top_reasons = [reasons[i] for i in top_indices[:5]]
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform([[score] for score in top_scores])
    recommendations = [(landmark, score[0], reason) for landmark, score, reason in zip(top_landmarks, normalized_scores, top_reasons)]
    
    # If user history is less than 5 landmarks, return 5 sample landmarks with popularity score >= 8
    if len(user_history) < 5:
        sample_landmarks = df[df['PopularityScore'] >= 8].sample(5)
        recommendations = [(landmark, score, "Sample recommendation due to insufficient user history") for landmark, score in zip(sample_landmarks['Name'], sample_landmarks['PopularityScore'])]
    
    # After finding the top_landmarks for recommendation calculate similarity score each top_landmarks with history landmark
    # and whichever history landmark have highest similar chose that as reason for recommending that landmark
    for i in range(len(recommendations)):
        landmark, score, _ = recommendations[i]
        idx = df[df['Name'] == landmark].index[0]
        sim_scores = [(history_landmark, cosine_similarity(tfidf_matrix[df[df['Name'] == history_landmark].index[0]], tfidf_matrix[idx])) for history_landmark in user_history.keys()]
        most_similar_landmark = max(sim_scores, key=lambda x: x[1])[0]
        recommendations[i] = (landmark, score, f"Recommended due to your visit to {most_similar_landmark}")
    
    return recommendations


# Define the root endpoint
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <body>
            <h2>Welcome to the Landmark Recommendation API.</h2>
            <form action="/recommendations/" method="post">
                <input type="submit" value="Get Recommendations">
            </form>
        </body>
    </html>
    """

# Define the endpoint to get recommendations
@app.post("/recommendations/", response_model=List[Tuple[str, float, str]])
def recommend_landmarks() -> List[Tuple[str, float, str]]:
    recommendations = get_recommendations(sample_user_history)
    return recommendations

# Open the browser with the FastAPI server URL
webbrowser.open('http://127.0.0.1:8000/')
# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
=======
from fastapi import FastAPI, HTTPException
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from sklearn.preprocessing import MinMaxScaler
import webbrowser
from fastapi.responses import HTMLResponse

# Initialize FastAPI app
app = FastAPI()

# Load the dataset and pre-process it
df = pd.read_csv('Landmark_Rec.csv')

df['Features'] = df['Category'] + ' ' + df['Tags']
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define sample user history (you can replace this with actual user history data)
sample_user_history = {
    # 'Haji Ali Dargah, Worli': 0,
    'Rizvi College of Engineering - Bandra': 1,
    # 'Mahalakshmi Temple, Mahalaxmi': 1,
    # 'Wadiaji Atash Behram, Grant Road': 1,
    # 'Sanjay Gandhi National Park': 0,
    # 'Gloria Church, Byculla': 1,
    # 'Raheja Towers': 0,
    # 'Wilson College - Chowpatty': 1,
    # 'Teerthdham Mangalayatan, Vasai': 1,
    'Rajiv Gandhi Institute of Technology (RGIT) - Versova': 1,
    'M.H. Saboo Siddik College of Engineering - Byculla' : 0,
    'Elphinstone college': 0,
    'Gateway of India': 1,
    "Victoria Terminus (Chhatrapati Shivaji Maharaj Terminus)": 1,
}

# Function to get the top similar landmarks
def get_recommendations(user_history: Dict[str, int]) -> List[Tuple[str, float, str]]:
    if not user_history:
        raise HTTPException(status_code=400, detail="User history cannot be empty")
    
    total_scores = [0] * len(df)
    popularity_scores = df['PopularityScore'].tolist()
    reasons = [''] * len(df)

    for landmark_name, user_rating in user_history.items():
        closest_match = process.extractOne(landmark_name, df['Name'])[0]
        idx = df[df['Name'] == closest_match].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = [(idx, score + 1) for idx, score in sim_scores]  # Add a small constant to avoid zero scores
        sim_scores = [(idx, score * user_rating * popularity_scores[idx]) for idx, score in sim_scores]
        for idx, score in sim_scores:
            total_scores[idx] += score
            if reasons[idx] == '':
                reasons[idx] = f"Recommended due to your visit to {landmark_name}"
            else:
                # Update the reason if the current landmark has a higher similarity score
                current_landmark = df['Name'].iloc[idx]
                current_score = cosine_similarity(tfidf_matrix[df[df['Name'] == current_landmark].index[0]], tfidf_matrix[idx])
                previous_landmark = reasons[idx].replace("Recommended due to your visit to ", "")
                previous_score = cosine_similarity(tfidf_matrix[df[df['Name'] == previous_landmark].index[0]], tfidf_matrix[idx])
                if current_score > previous_score:
                    reasons[idx] = f"Recommended due to your visit to {landmark_name}"

    top_indices = sorted(range(len(total_scores)), key=lambda i: total_scores[i], reverse=True)
    top_indices = [idx for idx in top_indices if df['Name'].iloc[idx] not in user_history.keys()]
    top_landmarks = df['Name'].iloc[top_indices[:5]]
    top_scores = [total_scores[i] for i in top_indices[:5]]
    top_reasons = [reasons[i] for i in top_indices[:5]]
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform([[score] for score in top_scores])
    recommendations = [(landmark, score[0], reason) for landmark, score, reason in zip(top_landmarks, normalized_scores, top_reasons)]
    
    # If user history is less than 5 landmarks, return 5 sample landmarks with popularity score >= 8
    if len(user_history) < 5:
        sample_landmarks = df[df['PopularityScore'] >= 8].sample(5)
        recommendations = [(landmark, score, "Sample recommendation due to insufficient user history") for landmark, score in zip(sample_landmarks['Name'], sample_landmarks['PopularityScore'])]
    
    # After finding the top_landmarks for recommendation calculate similarity score each top_landmarks with history landmark
    # and whichever history landmark have highest similar chose that as reason for recommending that landmark
    for i in range(len(recommendations)):
        landmark, score, _ = recommendations[i]
        idx = df[df['Name'] == landmark].index[0]
        sim_scores = [(history_landmark, cosine_similarity(tfidf_matrix[df[df['Name'] == history_landmark].index[0]], tfidf_matrix[idx])) for history_landmark in user_history.keys()]
        most_similar_landmark = max(sim_scores, key=lambda x: x[1])[0]
        recommendations[i] = (landmark, score, f"Recommended due to your visit to {most_similar_landmark}")
    
    return recommendations


# Define the root endpoint
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <body>
            <h2>Welcome to the Landmark Recommendation API.</h2>
            <form action="/recommendations/" method="post">
                <input type="submit" value="Get Recommendations">
            </form>
        </body>
    </html>
    """

# Define the endpoint to get recommendations
@app.post("/recommendations/", response_model=List[Tuple[str, float, str]])
def recommend_landmarks() -> List[Tuple[str, float, str]]:
    recommendations = get_recommendations(sample_user_history)
    return recommendations

# Open the browser with the FastAPI server URL
webbrowser.open('http://127.0.0.1:8000/')
# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
>>>>>>> e6355cffc4aa719fd962c71d273eca1d71421c21
