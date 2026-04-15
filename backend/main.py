import os
import pickle
import json
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
import datetime
import dotenv
from itertools import islice

dotenv.load_dotenv()

app = FastAPI(title="Intelligent YouTube Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
MODEL_PATH = "../amharic_sentiment_lstm_model.keras"
TOKENIZER_PATH = "../tokenizer.pickle"

model = None
tokenizer = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Model and Tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

class AnalyzeRequest(BaseModel):
    youtube_url: str
    max_comments: int = 100

class CompareRequest(BaseModel):
    youtube_urls: List[str]
    max_comments: int = 50

def get_video_id(url: str):
    if "v=" in url:
        return url.split("v=")[1][:11]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1][:11]
    return url

def fetch_real_youtube_comments(youtube_url: str, max_comments: int = 100):
    try:
        downloader = YoutubeCommentDownloader()
        # Sort by recent to simulate real-time monitoring of the latest feedback
        comments_generator = downloader.get_comments_from_url(youtube_url, sort_by=SORT_BY_RECENT)
        comments = []
        for comment in islice(comments_generator, max_comments):
            comments.append(comment['text'])
        return comments
    except Exception as e:
        print(f"YouTube Fetch Error: {e}")
        raise HTTPException(status_code=400, detail="Could not fetch comments. Make sure the video is public and has comments enabled.")

def predict_sentiment(texts):
    if not model or not tokenizer:
        return [np.random.choice(["Positive", "Negative", "Neutral"]) for _ in texts]
    
    # Real Model Inference
    sequences = tokenizer.texts_to_sequences(texts)
    # Most Jupyter-trained LSTMs use padding, assuming 100 from earlier tests.
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    predictions = model.predict(padded)
    
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'} # Ensure this maps to your label_info.pickle
    results = [label_map[np.argmax(pred)] for pred in predictions]
    return results

def detect_emotion(text):
    text = text.lower()
    if any(emoji in text for emoji in ['❤️', '🔥', '😂', '😍', 'wow', 'good', 'አሪፍ', 'ምርጥ', 'በርታ', '👏', '🙏', '🙌', '🤍', '🎉']):
        return "Joy"
    elif any(emoji in text for emoji in ['😡', '🤬', 'bad', 'terrible', 'tf', 'አይጠቅምም', 'ውሸት', 'አሳዛኝ']):
        return "Anger"
    elif any(emoji in text for emoji in ['😢', '😭', 'sad', 'ያሳዝናል', 'ነፍሳቸውን']):
        return "Sadness"
    return "Neutral"

def get_gemini_intelligence(positive, negative, neutral, raw_comments):
    genai_key = os.getenv("GEMINI_API_KEY")
    if not genai_key:
        return {
            "summary": "ማስጠንቀቂያ፡ የጂሚናይ ኤፒአይ ቁልፍ አልተገኘም።",
            "themes": [],
            "recommendations": [],
            "ai_virality_adjustment": 0,
            "sample_analysis": []
        }
    
    genai.configure(api_key=genai_key)
    gem_model = genai.GenerativeModel('gemini-1.5-flash')
    
    comments_text = "\n".join([f"- {c}" for c in raw_comments])
    
    prompt = f"""
    You are an Expert Amharic Social Media Intelligence Engine. Analyze this YouTube data:
    - Sentiment Stats (from LSTM): {positive} Positive, {negative} Negative, {neutral} Neutral.
    - Recent Comments for Deep Contextual Analysis:
    {comments_text}
    
    Instructions:
    1. Look beyond the raw stats. If comments are full of "Amen", "Blessings", or "Congrats", classify them as high-passion Positive.
    2. Identify specific Amharic cultural context in the comments.
    
    Return a JSON object with exactly these keys:
    1. "summary": A brief (2-sentence) executive summary in Amharic explaining the overall audience mood.
    2. "themes": A list of 3-4 recurring themes or topics mentioned in the comments (in Amharic).
    3. "recommendations": A list of 3 professional, actionable tips for the content creator (in Amharic).
    4. "ai_virality_adjustment": A number between -20 and +20 based on the intensity of comments. 
    5. "sample_analysis": A list of objects for the first 10 comments provided. Each object MUST have:
       - "text": The original comment text.
       - "sentiment": One of "Positive", "Negative", or "Neutral" (Be accurate based on context).
       - "emotion": One of "Joy", "Anger", "Sadness", "Fear", "Surprise", or "Neutral".
    
    Return ONLY valid JSON.
    """
    try:
        response = gem_model.generate_content(prompt)
        content = response.text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        return json.loads(content)
    except Exception as e:
        print(f"Gemini Intelligence Error: {e}")
        return {
            "summary": "Intelligence generation encountered an error.",
            "themes": ["Error in theme extraction"],
            "recommendations": ["Ensure API key is valid"],
            "ai_virality_adjustment": 0,
            "sample_analysis": [{"text": c, "sentiment": "Neutral", "emotion": "Neutral"} for c in raw_comments[:10]]
        }

def process_single_video(url: str, max_comments: int):
    video_id = get_video_id(url)
    comments = fetch_real_youtube_comments(url, max_comments)
    
    if not comments:
        raise HTTPException(status_code=400, detail="No comments found for this video.")
        
    sentiments = predict_sentiment(comments)
    
    positive_count = negative_count = neutral_count = 0
    for sentiment in sentiments:
        if sentiment == "Positive":
            positive_count += 1
        elif sentiment == "Negative":
            negative_count += 1
        else:
            neutral_count += 1
    
    total = len(sentiments) or 1
    
    # Get Intelligent Insights and AI-powered comment analysis for samples
    intelligence = get_gemini_intelligence(positive_count, negative_count, neutral_count, comments[:15])
    
    # Intelligent Virality Score
    base_virality = int((positive_count / total) * 100)
    virality_score = base_virality + intelligence.get("ai_virality_adjustment", 0)
    virality_score = max(0, min(100, virality_score))
    
    negative_ratio = negative_count / total
    alert = "Warning: Toxicity Spikes Detected!" if negative_ratio > 0.4 else None
    
    return {
        "video_url": url,
        "video_id": video_id,
        "analysis_timestamp": datetime.datetime.now().isoformat(),
        "total_analyzed": total,
        "sentiment_breakdown": {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count
        },
        "virality_score": virality_score,
        "alert": alert,
        "intelligence": intelligence,
        "sample_comments": intelligence.get("sample_analysis", [])
    }

@app.post("/api/analyze")
async def analyze_video(request: AnalyzeRequest):
    return process_single_video(request.youtube_url, request.max_comments)

@app.post("/api/compare")
async def compare_videos(request: CompareRequest):
    comparison_results = []
    for url in request.youtube_urls:
        try:
            data = process_single_video(url, request.max_comments)
            comparison_results.append({
                "video_url": data["video_url"],
                "video_id": data["video_id"],
                "spi_score": data["virality_score"],
                "positive_ratio": data["sentiment_breakdown"]["positive"] / data["total_analyzed"],
                "total_analyzed": data["total_analyzed"]
            })
        except Exception as e:
            comparison_results.append({
                "video_url": url,
                "error": str(e)
            })
            
    # Rank by SPI (Sentiment Performance Index)
    ranked_results = sorted([r for r in comparison_results if "error" not in r], key=lambda x: x["spi_score"], reverse=True)
    return {
        "ranked_videos": ranked_results,
        "errors": [r for r in comparison_results if "error" in r]
    }

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Intelligent YouTube Analytics API is running."}
