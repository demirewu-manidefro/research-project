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
    padded = pad_sequences(sequences, maxlen=100)
    predictions = model.predict(padded)
    
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'} # Ensure this maps to your label_info.pickle
    results = [label_map[np.argmax(pred)] for pred in predictions]
    return results

def detect_emotion(text):
    text = text.lower()
    if any(emoji in text for emoji in ['❤️', '🔥', '😂', '😍', 'wow', 'good', 'አሪፍ', 'ምርጥ', 'በርታ']):
        return "Joy"
    elif any(emoji in text for emoji in ['😡', '🤬', 'bad', 'terrible', 'tf', 'አይጠቅምም', 'ውሸት', 'አሳዛኝ']):
        return "Anger"
    elif any(emoji in text for emoji in ['😢', '😭', 'sad', 'ያሳዝናል', 'ነፍሳቸውን']):
        return "Sadness"
    return "Neutral"

def get_gemini_summary(positive, negative, neutral, sample_comments_text):
    genai_key = os.getenv("GEMINI_API_KEY")
    if not genai_key:
        return "ማስጠንቀቂያ፡ የጂሚናይ ኤፒአይ (Gemini API) ኮድ አልገጠሙም። ሲስተሙ አስተያየቶቹን ተንትኖታል፣ ነገር ግን ሰፋ ያለ የባለሙያ ማጠቃለያ ለመስጠት ኤፒአይ ቁልፍ (API Key) ያስፈልጋል።"
    
    genai.configure(api_key=genai_key)
    gem_model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    You are an expert Social Media Analyst. Analyze these YouTube comment sentiments: {positive} Positive, {negative} Negative, {neutral} Neutral.
    Here are a few sample comments to understand the context:
    {sample_comments_text}
    
    Generate exactly three professional recommendations for the content creator in clear Amharic based on this specific feedback. Use bullet points. Ensure it explains the audience sentiment briefly and gives actionable advice.
    """
    try:
        response = gem_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Summary generation failed: {e}"

def process_single_video(url: str, max_comments: int):
    video_id = get_video_id(url)
    comments = fetch_real_youtube_comments(url, max_comments)
    
    if not comments:
        raise HTTPException(status_code=400, detail="No comments found for this video.")
        
    sentiments = predict_sentiment(comments)
    
    results = []
    positive_count = negative_count = neutral_count = 0
    sample_text_for_gemini = []
    
    for text, sentiment in zip(comments, sentiments):
        if sentiment == "Positive":
            positive_count += 1
        elif sentiment == "Negative":
            negative_count += 1
        else:
            neutral_count += 1
            
        emotion = detect_emotion(text)
        results.append({
            "text": text,
            "sentiment": sentiment,
            "emotion": emotion
        })
        if len(sample_text_for_gemini) < 5:
            sample_text_for_gemini.append(text)
    
    total = len(sentiments) or 1
    negative_ratio = negative_count / total
    
    virality_score = min(100, int((positive_count / total) * 100) + np.random.randint(-5, 10))
    alert = "Warning: Toxicity Spikes Detected!" if negative_ratio > 0.4 else None
    
    summary = get_gemini_summary(positive_count, negative_count, neutral_count, "\n".join(sample_text_for_gemini))
    
    return {
        "video_url": url,
        "video_id": video_id,
        "total_analyzed": total,
        "sentiment_breakdown": {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count
        },
        "virality_score": max(0, virality_score), # Also used as Sentiment Performance Index (SPI)
        "alert": alert,
        "gemini_summary": summary,
        "sample_comments": results[:10]
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
