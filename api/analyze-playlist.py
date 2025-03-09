from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModelxs
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from openai import OpenAI
import logging
from typing import List, Dict, Any
import asyncio
import requests
import numpy as np
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")  

def cosine_similarity(vec1, vec2):
    if vec1.ndim == 1:
        vec1 = vec1.reshape(1, -1)
    if vec2.ndim == 1:
        vec2 = vec2.reshape(1, -1)
    dot_product = np.dot(vec1, vec2.T)  
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return np.array([[0.0]])
    return dot_product / (norm1 * norm2)


app = FastAPI(title="Spotify Playlist Analyzer")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PlaylistRequest(BaseModel):
    playlist_url: str


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"}
    )


def get_spotify_client():
    try:
        client = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        ))
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Spotify client: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not connect to Spotify")


def get_spotify_token() -> str:
    try:
        credentials_manager = SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        )
        token = credentials_manager.get_access_token(as_dict=False)
        return token
    except Exception as e:
        logger.error(f"Failed to obtain Spotify token: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not obtain Spotify token")


def get_openai_client():
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not connect to OpenAI")


def get_embedding(text: str, engine="text-embedding-ada-002"):
    openai_client = get_openai_client()
    response = openai_client.embeddings.create(
        input=text,
        model=engine
    )
    return response.data[0].embedding


reference_energy_text = "high energy, loud, intense, upbeat, dynamic, driving, fast"
reference_dance_text = "danceable, groovy, rhythmic, catchy, club, beat-driven, funky"

reference_positive_mood_text = (
    "cheerful, upbeat, radiant, joyful, exuberant, optimistic, lively, bright, inspiring, energetic, "
    "happy, positive, euphoric, elated, ecstatic, blissful, triumphant, hopeful"
)
reference_negative_mood_text = (
    "melancholic, sad, gloomy, depressing, somber, dark, moody, introspective, brooding, emotional, "
    "anxious, tense, angry, frustrated, bitter, pensive, nostalgic, wistful, yearning, lonely"
)


reference_energy_embedding = get_embedding(reference_energy_text)
reference_dance_embedding = get_embedding(reference_dance_text)
reference_positive_mood_embedding = get_embedding(reference_positive_mood_text)
reference_negative_mood_embedding = get_embedding(reference_negative_mood_text)


def get_lastfm_track_tags(artist: str, track: str, api_key: str) -> List[str]:
    url = (
        f"http://ws.audioscrobbler.com/2.0/"
        f"?method=track.getInfo&api_key={api_key}&artist={artist}&track={track}&format=json"
    )
    response = requests.get(url)
    try:
        data = response.json()
    except Exception:
        return []
    if not isinstance(data, dict) or "error" in data:
        return []
    tags = []
    try:
        for tag in data["track"]["toptags"]["tag"]:
            tags.append(tag["name"])
    except (KeyError, TypeError):
        pass
    return tags

def get_lastfm_album_tags(artist: str, album: str, api_key: str) -> List[str]:
    url = (
        f"http://ws.audioscrobbler.com/2.0/"
        f"?method=album.getInfo&api_key={api_key}&artist={artist}&album={album}&format=json"
    )
    response = requests.get(url)
    try:
        data = response.json()
    except Exception:
        return []
    if not isinstance(data, dict) or "error" in data:
        return []
    tags = []
    try:
        for tag in data["album"]["tags"]["tag"]:
            tags.append(tag["name"])
    except (KeyError, TypeError):
        pass
    return tags

def get_lastfm_artist_tags(artist: str, api_key: str) -> List[str]:
    url = (
        f"http://ws.audioscrobbler.com/2.0/"
        f"?method=artist.getInfo&api_key={api_key}&artist={artist}&format=json"
    )
    response = requests.get(url)
    try:
        data = response.json()
    except Exception:
        return []
    if not isinstance(data, dict) or "error" in data:
        return []
    tags = []
    try:
        for tag in data["artist"]["tags"]["tag"]:
            tags.append(tag["name"])
    except (KeyError, TypeError):
        pass
    return tags


async def fallback_llm_scores(artist: str, track: str, album: str) -> Dict[str, float]:
    """
    Calls GPT-4 to estimate numeric scores for energy, danceability, and mood.
    Returns a dict with keys "energy", "danceability", and "mood", each between 0.0 and 1.0.
    """
    client = get_openai_client()
    system_prompt = (
        "You are a helpful assistant that returns approximate numeric music scores.\n"
        "Return valid JSON with exactly these keys: 'energy', 'danceability', 'mood'.\n"
        "All values must be floats between 0.0 and 1.0. For mood, 0.0 means very negative, 0.5 neutral, and 1.0 very positive.\n"
        "Return only the JSON without any additional text."
    )
    user_prompt = (
        f"Given the track '{track}' by {artist} from the album '{album}', "
        "estimate its scores as follows:\n"
        "- Energy (0.0 = very calm, 1.0 = very energetic)\n"
        "- Danceability (0.0 = not danceable, 1.0 = very danceable)\n"
        "- Mood (0.0 = very negative/sad, 0.5 = neutral, 1.0 = very positive/happy)\n"
        "Return only a JSON object with the keys and numeric values."
    )

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        scores = json.loads(content)
        
        for key in ["energy", "danceability", "mood"]:
            if key not in scores:
                scores[key] = 0.5
            else:
                scores[key] = min(max(float(scores[key]), 0.0), 1.0)
        return scores
    except Exception as e:
        logger.error(f"LLM fallback error: {e}")
        return {"energy": 0.5, "danceability": 0.5, "mood": 0.5}


async def compute_embedding_features_for_track(
    track: Dict[str, Any],
    lastfm_api_key: str,
    ref_energy: list,
    ref_dance: list,
    ref_positive_mood: list,
    ref_negative_mood: list
) -> Dict[str, Any]:
    """
    1. Fetch track tags from Last.fm.
    2. If none found, fetch album then artist tags.
    3. If still none, use LLM fallback.
    4. Otherwise, compute cosine similarity for energy, danceability, and contrasting mood.
    """
    artist = track["artist"]
    name = track["name"]
    album = track.get("album", "")

    
    tags = await asyncio.to_thread(get_lastfm_track_tags, artist, name, lastfm_api_key)
    
    if not tags and album:
        tags = await asyncio.to_thread(get_lastfm_album_tags, artist, album, lastfm_api_key)
    
    if not tags:
        primary_artist = artist.split(",")[0].strip()
        tags = await asyncio.to_thread(get_lastfm_artist_tags, primary_artist, lastfm_api_key)

    if not tags:
        
        llm_scores = await fallback_llm_scores(artist, name, album)
        energy_score = llm_scores["energy"]
        danceability_score = llm_scores["danceability"]
        mood_score = llm_scores["mood"]
    else:
        
        text_data = " ".join(tags)
        track_embedding = await asyncio.to_thread(get_embedding, text_data)

        energy_score = cosine_similarity(
            np.array(track_embedding), np.array(ref_energy)
        )[0][0]
        danceability_score = cosine_similarity(
            np.array(track_embedding), np.array(ref_dance)
        )[0][0]
        
        positive_mood_score = cosine_similarity(
            np.array(track_embedding), np.array(ref_positive_mood)
        )[0][0]
        negative_mood_score = cosine_similarity(
            np.array(track_embedding), np.array(ref_negative_mood)
        )[0][0]
        
        mood_score = (positive_mood_score - negative_mood_score + 1) / 2
        mood_score = max(0.0, min(1.0, mood_score))

    logger.info(f"[{name}] Energy: {energy_score:.2f}, Dance: {danceability_score:.2f}, Mood: {mood_score:.2f}")
    return {
        "energy": energy_score,
        "danceability": danceability_score,
        "mood": mood_score,
        "source": "embedding" if tags else "LLM fallback"
    }


async def get_embedding_features(
    tracks: List[Dict[str, Any]],
    lastfm_api_key: str,
    ref_energy: list,
    ref_dance: list,
    ref_positive_mood: list,
    ref_negative_mood: list
) -> List[Dict[str, Any]]:
    tasks = [
        compute_embedding_features_for_track(
            track, lastfm_api_key, ref_energy, ref_dance, ref_positive_mood, ref_negative_mood
        )
        for track in tracks
    ]
    return await asyncio.gather(*tasks)


def extract_playlist_id(playlist_url: str) -> str:
    try:
        if "spotify.com" not in playlist_url:
            raise ValueError("Not a Spotify URL")
        if "playlist/" not in playlist_url:
            raise ValueError("Not a playlist URL")
        playlist_id = playlist_url.split("playlist/")[-1]
        if "?" in playlist_id:
            playlist_id = playlist_id.split("?")[0]
        return playlist_id
    except Exception as e:
        logger.error(f"Error extracting playlist ID: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid Spotify playlist URL")

def get_playlist_tracks(playlist_url: str) -> List[Dict[str, Any]]:
    sp = get_spotify_client()
    try:
        playlist_id = extract_playlist_id(playlist_url)
        results = sp.playlist_tracks(playlist_id, limit=30)
        tracks = []
        for item in results['items']:
            if not item or not item.get('track'):
                continue
            track = item['track']
            tracks.append({
                "id": track.get("id"),
                "name": track.get("name", "Unknown Track"),
                "artist": ", ".join([artist.get("name", "Unknown Artist") for artist in track.get("artists", [])]),
                "album": track.get("album", {}).get("name", "")
            })
        return tracks
    except spotipy.exceptions.SpotifyException as e:
        logger.error(f"Spotify API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error connecting to Spotify API")
    except Exception as e:
        logger.error(f"Unexpected error fetching playlist tracks: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

def generate_analysis(playlist_description: str) -> str:
    """
    Calls GPT-4 (or GPT-4o if available) to analyze the playlist.
    """
    try:
        client = get_openai_client()
        prompt = f"""
Analyze this playlist in a single short multi-paragraph essay without markdown formatting:
{playlist_description}

Include:
- Emotional progression
- Themes across songs
- Transitions between tracks
"""
        
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a skilled music analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            timeout=30
        )
        if not completion or not completion.choices or not completion.choices[0].message:
            raise ValueError("Empty response from OpenAI")
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating analysis: {str(e)}")
        return "Unable to generate analysis for this playlist."

@app.get("/")
async def root():
    return {"message": "Welcome to Spotify Playlist Analyzer API"}

@app.post("/api/analyze-playlist")
async def analyze_playlist(request: PlaylistRequest):
    try:
        if not request.playlist_url:
            return JSONResponse(status_code=400, content={"detail": "Playlist URL is required"})
        songs = get_playlist_tracks(request.playlist_url)
        if not songs:
            return JSONResponse(status_code=404, content={"detail": "No songs found in the playlist"})

        playlist_description = "\n".join(
            [f"{i+1}. \"{song['name']}\" by {song['artist']}" for i, song in enumerate(songs)]
        )

        
        analysis_task = asyncio.to_thread(generate_analysis, playlist_description)
        embedding_features_task = get_embedding_features(
            songs,
            LASTFM_API_KEY,
            reference_energy_embedding,
            reference_dance_embedding,
            reference_positive_mood_embedding,
            reference_negative_mood_embedding
        )

        analysis, features = await asyncio.gather(analysis_task, embedding_features_task)

        songs_with_features = []
        for i, song in enumerate(songs):
            feat = features[i] if i < len(features) else {"energy": 0.5, "danceability": 0.5, "mood": 0.5, "source": "default"}
            enhanced_song = {
                "title": song["name"],
                "artist": song["artist"],
                "energy": feat.get("energy", 0.5),
                "danceability": feat.get("danceability", 0.5),
                "mood": feat.get("mood", 0.5),
                "source": feat.get("source", "default")
            }
            songs_with_features.append(enhanced_song)

        return {
            "playlist_analysis": analysis,
            "track_count": len(songs),
            "feature_source": "embedding/LLM fallback",
            "songs": songs_with_features
        }
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    except Exception as e:
        logger.error(f"Unhandled error in analyze_playlist: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"An unexpected error occurred: {str(e)}"})

@app.get("/health")
async def health_check():
    return {"status": "ok"}
