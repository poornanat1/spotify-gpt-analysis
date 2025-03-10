from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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

class Config:
    SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET") 
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")

class ReferenceText:
    ENERGY = "high energy, loud, intense, upbeat, dynamic, driving, fast"
    DANCE = "danceable, groovy, rhythmic, catchy, club, beat-driven, funky"
    POSITIVE_MOOD = (
        "cheerful, upbeat, radiant, joyful, exuberant, optimistic, lively, bright, inspiring, energetic, "
        "happy, positive, euphoric, elated, ecstatic, blissful, triumphant, hopeful"
    )
    NEGATIVE_MOOD = (
        "melancholic, sad, gloomy, depressing, somber, dark, moody, introspective, brooding, emotional, "
        "anxious, tense, angry, frustrated, bitter, pensive, nostalgic, wistful, yearning, lonely"
    )

class EmbeddingService:
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.reference_embeddings = self._initialize_reference_embeddings()

    def _initialize_reference_embeddings(self):
        return {
            'energy': self._get_embedding(ReferenceText.ENERGY),
            'dance': self._get_embedding(ReferenceText.DANCE),
            'positive_mood': self._get_embedding(ReferenceText.POSITIVE_MOOD),
            'negative_mood': self._get_embedding(ReferenceText.NEGATIVE_MOOD)
        }

    def _get_embedding(self, text: str, engine="text-embedding-ada-002"):
        response = self.openai_client.embeddings.create(
            input=text,
            model=engine
        )
        return response.data[0].embedding

    async def get_embedding_async(self, text: str, engine="text-embedding-ada-002"):
        return await asyncio.to_thread(self._get_embedding, text, engine)

    @staticmethod
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

class LastFMService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    async def get_track_tags(self, artist: str, track: str) -> List[str]:
        url = (
            f"http://ws.audioscrobbler.com/2.0/"
            f"?method=track.getInfo&api_key={self.api_key}&artist={artist}&track={track}&format=json"
        )
        return await self._fetch_tags_async(url, "track")

    async def get_album_tags(self, artist: str, album: str) -> List[str]:
        url = (
            f"http://ws.audioscrobbler.com/2.0/"
            f"?method=album.getInfo&api_key={self.api_key}&artist={artist}&album={album}&format=json"
        )
        return await self._fetch_tags_async(url, "album")

    async def get_artist_tags(self, artist: str) -> List[str]:
        url = (
            f"http://ws.audioscrobbler.com/2.0/"
            f"?method=artist.getInfo&api_key={self.api_key}&artist={artist}&format=json"
        )
        return await self._fetch_tags_async(url, "artist")

    async def get_all_tags_parallel(self, artist: str, track: str, album: str = "") -> List[str]:
        tasks = [
            self.get_track_tags(artist, track)
        ]
        
        if album:
            tasks.append(self.get_album_tags(artist, album))
            
        primary_artist = artist.split(",")[0].strip()
        tasks.append(self.get_artist_tags(primary_artist))
        
        results = await asyncio.gather(*tasks)
        
        for tags in results:
            if tags:
                return tags
        return []

    async def _fetch_tags_async(self, url: str, entity_type: str) -> List[str]:
        try:
            response = await asyncio.to_thread(self.session.get, url)
            data = response.json()
            if not isinstance(data, dict) or "error" in data:
                return []
            return [tag["name"] for tag in data[entity_type]["tags"]["tag"]]
        except Exception as e:
            logger.error(f"Error fetching {entity_type} tags: {str(e)}")
            return []

class SpotifyService:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.client = self._get_client()

    def _get_client(self):
        try:
            return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            ))
        except Exception as e:
            logger.error(f"Failed to initialize Spotify client: {str(e)}")
            raise HTTPException(status_code=500, detail="Could not connect to Spotify")

    def extract_playlist_id(self, playlist_url: str) -> str:
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

    def get_playlist_tracks(self, playlist_url: str) -> List[Dict[str, Any]]:
        try:
            playlist_id = self.extract_playlist_id(playlist_url)
            playlist_info = self.client.playlist(playlist_id)
            
            if playlist_info.get("public") is False:
                raise HTTPException(status_code=400, detail=(
                    "The playlist you provided is private or unavailable. "
                    "Please make sure the playlist is public so it can be analyzed. "
                    "You can change its visibility in the Spotify app by selecting 'Make Public' in the playlist settings."
                ))
                
            results = self.client.playlist_tracks(playlist_id, limit=30)
            return [
                {
                    "id": track["track"].get("id"),
                    "name": track["track"].get("name", "Unknown Track"),
                    "artist": ", ".join([artist.get("name", "Unknown Artist") for artist in track["track"].get("artists", [])]),
                    "album": track["track"].get("album", {}).get("name", "")
                }
                for track in results['items']
                if track and track.get('track')
            ]
        except spotipy.exceptions.SpotifyException as e:
            logger.error(f"Spotify API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error connecting to Spotify API")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching playlist tracks: {str(e)}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred")

class OpenAIService:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    async def generate_analysis(self, playlist_description: str) -> str:
        try:
            prompt = f"""
            Analyze this playlist in a single short multi-paragraph essay without markdown formatting:
            {playlist_description}

            Include:
            - Emotional progression
            - Themes across songs
            - Transitions between tracks
            """
            
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a skilled music analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                timeout=30
            )
            
            if not completion or not completion.choices or not completion.choices[0].message:
                raise ValueError("Empty response from OpenAI")
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            return "Unable to generate analysis for this playlist."

    async def fallback_llm_scores(self, artist: str, track: str, album: str) -> Dict[str, float]:
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
                self.client.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            content = response.choices[0].message.content.strip()
            scores = json.loads(content)
            
            return {
                key: min(max(float(scores.get(key, 0.5)), 0.0), 1.0)
                for key in ["energy", "danceability", "mood"]
            }
        except Exception as e:
            logger.error(f"LLM fallback error: {e}")
            return {"energy": 0.5, "danceability": 0.5, "mood": 0.5}

class PlaylistAnalyzer:
    def __init__(self):
        self.spotify_service = SpotifyService(Config.SPOTIFY_CLIENT_ID, Config.SPOTIFY_CLIENT_SECRET)
        self.openai_service = OpenAIService(Config.OPENAI_API_KEY)
        self.lastfm_service = LastFMService(Config.LASTFM_API_KEY)
        self.embedding_service = EmbeddingService(self.openai_service.client)

    async def analyze(self, playlist_url: str):
        songs = self.spotify_service.get_playlist_tracks(playlist_url)
        if not songs:
            return JSONResponse(status_code=404, content={"detail": "No songs found in the playlist"})

        playlist_description = "\n".join(
            [f"{i+1}. \"{song['name']}\" by {song['artist']}" for i, song in enumerate(songs)]
        )

        analysis_task = self.openai_service.generate_analysis(playlist_description)
        features_task = self._get_features_for_songs(songs)

        analysis, features = await asyncio.gather(analysis_task, features_task)

        songs_with_features = [
            {
                "title": song["name"],
                "artist": song["artist"],
                "energy": feat.get("energy", 0.5),
                "danceability": feat.get("danceability", 0.5),
                "mood": feat.get("mood", 0.5),
                "source": feat.get("source", "default")
            }
            for song, feat in zip(songs, features)
        ]

        return {
            "playlist_analysis": analysis,
            "track_count": len(songs),
            "feature_source": "embedding/LLM fallback",
            "songs": songs_with_features
        }

    async def _get_features_for_songs(self, songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(10)
        
        async def process_with_semaphore(song):
            async with semaphore:
                return await self._compute_features_for_track(song)
        
        tasks = [process_with_semaphore(song) for song in songs]
        return await asyncio.gather(*tasks)

    async def _compute_features_for_track(self, track: Dict[str, Any]) -> Dict[str, Any]:
        artist = track["artist"]
        name = track["name"]
        album = track.get("album", "")

        tags = await self.lastfm_service.get_all_tags_parallel(artist, name, album)

        if not tags:
            return await self.openai_service.fallback_llm_scores(artist, name, album)

        text_data = " ".join(tags)
        track_embedding = await self.embedding_service.get_embedding_async(text_data)
        track_embedding_array = np.array(track_embedding)
        
        reference_embeddings = self.embedding_service.reference_embeddings
        
        async def calculate_similarity(ref_key):
            ref_embedding = np.array(reference_embeddings[ref_key])
            return await asyncio.to_thread(
                lambda: self.embedding_service.cosine_similarity(
                    track_embedding_array, ref_embedding
                )[0][0]
            )
        
        energy_task = calculate_similarity('energy')
        dance_task = calculate_similarity('dance')
        positive_task = calculate_similarity('positive_mood')
        negative_task = calculate_similarity('negative_mood')
        
        energy_score, danceability_score, positive_score, negative_score = await asyncio.gather(
            energy_task, dance_task, positive_task, negative_task
        )

        mood_score = max(0.0, min(1.0, (positive_score - negative_score + 1) / 2))

        logger.info(f"[{name}] Energy: {energy_score:.2f}, Dance: {danceability_score:.2f}, Mood: {mood_score:.2f}")
        
        return {
            "energy": energy_score,
            "danceability": danceability_score,
            "mood": mood_score,
            "source": "embedding"
        }

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

analyzer = PlaylistAnalyzer()

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"}
    )

@app.get("/")
async def root():
    return {"message": "Welcome to Spotify Playlist Analyzer API"}

@app.post("/api/analyze-playlist")
async def analyze_playlist(request: PlaylistRequest):
    try:
        if not request.playlist_url:
            return JSONResponse(status_code=400, content={"detail": "Playlist URL is required"})
        return await analyzer.analyze(request.playlist_url)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    except Exception as e:
        logger.error(f"Unhandled error in analyze_playlist: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"An unexpected error occurred: {str(e)}"})

@app.get("/health")
async def health_check():
    return {"status": "ok"}
