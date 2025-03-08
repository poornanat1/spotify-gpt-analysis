from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from openai import OpenAI
import logging
from typing import List, Dict, Any, Optional
import asyncio
from fastapi.exceptions import RequestValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="Spotify Playlist Analyzer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PlaylistRequest(BaseModel):
    playlist_url: str

# Error handler for any unhandled exceptions
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"}
    )

# Error handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request parameters"}
    )

# Initialize Spotify client only when needed
def get_spotify_client():
    """Get or create a Spotify client."""
    try:
        client = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        ))
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Spotify client: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not connect to Spotify")

# Initialize OpenAI client only when needed
def get_openai_client():
    """Get or create an OpenAI client."""
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not connect to OpenAI")

def extract_playlist_id(playlist_url: str) -> str:
    """Extract the playlist ID from a Spotify URL."""
    try:
        if "spotify.com" not in playlist_url:
            raise ValueError("Not a Spotify URL")
            
        if "playlist/" not in playlist_url:
            raise ValueError("Not a playlist URL")
            
        # Handle different URL formats
        if "playlist/" in playlist_url:
            playlist_id = playlist_url.split("playlist/")[-1]
            # Remove any query parameters
            if "?" in playlist_id:
                playlist_id = playlist_id.split("?")[0]
            return playlist_id
        else:
            raise ValueError("Could not extract playlist ID")
    except Exception as e:
        logger.error(f"Error extracting playlist ID: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid Spotify playlist URL")

def get_playlist_tracks(playlist_url: str) -> List[Dict[str, Any]]:
    """Fetch tracks from a Spotify playlist."""
    sp = get_spotify_client()
    try:
        playlist_id = extract_playlist_id(playlist_url)
        results = sp.playlist_tracks(playlist_id, limit=30)  # Limit to 30 tracks to avoid token limits
        
        tracks = []
        for item in results['items']:
            if not item or not item.get('track'):
                continue
                
            track = item['track']
            tracks.append({
                'name': track.get('name', 'Unknown Track'),
                'artist': ', '.join([artist.get('name', 'Unknown Artist') for artist in track.get('artists', [])]),
                'album': track.get('album', {}).get('name', 'Unknown Album'),
                'duration_ms': track.get('duration_ms', 0)
            })
                
        return tracks
        
    except spotipy.exceptions.SpotifyException as e:
        logger.error(f"Spotify API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error connecting to Spotify API")
    except Exception as e:
        logger.error(f"Unexpected error fetching playlist tracks: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

FEW_SHOT_EXAMPLES = """
Example 1:
Playlist:
1. "Bohemian Rhapsody" by Queen
2. "Stairway to Heaven" by Led Zeppelin
3. "Hotel California" by Eagles

Analysis:
A journey through time and space, this playlist opens with the operatic grandeur of "Bohemian Rhapsody," setting a tone of dramatic introspection. "Stairway to Heaven" seamlessly carries forward the mystical, philosophical reflection, its gradual crescendo symbolizing an ascent to enlightenment. Finally, "Hotel California" brings a haunting resolution, its enigmatic lyrics and hypnotic guitar solos leaving the listener in a dreamlike limbo.

Example 2:
Playlist:
1. "Someone Like You" by Adele
2. "Fix You" by Coldplay
3. "Fast Car" by Tracy Chapman

Analysis:
A poignant meditation on love, loss, and hope, this playlist flows with deep emotional undercurrents. "Someone Like You" opens the wounds of heartbreak, its raw vulnerability mirrored by "Fix You," which offers comfort and gradual healing. "Fast Car" closes the journey, embodying longing and the search for freedom, ending on a note of bittersweet acceptance.
"""

def generate_analysis(playlist_description: str) -> str:
    """Generate an analysis of the playlist using OpenAI."""
    try:
        client = get_openai_client()
        
        # Simplified prompt to reduce token usage
        prompt = f"""
Analyze this playlist in a single short multi-paragraph essay wothout markdown formatting:
{playlist_description}

Include:
- Emotional progression
- Themes across songs
- Transitions between tracks
"""

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a skilled music analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,  
            timeout=30  # Add timeout to prevent hanging requests
        )
        
        if not completion or not completion.choices or not completion.choices[0].message:
            raise ValueError("Empty response from OpenAI")
            
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating analysis: {str(e)}")
        return "Unable to generate analysis for this playlist."  # Fallback response

@app.get("/")
async def root():
    return {"message": "Welcome to Spotify Playlist Analyzer API"}

@app.post("/api/analyze-playlist")
async def analyze_playlist(request: PlaylistRequest):
    """Analyze a Spotify playlist and return a narrative analysis."""
    try:
        # Basic validation
        if not request.playlist_url:
            return JSONResponse(
                status_code=400,
                content={"detail": "Playlist URL is required"}
            )
        
        # Fetch tracks
        songs = get_playlist_tracks(request.playlist_url)
        
        if not songs:
            return JSONResponse(
                status_code=404,
                content={"detail": "No songs found in the playlist"}
            )
        
        # Format the playlist
        playlist_description = "\n".join(
            [f"{i+1}. \"{song['name']}\" by {song['artist']}" for i, song in enumerate(songs)]
        )
        
        # Generate analysis
        analysis = generate_analysis(playlist_description)
        
        # Return simple dict
        return {
            "playlist_analysis": analysis,
            "track_count": len(songs)
        }
    except HTTPException as e:
        # Convert HTTPException to JSONResponse
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail}
        )
    except Exception as e:
        logger.error(f"Unhandled error in analyze_playlist: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred"}
        )

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}