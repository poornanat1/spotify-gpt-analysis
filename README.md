# 🎵 Spotify Playlist Analyzer

## Overview
**Spotify Playlist Analyzer** is a **FastAPI**-powered backend that analyzes **Spotify playlists**, computing **energy, danceability, and mood** using **AI embeddings, cosine similarity, and Last.fm metadata**. It also generates **poetic summaries** of the playlist’s emotional flow and visualizes it using **Three.js**.

## 🚀 Tech Stack
- **Backend**: FastAPI, Spotipy, OpenAI API, Last.fm API, NumPy
- **Frontend**: HTML, TailwindCSS, Three.js
- **AI Features**: OpenAI embeddings + cosine similarity for music feature analysis
- **Visualization**: Interactive 3D bar charts (energy, danceability, mood)

## ⚙️ Installation
```sh
# Clone the repo and install dependencies
git clone <repo-url>
cd spotify-playlist-analyzer
pip install -r requirements.txt

# Set up environment variables
echo "SPOTIFY_CLIENT_ID=your-client-id" >> .env
echo "SPOTIFY_CLIENT_SECRET=your-client-secret" >> .env
echo "OPENAI_API_KEY=your-openai-key" >> .env
echo "LASTFM_API_KEY=your-lastfm-key" >> .env

# Start the API
uvicorn analyze-playlist:app --reload
```

## 🔌 API Endpoints
```http
POST /api/analyze-playlist
```
- **Description**: Returns **track-level** energy, danceability, and mood scores.
- **Request Body**:
  ```json
  {
    "playlist_url": "https://open.spotify.com/playlist/..."
  }
  ```
- **Response**:
  ```json
  {
    "playlist_analysis": "AI-generated poetic summary...",
    "track_count": 10,
    "songs": [
      {"title": "Song 1", "artist": "Artist 1", "energy": 0.8, "danceability": 0.7, "mood": 0.9}
    ]
  }
  ```

```http
GET /health
```
- **Description**: API health check.
- **Response**:
  ```json
  { "status": "ok" }
  ```

## 🛠️ How It Works
1. **Fetches Spotify Playlist Data** via Spotipy.
2. **Retrieves Last.fm Tags** → Converts them into **embeddings**.
3. **Uses OpenAI GPT-4o Fallback** if metadata is missing.
4. **Computes Cosine Similarity** against reference embeddings.
5. **Visualizes Features** dynamically using Three.js.

## 🔍 Troubleshooting
```sh
# Playlist not found?
Ensure it’s public on Spotify.

# Invalid API keys?
Check .env variables.

# Visualization not loading?
Ensure JavaScript is enabled in your browser.
```

## 🤝 Contributors
```txt
- Poorna Natarajan - Developer
```
```
