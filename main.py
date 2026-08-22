import os
import pickle
import asyncio
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# ==========================================================
# ENVIRONMENT
# ==========================================================

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")


# ==========================================================
# CONFIG
# ==========================================================

TMDB_BASE = "https://api.themoviedb.org/3"

TMDB_IMG_500 = (
    "https://image.tmdb.org/t/p/w500"
)

TMDB_IMG_ORIGINAL = (
    "https://image.tmdb.org/t/p/original"
)

TMDB_TIMEOUT = 15.0


# ==========================================================
# FASTAPI APP
# ==========================================================

app = FastAPI(
    title="Movie Recommender API",
    version="4.0",
    description="Movie recommendation backend using TMDB + TF-IDF",
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# FILE PATHS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DF_PATH = os.path.join(
    BASE_DIR,
    "df.pkl"
)

INDICES_PATH = os.path.join(
    BASE_DIR,
    "indices.pkl"
)

TFIDF_MATRIX_PATH = os.path.join(
    BASE_DIR,
    "tfidf_matrix.pkl"
)

TFIDF_PATH = os.path.join(
    BASE_DIR,
    "tfidf.pkl"
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "movies_metadata.csv"
)


# ==========================================================
# GLOBAL ML OBJECTS
# ==========================================================

df: Optional[pd.DataFrame] = None

indices_obj: Any = None

tfidf_matrix: Any = None

tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# ==========================================================
# SIMPLE IN-MEMORY CACHE
# ==========================================================

HOME_CACHE: Dict[str, Any] = {}

HOME_CACHE_TIME: Dict[str, float] = {}

HOME_CACHE_TTL = 300


# ==========================================================
# PYDANTIC MODELS
# ==========================================================

class TMDBMovieCard(BaseModel):

    tmdb_id: int

    title: str

    poster_url: Optional[str] = None

    release_date: Optional[str] = None

    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):

    tmdb_id: int

    title: str

    overview: Optional[str] = None

    release_date: Optional[str] = None

    poster_url: Optional[str] = None

    backdrop_url: Optional[str] = None

    genres: List[dict] = []


class TFIDFRecItem(BaseModel):

    title: str

    score: float

    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):

    query: str

    movie_details: TMDBMovieDetails

    tfidf_recommendations: List[
        TFIDFRecItem
    ]

    genre_recommendations: List[
        TMDBMovieCard
    ]


# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def normalize_title(title: str) -> str:

    return str(title).strip().lower()


def make_img_url(
    path: Optional[str]
) -> Optional[str]:

    if not path:
        return None

    if path.startswith("http"):
        return path

    return f"{TMDB_IMG_500}{path}"


def make_backdrop_url(
    path: Optional[str]
) -> Optional[str]:

    if not path:
        return None

    if path.startswith("http"):
        return path

    return f"{TMDB_IMG_ORIGINAL}{path}"


# ==========================================================
# TMDB HTTP CLIENT
# ==========================================================

async def tmdb_get(
    path: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:

    if not TMDB_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="TMDB_API_KEY is not configured on the backend.",
        )

    query = dict(params)

    query["api_key"] = TMDB_API_KEY

    url = f"{TMDB_BASE}{path}"

    timeout = httpx.Timeout(
        timeout=TMDB_TIMEOUT,
        connect=5.0,
    )

    try:

        async with httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
            ),
        ) as client:

            response = await client.get(
                url,
                params=query,
            )

    except httpx.TimeoutException:

        raise HTTPException(
            status_code=504,
            detail=(
                "TMDB request timed out. "
                "Please try again."
            ),
        )

    except httpx.RequestError as error:

        raise HTTPException(
            status_code=502,
            detail=(
                f"TMDB connection error: "
                f"{type(error).__name__}"
            ),
        )

    if response.status_code != 200:

        raise HTTPException(
            status_code=502,
            detail=(
                f"TMDB returned HTTP "
                f"{response.status_code}: "
                f"{response.text[:500]}"
            ),
        )

    try:

        return response.json()

    except Exception:

        raise HTTPException(
            status_code=502,
            detail="TMDB returned invalid JSON.",
        )


# ==========================================================
# TMDB RESULTS → MOVIE CARDS
# ==========================================================

def tmdb_cards_from_results(
    results: List[dict],
    limit: int = 20,
) -> List[TMDBMovieCard]:

    cards = []

    for movie in (results or [])[:limit]:

        movie_id = movie.get("id")

        if not movie_id:
            continue

        title = (
            movie.get("title")
            or movie.get("name")
            or "Untitled"
        )

        cards.append(
            TMDBMovieCard(
                tmdb_id=int(movie_id),

                title=title,

                poster_url=make_img_url(
                    movie.get("poster_path")
                ),

                release_date=movie.get(
                    "release_date"
                ),

                vote_average=movie.get(
                    "vote_average"
                ),
            )
        )

    return cards


# ==========================================================
# TMDB MOVIE DETAILS
# ==========================================================

async def tmdb_movie_details(
    movie_id: int,
) -> TMDBMovieDetails:

    data = await tmdb_get(
        f"/movie/{movie_id}",
        {
            "language": "en-US"
        },
    )

    return TMDBMovieDetails(

        tmdb_id=int(
            data["id"]
        ),

        title=(
            data.get("title")
            or "Unknown"
        ),

        overview=data.get(
            "overview"
        ),

        release_date=data.get(
            "release_date"
        ),

        poster_url=make_img_url(
            data.get("poster_path")
        ),

        backdrop_url=make_backdrop_url(
            data.get("backdrop_path")
        ),

        genres=data.get(
            "genres",
            [],
        ) or [],
    )


# ==========================================================
# TMDB SEARCH
# ==========================================================

async def tmdb_search_movies(
    query: str,
    page: int = 1,
) -> Dict[str, Any]:

    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(
    query: str,
) -> Optional[dict]:

    data = await tmdb_search_movies(
        query=query,
        page=1,
    )

    results = data.get(
        "results",
        [],
    )

    if not results:
        return None

    return results[0]


# ==========================================================
# TF-IDF INDEX
# ==========================================================

def build_title_to_idx_map(
    indices: Any,
) -> Dict[str, int]:

    title_to_idx = {}

    if isinstance(
        indices,
        dict,
    ):

        for key, value in indices.items():

            title_to_idx[
                normalize_title(key)
            ] = int(value)

        return title_to_idx

    try:

        for key, value in indices.items():

            title_to_idx[
                normalize_title(key)
            ] = int(value)

        return title_to_idx

    except Exception:

        raise RuntimeError(
            "indices.pkl must be a "
            "dictionary or pandas Series."
        )


# ==========================================================
# GET LOCAL DATASET INDEX
# ==========================================================

def get_local_idx_by_title(
    title: str,
) -> int:

    if TITLE_TO_IDX is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "TF-IDF index "
                "is not initialized."
            ),
        )

    key = normalize_title(title)

    if key in TITLE_TO_IDX:

        return int(
            TITLE_TO_IDX[key]
        )

    raise HTTPException(
        status_code=404,
        detail=(
            f"Movie '{title}' "
            "not found in local dataset."
        ),
    )


# ==========================================================
# TF-IDF RECOMMENDATION
# ==========================================================

def tfidf_recommend_titles(
    query_title: str,
    top_n: int = 10,
) -> List[Tuple[str, float]]:

    global df
    global tfidf_matrix

    if df is None:

        raise HTTPException(
            status_code=500,
            detail="DataFrame not loaded.",
        )

    if tfidf_matrix is None:

        raise HTTPException(
            status_code=500,
            detail="TF-IDF matrix not loaded.",
        )

    index = get_local_idx_by_title(
        query_title
    )

    query_vector = (
        tfidf_matrix[index]
    )

    scores = (
        tfidf_matrix @ query_vector.T
    ).toarray().ravel()

    order = np.argsort(
        -scores
    )

    recommendations = []

    for item_index in order:

        item_index = int(
            item_index
        )

        if item_index == index:
            continue

        try:

            title = str(
                df.iloc[
                    item_index
                ]["title"]
            )

        except Exception:

            continue

        recommendations.append(
            (
                title,
                float(
                    scores[item_index]
                ),
            )
        )

        if len(
            recommendations
        ) >= top_n:

            break

    return recommendations


# ==========================================================
# LOCAL TITLE → TMDB CARD
# ==========================================================

async def attach_tmdb_card_by_title(
    title: str,
) -> Optional[TMDBMovieCard]:

    try:

        movie = await tmdb_search_first(
            title
        )

        if not movie:
            return None

        return TMDBMovieCard(

            tmdb_id=int(
                movie["id"]
            ),

            title=(
                movie.get("title")
                or title
            ),

            poster_url=make_img_url(
                movie.get(
                    "poster_path"
                )
            ),

            release_date=movie.get(
                "release_date"
            ),

            vote_average=movie.get(
                "vote_average"
            ),
        )

    except Exception:

        return None


# ==========================================================
# LOAD PICKLE FILES
# ==========================================================

def build_dataset_from_csv() -> pd.DataFrame:

    if not os.path.exists(CSV_PATH):
        raise RuntimeError(f"Dataset not found: {CSV_PATH}")

    source = pd.read_csv(CSV_PATH)
    source = source.drop_duplicates().reset_index(drop=True)
    source = source[
        ["title", "overview", "genres", "tagline", "vote_average", "popularity"]
    ]
    source = source.dropna(subset=["title"]).copy()
    source["overview"] = source["overview"].fillna("")
    source["tagline"] = source["tagline"].fillna("")

    def parse_genres(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return ""
        try:
            return " ".join(
                item["name"]
                for item in __import__("ast").literal_eval(value)
                if isinstance(item, dict) and item.get("name")
            )
        except (ValueError, SyntaxError, TypeError):
            return ""

    source["genres"] = source["genres"].map(parse_genres)
    source["tags"] = (
        source["overview"] + " " + source["genres"] + " " + source["tagline"]
    )
    return source


def load_pickle(path: str, label: str) -> Any:
    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, ValueError) as error:
        print(f"{label} unavailable ({error}); using fallback when supported.")
        return None

@app.on_event("startup")
def load_pickles():

    global df
    global indices_obj
    global tfidf_matrix
    global tfidf_obj
    global TITLE_TO_IDX

    print("====================================")
    print("Starting Movie Recommender API...")
    print("====================================")

    # ------------------------------------------------------
    # df.pkl
    # ------------------------------------------------------

    df = load_pickle(DF_PATH, "df.pkl")
    if not isinstance(df, pd.DataFrame) or "title" not in df.columns:
        df = build_dataset_from_csv()
    print(f"Dataset loaded: {len(df)} rows")

    # ------------------------------------------------------
    # indices.pkl
    # ------------------------------------------------------

    indices_obj = load_pickle(INDICES_PATH, "indices.pkl")
    if indices_obj is None:
        indices_obj = pd.Series(df.index, index=df["title"]).drop_duplicates()

    # ------------------------------------------------------
    # tfidf_matrix.pkl
    # ------------------------------------------------------

    tfidf_matrix = load_pickle(TFIDF_MATRIX_PATH, "tfidf_matrix.pkl")
    if tfidf_matrix is None:
        raise RuntimeError("tfidf_matrix.pkl is missing or invalid.")

    # ------------------------------------------------------
    # tfidf.pkl
    # ------------------------------------------------------

    tfidf_obj = load_pickle(TFIDF_PATH, "tfidf.pkl")

    # ------------------------------------------------------
    # BUILD INDEX
    # ------------------------------------------------------

    try:

        TITLE_TO_IDX = (
            build_title_to_idx_map(
                indices_obj
            )
        )

        print(
            "TITLE_TO_IDX created: "
            f"{len(TITLE_TO_IDX)} titles"
        )

    except Exception as error:

        raise RuntimeError(
            f"Failed to build title index: {error}"
        )

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    if len(df) != tfidf_matrix.shape[0]:
        raise RuntimeError(
            f"Dataset rows ({len(df)}) do not match TF-IDF rows ({tfidf_matrix.shape[0]})."
        )

    print("====================================")
    print("Movie Recommender API READY")
    print("====================================")


# ==========================================================
# ROOT
# ==========================================================

@app.get("/")
def root():

    return {
        "message": "Movie Recommender API is running",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "tmdb_configured": bool(
            TMDB_API_KEY
        ),
        "dataset_loaded": df is not None,
        "tfidf_loaded": (
            tfidf_matrix is not None
        ),
    }


# ==========================================================
# HOME FEED
# ==========================================================

@app.get(
    "/home",
    response_model=List[TMDBMovieCard],
)
async def home(

    category: str = Query(
        "popular"
    ),

    limit: int = Query(
        24,
        ge=1,
        le=50,
    ),
):

    valid_categories = {
        "trending",
        "popular",
        "top_rated",
        "upcoming",
        "now_playing",
    }

    if category not in valid_categories:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid category. "
                f"Use one of: "
                f"{', '.join(valid_categories)}"
            ),
        )

    # ------------------------------------------------------
    # CACHE
    # ------------------------------------------------------

    import time

    current_time = time.time()

    if category in HOME_CACHE:

        cache_time = HOME_CACHE_TIME.get(
            category,
            0,
        )

        if (
            current_time - cache_time
            < HOME_CACHE_TTL
        ):

            cached = HOME_CACHE[
                category
            ]

            return cached[:limit]

    # ------------------------------------------------------
    # TMDB REQUEST
    # ------------------------------------------------------

    try:

        if category == "trending":

            data = await tmdb_get(
                "/trending/movie/day",
                {
                    "language": "en-US"
                },
            )

        else:

            data = await tmdb_get(
                f"/movie/{category}",
                {
                    "language": "en-US",
                    "page": 1,
                },
            )

        cards = tmdb_cards_from_results(
            data.get(
                "results",
                [],
            ),
            limit=50,
        )

        # --------------------------------------------------
        # SAVE CACHE
        # --------------------------------------------------

        HOME_CACHE[
            category
        ] = cards

        HOME_CACHE_TIME[
            category
        ] = current_time

        return cards[:limit]

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Home route failed: "
                f"{error}"
            ),
        )


# ==========================================================
# TMDB SEARCH
# ==========================================================

@app.get("/tmdb/search")
async def tmdb_search(

    query: str = Query(
        ...,
        min_length=1,
    ),

    page: int = Query(
        1,
        ge=1,
        le=10,
    ),
):

    query = query.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    return await tmdb_search_movies(
        query=query,
        page=page,
    )


# ==========================================================
# MOVIE DETAILS
# ==========================================================

@app.get(
    "/movie/id/{tmdb_id}",
    response_model=TMDBMovieDetails,
)
async def movie_details_route(
    tmdb_id: int,
):

    if tmdb_id <= 0:

        raise HTTPException(
            status_code=400,
            detail="Invalid TMDB movie ID.",
        )

    return await tmdb_movie_details(
        tmdb_id
    )


# ==========================================================
# GENRE RECOMMENDATIONS
# ==========================================================

@app.get(
    "/recommend/genre",
    response_model=List[TMDBMovieCard],
)
async def recommend_genre(

    tmdb_id: int = Query(
        ...,
        gt=0,
    ),

    limit: int = Query(
        18,
        ge=1,
        le=50,
    ),
):

    details = await tmdb_movie_details(
        tmdb_id
    )

    if not details.genres:

        return []

    genre_id = details.genres[0].get(
        "id"
    )

    if not genre_id:

        return []

    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )

    cards = tmdb_cards_from_results(
        discover.get(
            "results",
            [],
        ),
        limit=limit + 1,
    )

    return [
        card
        for card in cards
        if card.tmdb_id != tmdb_id
    ][:limit]


# ==========================================================
# TF-IDF ONLY
# ==========================================================

@app.get("/recommend/tfidf")
async def recommend_tfidf(

    title: str = Query(
        ...,
        min_length=1,
    ),

    top_n: int = Query(
        10,
        ge=1,
        le=50,
    ),
):

    title = title.strip()

    if not title:

        raise HTTPException(
            status_code=400,
            detail="Title cannot be empty.",
        )

    recommendations = (
        tfidf_recommend_titles(
            title,
            top_n=top_n,
        )
    )

    return [
        {
            "title": movie_title,
            "score": score,
        }
        for movie_title, score
        in recommendations
    ]


# ==========================================================
# SEARCH BUNDLE
# ==========================================================

@app.get(
    "/movie/search",
    response_model=SearchBundleResponse,
)
async def search_bundle(

    query: str = Query(
        ...,
        min_length=1,
    ),

    tfidf_top_n: int = Query(
        12,
        ge=1,
        le=30,
    ),

    genre_limit: int = Query(
        12,
        ge=1,
        le=30,
    ),
):

    query = query.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    # ------------------------------------------------------
    # FIND MOVIE
    # ------------------------------------------------------

    best = await tmdb_search_first(
        query
    )

    if not best:

        raise HTTPException(
            status_code=404,
            detail=(
                f"No movie found for "
                f"'{query}'"
            ),
        )

    tmdb_id = int(
        best["id"]
    )

    # ------------------------------------------------------
    # MOVIE DETAILS
    # ------------------------------------------------------

    details = await tmdb_movie_details(
        tmdb_id
    )

    # ------------------------------------------------------
    # TF-IDF
    # ------------------------------------------------------

    tfidf_items = []

    recommendations = []

    try:

        recommendations = (
            tfidf_recommend_titles(
                details.title,
                top_n=tfidf_top_n,
            )
        )

    except Exception:

        try:

            recommendations = (
                tfidf_recommend_titles(
                    query,
                    top_n=tfidf_top_n,
                )
            )

        except Exception:

            recommendations = []

    # ------------------------------------------------------
    # FETCH TMDB CARDS IN PARALLEL
    # ------------------------------------------------------

    if recommendations:

        tasks = [
            attach_tmdb_card_by_title(
                title
            )
            for title, score
            in recommendations
        ]

        cards = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        for (
            recommendation,
            card,
        ) in zip(
            recommendations,
            cards,
        ):

            title, score = recommendation

            if isinstance(
                card,
                TMDBMovieCard,
            ):

                tfidf_items.append(
                    TFIDFRecItem(
                        title=title,
                        score=score,
                        tmdb=card,
                    )
                )

            else:

                tfidf_items.append(
                    TFIDFRecItem(
                        title=title,
                        score=score,
                        tmdb=None,
                    )
                )

    # ------------------------------------------------------
    # GENRE RECOMMENDATIONS
    # ------------------------------------------------------

    genre_recommendations = []

    if details.genres:

        genre_id = details.genres[0].get(
            "id"
        )

        if genre_id:

            discover = await tmdb_get(
                "/discover/movie",
                {
                    "with_genres": genre_id,
                    "language": "en-US",
                    "sort_by": "popularity.desc",
                    "page": 1,
                },
            )

            cards = tmdb_cards_from_results(
                discover.get(
                    "results",
                    [],
                ),
                limit=genre_limit + 1,
            )

            genre_recommendations = [
                card
                for card in cards
                if card.tmdb_id
                != details.tmdb_id
            ][:genre_limit]

    # ------------------------------------------------------
    # RESPONSE
    # ------------------------------------------------------

    return SearchBundleResponse(

        query=query,

        movie_details=details,

        tfidf_recommendations=(
            tfidf_items
        ),

        genre_recommendations=(
            genre_recommendations
        ),
    )