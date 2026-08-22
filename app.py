import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# CONFIG
# ==========================================================

API_BASE = os.getenv(
    "API_BASE",
    "http://127.0.0.1:8000",
).rstrip("/")

TMDB_IMG = "https://image.tmdb.org/t/p/w500"

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


# ==========================================================
# STYLES
# ==========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.92rem;
    }

    .movie-title {
        font-size: 0.9rem;
        line-height: 1.15rem;
        min-height: 2.3rem;
        overflow: hidden;
        margin-top: 6px;
    }

    .card {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 16px;
        padding: 14px;
        background: rgba(255,255,255,0.7);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "view" not in st.session_state:
    st.session_state.view = "home"

if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None


# ==========================================================
# QUERY PARAMETERS
# ==========================================================

qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")

if qp_view in ("home", "details"):
    st.session_state.view = qp_view

if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except (ValueError, TypeError):
        st.session_state.selected_tmdb_id = None


# ==========================================================
# NAVIGATION
# ==========================================================

def goto_home():
    st.session_state.view = "home"
    st.session_state.selected_tmdb_id = None

    st.query_params.clear()
    st.query_params["view"] = "home"

    st.rerun()


def goto_details(tmdb_id):
    try:
        tmdb_id = int(tmdb_id)
    except (ValueError, TypeError):
        st.error("Invalid movie ID.")
        return

    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = tmdb_id

    st.query_params["view"] = "details"
    st.query_params["id"] = str(tmdb_id)

    st.rerun()


# ==========================================================
# API HELPER
# ==========================================================

@st.cache_data(ttl=30)
def api_get_json(path, params=None):

    url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"

    try:
        response = requests.get(
            url,
            params=params,
            timeout=60,
        )

        if response.status_code >= 400:
            return None, (
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        try:
            return response.json(), None

        except ValueError:
            return None, "Backend returned invalid JSON."

    except requests.exceptions.Timeout:
        return None, "Backend request timed out."

    except requests.exceptions.ConnectionError:
        return None, (
            f"Cannot connect to backend: {API_BASE}"
        )

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {e}"

    except Exception as e:
        return None, f"Unexpected error: {e}"


# ==========================================================
# POSTER GRID
# ==========================================================

def poster_grid(cards, cols=6, key_prefix="grid"):

    if not cards:
        st.info("No movies to show.")
        return

    rows = (len(cards) + cols - 1) // cols

    idx = 0

    for row in range(rows):

        colset = st.columns(cols)

        for col in range(cols):

            if idx >= len(cards):
                break

            movie = cards[idx]
            current_idx = idx
            idx += 1

            tmdb_id = movie.get("tmdb_id") or movie.get("id")
            title = movie.get("title") or "Untitled"
            poster = movie.get("poster_url")

            with colset[col]:

                if poster:
                    try:
                        st.image(
                            poster,
                            width="stretch",
                        )
                    except Exception:
                        st.write("🖼️ Poster unavailable")
                else:
                    st.write("🖼️ No poster")

                if st.button(
                    "Open",
                    key=f"{key_prefix}_{row}_{col}_{current_idx}_{tmdb_id}",
                    use_container_width=True,
                ):
                    if tmdb_id:
                        goto_details(tmdb_id)

                st.markdown(
                    f"""
                    <div class="movie-title">
                        {title}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ==========================================================
# TF-IDF → CARDS
# ==========================================================

def to_cards_from_tfidf_items(tfidf_items):

    cards = []

    for item in tfidf_items or []:

        tmdb = item.get("tmdb") or {}

        tmdb_id = tmdb.get("tmdb_id") or tmdb.get("id")

        if not tmdb_id:
            continue

        cards.append(
            {
                "tmdb_id": tmdb_id,
                "title": (
                    tmdb.get("title")
                    or item.get("title")
                    or "Untitled"
                ),
                "poster_url": tmdb.get("poster_url"),
            }
        )

    return cards


# ==========================================================
# TMDB SEARCH PARSER
# ==========================================================

def parse_tmdb_search_to_cards(
    data,
    keyword,
    limit=24,
):

    keyword_l = keyword.strip().lower()

    raw_items = []

    # ------------------------------------------------------
    # API RESPONSE:
    # {"results": [...]}
    # ------------------------------------------------------

    if isinstance(data, dict) and "results" in data:

        raw = data.get("results") or []

        for movie in raw:

            title = (
                movie.get("title")
                or movie.get("name")
                or ""
            ).strip()

            tmdb_id = movie.get("id")

            poster_path = movie.get("poster_path")

            if not title or not tmdb_id:
                continue

            poster_url = None

            if poster_path:

                if poster_path.startswith("http"):
                    poster_url = poster_path

                else:
                    poster_url = f"{TMDB_IMG}{poster_path}"

            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": poster_url,
                    "release_date": movie.get(
                        "release_date",
                        "",
                    ),
                }
            )

    # ------------------------------------------------------
    # API RESPONSE:
    # [{tmdb_id, title, poster_url}]
    # ------------------------------------------------------

    elif isinstance(data, list):

        for movie in data:

            if not isinstance(movie, dict):
                continue

            tmdb_id = (
                movie.get("tmdb_id")
                or movie.get("id")
            )

            title = (
                movie.get("title")
                or movie.get("name")
                or ""
            ).strip()

            poster_url = movie.get("poster_url")

            if not title or not tmdb_id:
                continue

            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": poster_url,
                    "release_date": movie.get(
                        "release_date",
                        "",
                    ),
                }
            )

    else:
        return [], []

    # ------------------------------------------------------
    # MATCHING
    # ------------------------------------------------------

    matched = [
        movie
        for movie in raw_items
        if keyword_l in movie["title"].lower()
    ]

    final_list = matched if matched else raw_items

    # ------------------------------------------------------
    # SUGGESTIONS
    # ------------------------------------------------------

    suggestions = []

    for movie in final_list[:10]:

        year = (
            movie.get("release_date") or ""
        )[:4]

        if year:
            label = f"{movie['title']} ({year})"
        else:
            label = movie["title"]

        suggestions.append(
            (
                label,
                movie["tmdb_id"],
            )
        )

    # ------------------------------------------------------
    # CARDS
    # ------------------------------------------------------

    cards = [
        {
            "tmdb_id": movie["tmdb_id"],
            "title": movie["title"],
            "poster_url": movie["poster_url"],
        }
        for movie in final_list[:limit]
    ]

    return suggestions, cards


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("## 🎬 Menu")

    if st.button(
        "🏠 Home",
        use_container_width=True,
    ):
        goto_home()

    st.markdown("---")

    st.markdown("### 🏠 Home Feed")

    home_category = st.selectbox(
        "Category",
        [
            "trending",
            "popular",
            "top_rated",
            "now_playing",
            "upcoming",
        ],
        index=0,
    )

    grid_cols = st.slider(
        "Grid columns",
        min_value=4,
        max_value=8,
        value=6,
    )


# ==========================================================
# HEADER
# ==========================================================

st.title("🎬 Movie Recommender")

st.markdown(
    """
    <div class="small-muted">
        Type keyword → suggestions → matching movies →
        open movie → details + recommendations
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ==========================================================
# HOME
# ==========================================================

if st.session_state.view == "home":

    typed = st.text_input(
        "Search by movie title",
        placeholder="Type: avenger, batman, love...",
    )

    st.divider()

    # ======================================================
    # SEARCH MODE
    # ======================================================

    if typed.strip():

        if len(typed.strip()) < 2:

            st.caption(
                "Type at least 2 characters for suggestions."
            )

        else:

            data, err = api_get_json(
                "/tmdb/search",
                params={
                    "query": typed.strip()
                },
            )

            if err or data is None:

                st.error(
                    f"Search failed: {err}"
                )

            else:

                suggestions, cards = (
                    parse_tmdb_search_to_cards(
                        data,
                        typed.strip(),
                        limit=24,
                    )
                )

                # --------------------------------------------------
                # SUGGESTIONS
                # --------------------------------------------------

                if suggestions:

                    labels = [
                        "-- Select a movie --"
                    ]

                    labels.extend(
                        [item[0] for item in suggestions]
                    )

                    selected = st.selectbox(
                        "Suggestions",
                        labels,
                        index=0,
                    )

                    if (
                        selected
                        != "-- Select a movie --"
                    ):

                        label_to_id = {
                            label: movie_id
                            for label, movie_id
                            in suggestions
                        }

                        selected_id = (
                            label_to_id.get(selected)
                        )

                        if selected_id:
                            goto_details(selected_id)

                else:

                    st.info(
                        "No suggestions found. "
                        "Try another keyword."
                    )

                # --------------------------------------------------
                # SEARCH RESULTS
                # --------------------------------------------------

                st.markdown("### 🔎 Results")

                poster_grid(
                    cards,
                    cols=grid_cols,
                    key_prefix="search_results",
                )

        st.stop()

    # ======================================================
    # HOME FEED
    # ======================================================

    st.markdown(
        f"### 🏠 Home — "
        f"{home_category.replace('_', ' ').title()}"
    )

    home_cards, err = api_get_json(
        "/home",
        params={
            "category": home_category,
            "limit": 24,
        },
    )

    if err:

        st.error(
            f"Home feed failed: {err}"
        )

        st.info(
            "Make sure your FastAPI backend is "
            "running on Render."
        )

        st.stop()

    if not home_cards:

        st.warning(
            "Backend returned no movies."
        )

        st.stop()

    poster_grid(
        home_cards,
        cols=grid_cols,
        key_prefix="home_feed",
    )


# ==========================================================
# DETAILS
# ==========================================================

elif st.session_state.view == "details":

    tmdb_id = st.session_state.selected_tmdb_id

    if not tmdb_id:

        st.warning(
            "No movie selected."
        )

        if st.button("← Back to Home"):
            goto_home()

        st.stop()

    # ------------------------------------------------------
    # TOP BAR
    # ------------------------------------------------------

    left_top, right_top = st.columns(
        [3, 1]
    )

    with left_top:

        st.markdown(
            "### 📄 Movie Details"
        )

    with right_top:

        if st.button(
            "← Back to Home",
            use_container_width=True,
        ):
            goto_home()

    # ------------------------------------------------------
    # MOVIE DETAILS
    # ------------------------------------------------------

    data, err = api_get_json(
        f"/movie/id/{tmdb_id}"
    )

    if err or not data:

        st.error(
            f"Could not load details: "
            f"{err or 'Unknown error'}"
        )

        st.stop()

    # ------------------------------------------------------
    # DETAILS LAYOUT
    # ------------------------------------------------------

    left, right = st.columns(
        [1, 2.4],
        gap="large",
    )

    # ------------------------------------------------------
    # POSTER
    # ------------------------------------------------------

    with left:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True,
        )

        poster_url = data.get(
            "poster_url"
        )

        if poster_url:

            st.image(
                poster_url,
                width="stretch",
            )

        else:

            st.write(
                "🖼️ No poster available"
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------
    # INFORMATION
    # ------------------------------------------------------

    with right:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"## {data.get('title', 'Unknown')}"
        )

        release = (
            data.get("release_date")
            or "-"
        )

        genres_data = (
            data.get("genres")
            or []
        )

        genres = ", ".join(
            [
                g.get("name", "")
                for g in genres_data
                if isinstance(g, dict)
            ]
        )

        if not genres:
            genres = "-"

        st.markdown(
            f"""
            <div class="small-muted">
                Release: {release}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="small-muted">
                Genres: {genres}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown(
            "### Overview"
        )

        st.write(
            data.get("overview")
            or "No overview available."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------
    # BACKDROP
    # ------------------------------------------------------

    backdrop_url = data.get(
        "backdrop_url"
    )

    if backdrop_url:

        st.markdown(
            "#### 🖼️ Backdrop"
        )

        st.image(
            backdrop_url,
            width="stretch",
        )

    st.divider()

    # ======================================================
    # RECOMMENDATIONS
    # ======================================================

    st.markdown(
        "### ✅ Recommendations"
    )

    title = (
        data.get("title") or ""
    ).strip()

    if title:

        bundle, err2 = api_get_json(
            "/movie/search",
            params={
                "query": title,
                "tfidf_top_n": 12,
                "genre_limit": 12,
            },
        )

        if not err2 and bundle:

            # --------------------------------------------------
            # TF-IDF
            # --------------------------------------------------

            tfidf_items = (
                bundle.get(
                    "tfidf_recommendations"
                )
                or []
            )

            if tfidf_items:

                st.markdown(
                    "#### 🔎 Similar Movies (TF-IDF)"
                )

                poster_grid(
                    to_cards_from_tfidf_items(
                        tfidf_items
                    ),
                    cols=grid_cols,
                    key_prefix="details_tfidf",
                )

            # --------------------------------------------------
            # GENRE
            # --------------------------------------------------

            genre_items = (
                bundle.get(
                    "genre_recommendations"
                )
                or []
            )

            if genre_items:

                st.markdown(
                    "#### 🎭 More Like This (Genre)"
                )

                poster_grid(
                    genre_items,
                    cols=grid_cols,
                    key_prefix="details_genre",
                )

            if not tfidf_items and not genre_items:

                st.info(
                    "No recommendations available."
                )

        else:

            # --------------------------------------------------
            # FALLBACK GENRE API
            # --------------------------------------------------

            st.info(
                "Trying Genre recommendations..."
            )

            genre_only, err3 = api_get_json(
                "/recommend/genre",
                params={
                    "tmdb_id": tmdb_id,
                    "limit": 18,
                },
            )

            if not err3 and genre_only:

                poster_grid(
                    genre_only,
                    cols=grid_cols,
                    key_prefix="details_genre_fallback",
                )

            else:

                st.warning(
                    "No recommendations available right now."
                )

    else:

        st.warning(
            "No movie title available "
            "to compute recommendations."
        )