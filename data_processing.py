from pathlib import Path
from collections import defaultdict
import pandas as pd


def load_movielens(data_dir=None):
    """
    Load the MovieLens 1M ratings and movie metadata files.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory containing ratings.dat and movies.dat.
        If omitted, the function uses the repository's data/ folder.

    Returns
    -------
    ratings : pandas.DataFrame
    movies : pandas.DataFrame
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data"
    else:
        data_dir = Path(data_dir)

    ratings_path = data_dir / "ratings.dat"
    movies_path = data_dir / "movies.dat"

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["userID", "movieID", "rating", "timestamp"],
        encoding="ISO-8859-1",
    )

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movieID", "title", "genres"],
        encoding="ISO-8859-1",
    )

    return ratings, movies


def build_genre_rewards(ratings, movies):
    """
    Build reward distributions using movie genres as bandit arms.

    Each rating is added to every genre associated with that movie.
    """
    merged_df = pd.merge(
        ratings,
        movies[["movieID", "genres"]],
        on="movieID",
        how="inner",
    )

    genre_rewards = defaultdict(list)

    for row in merged_df.itertuples(index=False):
        for genre in row.genres.split("|"):
            genre_rewards[genre].append(row.rating)

    return dict(genre_rewards)


def build_movie_rewards(ratings):
    """
    Build reward distributions using movie IDs as bandit arms.
    """
    movie_rewards = defaultdict(list)

    for row in ratings.itertuples(index=False):
        movie_rewards[row.movieID].append(row.rating)

    return dict(movie_rewards)


def prepare_reward_data(data_dir=None):
    """
    Load MovieLens 1M and create both genre-level and movie-level
    reward distributions.
    """
    ratings, movies = load_movielens(data_dir)

    genre_rewards = build_genre_rewards(ratings, movies)
    movie_rewards = build_movie_rewards(ratings)

    return genre_rewards, movie_rewards


if __name__ == "__main__":
    genre_rewards, movie_rewards = prepare_reward_data()

    print(f"Number of genre arms: {len(genre_rewards)}")
    print(f"Number of movie arms: {len(movie_rewards)}")
