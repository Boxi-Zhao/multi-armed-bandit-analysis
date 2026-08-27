from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from data_processing import prepare_reward_data


# Experiment configuration
MOVIE_ARM_SETTINGS = [7, 70]
NUM_ROUNDS = 400_000
NUM_EXPERIMENTS = 100
EXPLORATION_SCALE = 4.0
SWITCH_ROUNDS = [100_000, 200_000, 300_000]
RANDOM_SEED = 42


def select_random_movies(movie_rewards, num_movies, rng):
    """Randomly select movie-ID arms without replacement."""
    movie_ids = np.array(list(movie_rewards.keys()))
    return rng.choice(movie_ids, size=num_movies, replace=False).tolist()


def calculate_optimal_reward(movie_rewards, selected_movies):
    """
    Return the highest expected reward among the selected movie arms.
    """
    mean_rewards = [
        np.mean(movie_rewards[movie_id])
        for movie_id in selected_movies
        if movie_rewards[movie_id]
    ]
    return max(mean_rewards) if mean_rewards else 0.0


def run_ucb(
    movie_rewards,
    initial_movies,
    num_rounds,
    exploration_scale,
    rng,
    switch_rounds=None,
):
    """
    Run one UCB simulation with movie IDs as arms.
    """
    selected_movies = list(initial_movies)
    num_arms = len(selected_movies)

    total_rewards = np.zeros(num_arms)
    total_selections = np.zeros(num_arms)
    cumulative_regret = np.zeros(num_rounds)

    current_segment_start = 0
    reward_samples = {
        movie_id: rng.choice(
            movie_rewards[movie_id],
            size=num_rounds,
            replace=True,
        )
        for movie_id in selected_movies
    }

    optimal_reward = calculate_optimal_reward(
        movie_rewards,
        selected_movies,
    )
    total_regret = 0.0

    for t in range(num_rounds):
        if switch_rounds and t in switch_rounds:
            selected_movies = select_random_movies(
                movie_rewards,
                num_arms,
                rng,
            )

            total_rewards = np.zeros(num_arms)
            total_selections = np.zeros(num_arms)
            current_segment_start = t

            remaining_rounds = num_rounds - t
            reward_samples = {
                movie_id: rng.choice(
                    movie_rewards[movie_id],
                    size=remaining_rounds,
                    replace=True,
                )
                for movie_id in selected_movies
            }

            optimal_reward = calculate_optimal_reward(
                movie_rewards,
                selected_movies,
            )

        segment_t = t - current_segment_start

        if segment_t < num_arms:
            arm_index = segment_t
        else:
            averages = total_rewards / np.maximum(total_selections, 1)
            confidence_bounds = (
                exploration_scale
                * np.sqrt(
                    2 * np.log(segment_t + 1)
                    / np.maximum(total_selections, 1)
                )
            )
            ucb_values = averages + confidence_bounds
            arm_index = int(np.argmax(ucb_values))

        selected_movie = selected_movies[arm_index]
        reward = reward_samples[selected_movie][segment_t]

        total_rewards[arm_index] += reward
        total_selections[arm_index] += 1

        total_regret += optimal_reward - reward
        cumulative_regret[t] = total_regret

    return cumulative_regret


def run_thompson_sampling(
    movie_rewards,
    initial_movies,
    num_rounds,
    exploration_scale,
    rng,
    switch_rounds=None,
):
    """
    Run one Thompson Sampling simulation with movie IDs as arms.

    This follows the normal-distribution sampling approach used
    in the original project.
    """
    selected_movies = list(initial_movies)
    num_arms = len(selected_movies)

    means = {movie_id: 0.0 for movie_id in selected_movies}
    counts = {movie_id: 1 for movie_id in selected_movies}
    cumulative_regret = np.zeros(num_rounds)

    current_segment_start = 0
    reward_samples = {
        movie_id: rng.choice(
            movie_rewards[movie_id],
            size=num_rounds,
            replace=True,
        )
        for movie_id in selected_movies
    }

    optimal_reward = calculate_optimal_reward(
        movie_rewards,
        selected_movies,
    )
    total_regret = 0.0

    for t in range(num_rounds):
        if switch_rounds and t in switch_rounds:
            selected_movies = select_random_movies(
                movie_rewards,
                num_arms,
                rng,
            )

            means = {movie_id: 0.0 for movie_id in selected_movies}
            counts = {movie_id: 1 for movie_id in selected_movies}
            current_segment_start = t

            remaining_rounds = num_rounds - t
            reward_samples = {
                movie_id: rng.choice(
                    movie_rewards[movie_id],
                    size=remaining_rounds,
                    replace=True,
                )
                for movie_id in selected_movies
            }

            optimal_reward = calculate_optimal_reward(
                movie_rewards,
                selected_movies,
            )

        segment_t = t - current_segment_start

        sampled_rewards = {
            movie_id: rng.normal(
                means[movie_id],
                np.sqrt(exploration_scale**2 / counts[movie_id]),
            )
            for movie_id in selected_movies
        }

        selected_movie = max(
            sampled_rewards,
            key=sampled_rewards.get,
        )
        reward = reward_samples[selected_movie][segment_t]

        current_count = counts[selected_movie]
        current_mean = means[selected_movie]

        means[selected_movie] = (
            current_mean * current_count + reward
        ) / (current_count + 1)
        counts[selected_movie] += 1

        total_regret += optimal_reward - reward
        cumulative_regret[t] = total_regret

    return cumulative_regret


def run_repeated_experiments(
    algorithm,
    movie_rewards,
    num_movies,
    num_rounds,
    exploration_scale,
    num_experiments,
    rng,
    switch_rounds=None,
):
    """
    Run repeated experiments and return the mean cumulative regret
    and one-standard-deviation values.
    """
    all_regrets = np.zeros((num_experiments, num_rounds))

    for experiment in range(num_experiments):
        print(
            f"{num_movies} movie arms | "
            f"{algorithm.__name__}: "
            f"experiment {experiment + 1}/{num_experiments}"
        )

        initial_movies = select_random_movies(
            movie_rewards,
            num_movies,
            rng,
        )

        all_regrets[experiment] = algorithm(
            movie_rewards=movie_rewards,
            initial_movies=initial_movies,
            num_rounds=num_rounds,
            exploration_scale=exploration_scale,
            rng=rng,
            switch_rounds=switch_rounds,
        )

    return all_regrets.mean(axis=0), all_regrets.std(axis=0)


def run_movie_setting(movie_rewards, num_movies, rng):
    """
    Run static and dynamic UCB/TS experiments for one movie-arm setting.
    """
    static_ucb_mean, static_ucb_std = run_repeated_experiments(
        algorithm=run_ucb,
        movie_rewards=movie_rewards,
        num_movies=num_movies,
        num_rounds=NUM_ROUNDS,
        exploration_scale=EXPLORATION_SCALE,
        num_experiments=NUM_EXPERIMENTS,
        rng=rng,
    )

    static_ts_mean, static_ts_std = run_repeated_experiments(
        algorithm=run_thompson_sampling,
        movie_rewards=movie_rewards,
        num_movies=num_movies,
        num_rounds=NUM_ROUNDS,
        exploration_scale=EXPLORATION_SCALE,
        num_experiments=NUM_EXPERIMENTS,
        rng=rng,
    )

    dynamic_ucb_mean, dynamic_ucb_std = run_repeated_experiments(
        algorithm=run_ucb,
        movie_rewards=movie_rewards,
        num_movies=num_movies,
        num_rounds=NUM_ROUNDS,
        exploration_scale=EXPLORATION_SCALE,
        num_experiments=NUM_EXPERIMENTS,
        rng=rng,
        switch_rounds=SWITCH_ROUNDS,
    )

    dynamic_ts_mean, dynamic_ts_std = run_repeated_experiments(
        algorithm=run_thompson_sampling,
        movie_rewards=movie_rewards,
        num_movies=num_movies,
        num_rounds=NUM_ROUNDS,
        exploration_scale=EXPLORATION_SCALE,
        num_experiments=NUM_EXPERIMENTS,
        rng=rng,
        switch_rounds=SWITCH_ROUNDS,
    )

    return [
        ("UCB - Static", static_ucb_mean, static_ucb_std),
        ("TS - Static", static_ts_mean, static_ts_std),
        ("UCB - Dynamic", dynamic_ucb_mean, dynamic_ucb_std),
        ("TS - Dynamic", dynamic_ts_mean, dynamic_ts_std),
    ]


def plot_results(results, num_movies, output_path):
    """Plot and save results for one movie-arm setting."""
    rounds = np.arange(NUM_ROUNDS)

    plt.figure(figsize=(12, 8))

    for label, mean_regret, std_regret in results:
        plt.plot(rounds, mean_regret, label=label)
        plt.fill_between(
            rounds,
            mean_regret - std_regret,
            mean_regret + std_regret,
            alpha=0.2,
        )

    plt.title(
        f"UCB vs. Thompson Sampling with {num_movies} Movie-ID Arms"
    )
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()


def main():
    _, movie_rewards = prepare_reward_data()

    rng = np.random.default_rng(RANDOM_SEED)

    results_dir = Path(__file__).resolve().parents[1] / "results"

    for num_movies in MOVIE_ARM_SETTINGS:
        results = run_movie_setting(
            movie_rewards=movie_rewards,
            num_movies=num_movies,
            rng=rng,
        )

        output_path = (
            results_dir
            / f"movie_{num_movies}_comparison.png"
        )

        plot_results(
            results=results,
            num_movies=num_movies,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
