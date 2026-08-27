from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from data_processing import prepare_reward_data


# Experiment configuration
NUM_GENRES = 7
NUM_ROUNDS = 400_000
NUM_EXPERIMENTS = 100
EXPLORATION_SCALE = 4.0
SWITCH_ROUNDS = [100_000, 200_000, 300_000]
RANDOM_SEED = 42


def select_random_genres(genre_rewards, num_genres, rng):
    """Randomly select genre arms without replacement."""
    genres = np.array(list(genre_rewards.keys()))
    return rng.choice(genres, size=num_genres, replace=False).tolist()


def calculate_optimal_reward(genre_rewards, selected_genres):
    """
    Return the highest expected reward among the selected genre arms.
    """
    mean_rewards = [
        np.mean(genre_rewards[genre])
        for genre in selected_genres
        if genre_rewards[genre]
    ]
    return max(mean_rewards) if mean_rewards else 0.0


def run_ucb(
    genre_rewards,
    initial_genres,
    num_rounds,
    exploration_scale,
    rng,
    switch_rounds=None,
):
    """
    Run one UCB simulation and return cumulative regret.
    """
    selected_genres = list(initial_genres)
    num_arms = len(selected_genres)

    total_rewards = np.zeros(num_arms)
    total_selections = np.zeros(num_arms)
    cumulative_regret = np.zeros(num_rounds)

    current_segment_start = 0
    reward_samples = {
        genre: rng.choice(genre_rewards[genre], size=num_rounds, replace=True)
        for genre in selected_genres
    }

    optimal_reward = calculate_optimal_reward(genre_rewards, selected_genres)
    total_regret = 0.0

    for t in range(num_rounds):
        if switch_rounds and t in switch_rounds:
            selected_genres = select_random_genres(
                genre_rewards,
                num_arms,
                rng,
            )

            total_rewards = np.zeros(num_arms)
            total_selections = np.zeros(num_arms)
            current_segment_start = t

            remaining_rounds = num_rounds - t
            reward_samples = {
                genre: rng.choice(
                    genre_rewards[genre],
                    size=remaining_rounds,
                    replace=True,
                )
                for genre in selected_genres
            }

            optimal_reward = calculate_optimal_reward(
                genre_rewards,
                selected_genres,
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

        selected_genre = selected_genres[arm_index]
        reward = reward_samples[selected_genre][segment_t]

        total_rewards[arm_index] += reward
        total_selections[arm_index] += 1

        total_regret += optimal_reward - reward
        cumulative_regret[t] = total_regret

    return cumulative_regret


def run_thompson_sampling(
    genre_rewards,
    initial_genres,
    num_rounds,
    exploration_scale,
    rng,
    switch_rounds=None,
):
    """
    Run one Thompson Sampling simulation and return cumulative regret.

    The implementation follows the normal-distribution sampling approach
    used in the original project.
    """
    selected_genres = list(initial_genres)
    num_arms = len(selected_genres)

    means = {genre: 0.0 for genre in selected_genres}
    counts = {genre: 1 for genre in selected_genres}
    cumulative_regret = np.zeros(num_rounds)

    current_segment_start = 0
    reward_samples = {
        genre: rng.choice(genre_rewards[genre], size=num_rounds, replace=True)
        for genre in selected_genres
    }

    optimal_reward = calculate_optimal_reward(genre_rewards, selected_genres)
    total_regret = 0.0

    for t in range(num_rounds):
        if switch_rounds and t in switch_rounds:
            selected_genres = select_random_genres(
                genre_rewards,
                num_arms,
                rng,
            )

            means = {genre: 0.0 for genre in selected_genres}
            counts = {genre: 1 for genre in selected_genres}
            current_segment_start = t

            remaining_rounds = num_rounds - t
            reward_samples = {
                genre: rng.choice(
                    genre_rewards[genre],
                    size=remaining_rounds,
                    replace=True,
                )
                for genre in selected_genres
            }

            optimal_reward = calculate_optimal_reward(
                genre_rewards,
                selected_genres,
            )

        segment_t = t - current_segment_start

        sampled_rewards = {
            genre: rng.normal(
                means[genre],
                np.sqrt(exploration_scale**2 / counts[genre]),
            )
            for genre in selected_genres
        }

        selected_genre = max(sampled_rewards, key=sampled_rewards.get)
        reward = reward_samples[selected_genre][segment_t]

        current_count = counts[selected_genre]
        current_mean = means[selected_genre]

        means[selected_genre] = (
            current_mean * current_count + reward
        ) / (current_count + 1)

        counts[selected_genre] += 1

        total_regret += optimal_reward - reward
        cumulative_regret[t] = total_regret

    return cumulative_regret


def run_repeated_experiments(
    algorithm,
    genre_rewards,
    num_genres,
    num_rounds,
    exploration_scale,
    num_experiments,
    rng,
    switch_rounds=None,
):
    """
    Run repeated experiments and return mean cumulative regret
    and one-standard-deviation error values.
    """
    all_regrets = np.zeros((num_experiments, num_rounds))

    for experiment in range(num_experiments):
        print(
            f"{algorithm.__name__}: "
            f"experiment {experiment + 1}/{num_experiments}"
        )

        initial_genres = select_random_genres(
            genre_rewards,
            num_genres,
            rng,
        )

        all_regrets[experiment] = algorithm(
            genre_rewards=genre_rewards,
            initial_genres=initial_genres,
            num_rounds=num_rounds,
            exploration_scale=exploration_scale,
            rng=rng,
            switch_rounds=switch_rounds,
        )

    return all_regrets.mean(axis=0), all_regrets.std(axis=0)


def plot_results(results, output_path):
    """Plot and save the four genre experiment conditions."""
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
        "UCB vs. Thompson Sampling with 7 Genre Arms"
    )
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()


def main():
    genre_rewards, _ = prepare_reward_data()

    rng = np.random.default_rng(RANDOM_SEED)

    static_ucb_mean, static_ucb_std = run_repeated_experiments(
        algorithm=run_ucb,
        genre_rewards=genre_rewards,
        num_genres=NUM_GENRES,
        num_rounds=NUM_ROUNDS,
        exploration_scale=EXPLORATION_SCALE,
        num_experiments=NUM_EXPERIMENTS,
        rng=rng,
    )

    static_ts_mean, static_ts_std = run_repeated_experiments(
        algorithm=run_thompson_sampling,
        genre_rewards=genre_rewards,
        num_genres=NUM_GENRES,
        num_rounds=NUM_ROUNDS,
        exploration_scale=EXPLORATION_SCALE,
        num_experiments=NUM_EXPERIMENTS,
        rng=rng,
    )

    dynamic_ucb_mean, dynamic_ucb_std = run_repeated_experiments(
        algorithm=run_ucb,
        genre_rewards=genre_rewards,
        num_genres=NUM_GENRES,
        num_rounds=NUM_ROUNDS,
        exploration_scale=EXPLORATION_SCALE,
        num_experiments=NUM_EXPERIMENTS,
        rng=rng,
        switch_rounds=SWITCH_ROUNDS,
    )

    dynamic_ts_mean, dynamic_ts_std = run_repeated_experiments(
        algorithm=run_thompson_sampling,
        genre_rewards=genre_rewards,
        num_genres=NUM_GENRES,
        num_rounds=NUM_ROUNDS,
        exploration_scale=EXPLORATION_SCALE,
        num_experiments=NUM_EXPERIMENTS,
        rng=rng,
        switch_rounds=SWITCH_ROUNDS,
    )

    results = [
        ("UCB - Static", static_ucb_mean, static_ucb_std),
        ("TS - Static", static_ts_mean, static_ts_std),
        ("UCB - Dynamic", dynamic_ucb_mean, dynamic_ucb_std),
        ("TS - Dynamic", dynamic_ts_mean, dynamic_ts_std),
    ]

    output_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "genre_comparison.png"
    )

    plot_results(results, output_path)


if __name__ == "__main__":
    main()
