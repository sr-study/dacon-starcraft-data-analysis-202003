import numpy as np


def sample_data(df, n_games, seed=None):
    if seed is not None:
        np.random.seed(seed)

    game_ids = df['game_id'].unique()
    sampled_game_ids = np.random.choice(game_ids, size=n_games, replace=False)

    return df[df['game_id'].isin(sampled_game_ids)]
