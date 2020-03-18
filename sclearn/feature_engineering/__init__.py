import gc
import multiprocessing

import numpy as np
import pandas as pd

from .ability_counts import AbilityCounts
from .camera_state import CameraState
from .extract_event_counts import extract_event_counts
from .extract_game_states import extract_game_states
from .extract_playtime import extract_playtime
from .extract_species import extract_species
from .extract_winner import extract_winner


def prepare_x_data(df):
    if 'winner' in df.columns:
        df = df.drop(columns=['winner'])

    features = []
    features.append(extract_playtime(df))
    features.append(extract_species(df))
    features.append(extract_event_counts(df))
    features.append(extract_game_states(df, [
        CameraState(),
        AbilityCounts(),
    ]))

    return pd.concat(features, axis=1)

def prepare_y_data(df):
    winners = extract_winner(df)
    return np.array(winners)

def split_dataframe_by_game_id(df, n_sections):
    game_ids = df['game_id'].unique()
    if len(game_ids) < n_sections:
        n_sections = len(game_ids)

    game_id_groups = np.array_split(game_ids, n_sections)
    return [df[df['game_id'].isin(ids)] for ids in game_id_groups]

def parallelize_dataframe(func, df, n_cores=None):
    if n_cores is None:
        n_cores = multiprocessing.cpu_count()

    assert n_cores > 0

    df_split = split_dataframe_by_game_id(df, n_cores)
    n_cores = len(df_split)

    pool = multiprocessing.Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()

    del df_split
    gc.collect()

    return df
