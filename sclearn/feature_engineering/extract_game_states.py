import gc

import pandas as pd
from tqdm.auto import tqdm


class GameState:
    def init(self):
        pass

    def update(self, game_id, time, player, species, event, event_contents):
        pass

    def to_dict(self):
        return {}


class GameStateManager:
    def __init__(self):
        self._states = []

    def add(self, game_state):
        self._states.append(game_state)

    def init(self):
        for state in self._states:
            state.init()

    def update(self, game_id, time, player, species, event, event_contents):
        for state in self._states:
            state.update(game_id, time, player, species, event, event_contents)

    def to_dict(self):
        ret = {}
        for state in self._states:
            ret.update(state.to_dict())
        return ret


def extract_game_states(df, game_states):
    mat = df.to_numpy()

    data = {}

    cur_game_id = -1

    game_state = GameStateManager()
    for state in game_states:
        game_state.add(state)

    for row in tqdm(mat):
        game_id, time, player, species, event, event_contents = row

        if game_id != cur_game_id:
            if cur_game_id != -1:
                data[cur_game_id] = game_state.to_dict()

            cur_game_id = game_id
            game_state.init()

        game_state.update(game_id, time, player, species, event, event_contents)

    if cur_game_id != -1:
        data[cur_game_id] = game_state.to_dict()

    del mat
    gc.collect()

    return pd.DataFrame.from_dict(data, orient='index')
