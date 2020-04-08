import pandas as pd
import numpy as np

def get_list(values):
    base_pos_list = list()
    for value in values:
        base_pos_list.append(value)

    return base_pos_list


def calc_p0_valid_attack(row):
    p0_attack_x_list = row['p0_attack_x']
    p0_attack_y_list = row['p0_attack_y']

    p1_base_x_list = row['p1_base_x']
    p1_base_y_list = row['p1_base_y']

    cnt = 0
    for i in range(len(p0_attack_x_list)):
        min_dist = 987654321.0

        for j in range(len(p1_base_x_list)):
            cur_dist = np.sqrt(((p1_base_x_list[j] - p0_attack_x_list[i]) ** 2) +
                               ((p1_base_y_list[j] - p0_attack_y_list[i]) ** 2))

            min_dist = min(min_dist, cur_dist)
        if min_dist <= 100.0:
            cnt += 1

    row['p0_valid_attack'] = cnt
    return row


def calc_p1_valid_attack(row):
    p1_attack_x_list = row['p1_attack_x']
    p1_attack_y_list = row['p1_attack_y']

    p0_base_x_list = row['p0_base_x']
    p0_base_y_list = row['p0_base_y']

    cnt = 0
    for i in range(len(p1_attack_x_list)):
        min_dist = 987654321.0

        for j in range(len(p0_base_x_list)):
            cur_dist = np.sqrt(((p0_base_x_list[j] - p1_attack_x_list[i]) ** 2) +
                               ((p0_base_y_list[j] - p1_attack_y_list[i]) ** 2))

            min_dist = min(min_dist, cur_dist)
        if min_dist <= 100.0:
            cnt += 1

    row['p1_valid_attack'] = cnt
    return row


def merge_base_and_preprocess(row):
    p0_base_x_list = row['p0_base_x']
    p0_base_y_list = row['p0_base_y']

    if type(p0_base_x_list) == float:
        p0_base_x_list = list()
        p0_base_y_list = list()

    if np.isnan(row['p0_start_base_x']) is False:
        p0_base_x_list.append(row['p0_start_base_x'])
        p0_base_y_list.append(row['p0_start_base_y'])

    row['p0_base_x'] = p0_base_x_list
    row['p0_base_y'] = p0_base_y_list

    p1_base_x_list = row['p1_base_x']
    p1_base_y_list = row['p1_base_y']

    if type(p1_base_x_list) == float:
        p1_base_x_list = list()
        p1_base_y_list = list()

    if np.isnan(row['p1_start_base_x']) != float:
        p1_base_x_list.append(row['p1_start_base_x'])
        p1_base_y_list.append(row['p1_start_base_y'])

    row['p1_base_x'] = p1_base_x_list
    row['p1_base_y'] = p1_base_y_list

    if type(row['p0_attack_x']) == float:
        row['p0_attack_x'] = list()

    if type(row['p0_attack_y']) == float:
        row['p0_attack_y'] = list()

    if type(row['p1_attack_x']) == float:
        row['p1_attack_x'] = list()

    if type(row['p1_attack_y']) == float:
        row['p1_attack_y'] = list()

    return row


def extract_base_attack_cnt(df):

    ability_df = df[df['event'] == 'Ability']
    player_0_ability_df = ability_df[ability_df['player'] == 0]
    player_1_ability_df = ability_df[ability_df['player'] == 1]

    # This code is from https://dacon.io/competitions/official/235583/codeshare/743
    df_train = pd.DataFrame(df.game_id.unique(), columns=['game_id'])
    df_train.index = df_train.game_id
    df_train = df_train.drop(['game_id'], axis=1)

    df_train_p0 = df[(df.event == 'Camera') & (df.player == 0)]
    df_train_p0 = df_train_p0[df_train_p0.shift(1).game_id != df_train_p0.game_id]  # 쉬프트를 이용하여 각 게임의 첫번째 데이터 찾기
    df_train_p0 = df_train_p0.iloc[:, [0, 5]].rename({'event_contents': 'player0_starting'}, axis=1)
    df_train_p0.index = df_train_p0['game_id']
    df_train_p0 = df_train_p0.drop(['game_id'], axis=1)
    df_train = pd.merge(df_train, df_train_p0, on='game_id', how='left')

    df_train_p1 = df[(df.event == 'Camera') & (df.player == 1)]
    df_train_p1 = df_train_p1[df_train_p1.shift(1).game_id != df_train_p1.game_id]
    df_train_p1 = df_train_p1.iloc[:, [0, 5]].rename({'event_contents': 'player1_starting'}, axis=1)
    df_train_p1.index = df_train_p1['game_id']
    df_train_p1 = df_train_p1.drop(['game_id'], axis=1)
    df_train = pd.merge(df_train, df_train_p1, on='game_id', how='left')

    # player0 attack position
    player0_attack_df = player_0_ability_df[(player_0_ability_df['event_contents'].str.contains('Attack'))]
    player0_attack_locations_df = player0_attack_df['event_contents'].str.extract(
        r'(?P<p0_attack_x>\d+\.\d+)\, (?P<p0_attack_y>\d+\.\d+)')
    player0_attack_df = pd.concat([player0_attack_df, player0_attack_locations_df], axis=1)
    player0_attack_df['p0_attack_x'] = player0_attack_df['p0_attack_x'].astype(float)
    player0_attack_df['p0_attack_y'] = player0_attack_df['p0_attack_y'].astype(float)

    p0_grp_attack_x_df = player0_attack_df.groupby('game_id').p0_attack_x.agg(get_list)
    p0_grp_attack_y_df = player0_attack_df.groupby('game_id').p0_attack_y.agg(get_list)

    p0_attack_list_df = pd.concat([p0_grp_attack_x_df, p0_grp_attack_y_df], axis=1)

    # player1 attack position
    player1_attack_df = player_1_ability_df[(player_1_ability_df['event_contents'].str.contains('Attack'))]
    player1_attack_locations_df = player1_attack_df['event_contents'].str.extract(
        r'(?P<p1_attack_x>\d+\.\d+)\, (?P<p1_attack_y>\d+\.\d+)')
    player1_attack_df = pd.concat([player1_attack_df, player1_attack_locations_df], axis=1)
    player1_attack_df['p1_attack_x'] = player1_attack_df['p1_attack_x'].astype(float)
    player1_attack_df['p1_attack_y'] = player1_attack_df['p1_attack_y'].astype(float)

    p1_grp_attack_x_df = player1_attack_df.groupby('game_id').p1_attack_x.agg(get_list)
    p1_grp_attack_y_df = player1_attack_df.groupby('game_id').p1_attack_y.agg(get_list)

    p1_attack_list_df = pd.concat([p1_grp_attack_x_df, p1_grp_attack_y_df], axis=1)

    # player0 base position
    player0_base_position_df = player_0_ability_df[player_0_ability_df['event_contents'].str.contains('CommandCenter') |
                                                   player_0_ability_df['event_contents'].str.contains('BuildNexus') |
                                                   player_0_ability_df['event_contents'].str.contains('BuildHatchery')]

    p0_base_df = pd.concat([player_0_ability_df['game_id'],
                            player0_base_position_df['event_contents'].str.extract(
                                r'(?P<p0_base_x>\d+\.\d+)\, (?P<p0_base_y>\d+\.\d+)')],
                           axis=1)
    p0_base_df['p0_base_x'] = p0_base_df['p0_base_x'].astype(float)
    p0_base_df['p0_base_y'] = p0_base_df['p0_base_y'].astype(float)
    p0_base_df = p0_base_df.dropna()

    p0_base_grp_x_df = p0_base_df.groupby('game_id').p0_base_x.agg(get_list)
    p0_base_grp_y_df = p0_base_df.groupby('game_id').p0_base_y.agg(get_list)

    p0_base_list_df = pd.concat([p0_base_grp_x_df, p0_base_grp_y_df], axis=1)

    # player1 base position
    player1_base_position_df = player_1_ability_df[player_1_ability_df['event_contents'].str.contains('CommandCenter') |
                                                   player_1_ability_df['event_contents'].str.contains('BuildNexus') |
                                                   player_1_ability_df['event_contents'].str.contains('BuildHatchery')]

    p1_base_df = pd.concat([player_1_ability_df['game_id'],
                            player1_base_position_df['event_contents'].str.extract(
                                r'(?P<p1_base_x>\d+\.\d+)\, (?P<p1_base_y>\d+\.\d+)')],
                           axis=1)
    p1_base_df['p1_base_x'] = p1_base_df['p1_base_x'].astype(float)
    p1_base_df['p1_base_y'] = p1_base_df['p1_base_y'].astype(float)
    p1_base_df = p1_base_df.dropna()

    p1_base_grp_x_df = p1_base_df.groupby('game_id').p1_base_x.agg(get_list)
    p1_base_grp_y_df = p1_base_df.groupby('game_id').p1_base_y.agg(get_list)

    p1_base_list_df = pd.concat([p1_base_grp_x_df, p1_base_grp_y_df], axis=1)

    listed_df = pd.concat([p0_base_list_df, p1_base_list_df, p0_attack_list_df, p1_attack_list_df], axis=1)
    listed_df = listed_df.reset_index()

    df_train = df_train.reset_index()
    extract_p0_start_df = df_train['player0_starting'].str.extract(r'(?P<p0_start_base_x>\d+\.\d+)\, (?P<p0_start_base_y>\d+\.\d+)')
    extract_p0_start_df['p0_start_base_x'] = extract_p0_start_df['p0_start_base_x'].astype(float)
    extract_p0_start_df['p0_start_base_y'] = extract_p0_start_df['p0_start_base_y'].astype(float)
    df_train = pd.concat([df_train, extract_p0_start_df], axis=1)

    extract_p1_start_df = df_train['player1_starting'].str.extract(
        r'(?P<p1_start_base_x>\d+\.\d+)\, (?P<p1_start_base_y>\d+\.\d+)')
    extract_p1_start_df['p1_start_base_x'] = extract_p1_start_df['p1_start_base_x'].astype(float)
    extract_p1_start_df['p1_start_base_y'] = extract_p1_start_df['p1_start_base_y'].astype(float)
    df_train = pd.concat([df_train, extract_p1_start_df], axis=1)

    listed_df = pd.merge(listed_df, df_train[
        ['game_id', 'p0_start_base_x', 'p0_start_base_y', 'p1_start_base_x', 'p1_start_base_y']],
                         on='game_id', how='left')

    listed_df = listed_df.apply(merge_base_and_preprocess, axis=1)

    listed_df = listed_df.apply(calc_p0_valid_attack, axis=1)
    listed_df = listed_df.apply(calc_p1_valid_attack, axis=1)
    listed_df['delta_valid_attack'] = listed_df['p0_valid_attack'] - listed_df['p1_valid_attack']
    listed_df = listed_df.set_index('game_id')

    return listed_df[['p0_valid_attack', 'p1_valid_attack', 'delta_valid_attack']]
