def extract_winner(df):
    return df.groupby(['game_id'])['winner'].first()
