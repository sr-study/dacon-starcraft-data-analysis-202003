def extract_playtime(df):
    def min_to_sec(t):
        m = int(t)
        s = (t - m) * 100
        return (m * 60) + s

    return df.groupby(['game_id'])['time'].max().apply(min_to_sec)
