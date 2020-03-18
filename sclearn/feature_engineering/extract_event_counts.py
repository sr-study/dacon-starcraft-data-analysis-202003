import pandas as pd


def extract_event_counts(df):
    events = ['Ability', 'AddToControlGroup', 'Camera', 'ControlGroup', 'GetControlGroup', 'Right Click', 'Selection', 'SetControlGroup']

    event_counts = df.groupby(['game_id', 'player'])['event'].value_counts()
    event_counts = event_counts.unstack(level=-1).unstack(level=-1)
    event_counts.columns = event_counts.columns.map(lambda x: f'p{x[1]}_event_{x[0]}')
    event_counts = event_counts.fillna(0)

    result = pd.DataFrame(index=event_counts.index)

    for player in ['p0', 'p1']:
        for event in events:
            result[f'{player}_event_{event}'] = event_counts.get(f'{player}_event_{event}', 0.0)

    for event in events:
        result[f'delta_event_{event}'] = result[f'p0_event_{event}'] - result[f'p1_event_{event}']

    return result
