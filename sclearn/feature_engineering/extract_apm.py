import pandas as pd


def extract_apm(x_data):
    action_events = ['Ability', 'AddToControlGroup', 'ControlGroup', 'GetControlGroup', 'Right Click', 'Selection', 'SetControlGroup']

    p0_apm = x_data[[f'p0_event_{event}' for event in action_events]].sum(axis='columns') / x_data['time']
    p1_apm = x_data[[f'p1_event_{event}' for event in action_events]].sum(axis='columns') / x_data['time']
    delta_apm = p0_apm - p1_apm

    p0_apm.name = 'p0_apm'
    p1_apm.name = 'p1_apm'
    delta_apm.name = 'delta_apm'
    
    return pd.concat([p0_apm, p1_apm, delta_apm], axis='columns', copy=False)
