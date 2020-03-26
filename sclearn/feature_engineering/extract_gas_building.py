import pandas as pd


def extract_gas_building(x_data):
    abilities = ['(1022) - BuildRefinery', '(1542) - BuildAssimilator', '(16E2) - BuildExtractor']

    p0_gas_building = x_data[[f'p0_ability_{ability}' for ability in abilities]].sum(axis='columns')
    p1_gas_building = x_data[[f'p1_ability_{ability}' for ability in abilities]].sum(axis='columns')
    delta_gas_building = p0_gas_building - p1_gas_building

    p0_gas_building.name = 'p0_gas_building'
    p1_gas_building.name = 'p1_gas_building'
    delta_gas_building.name = 'delta_gas_building'
    
    return pd.concat([p0_gas_building, p1_gas_building, delta_gas_building], axis='columns', copy=False)
