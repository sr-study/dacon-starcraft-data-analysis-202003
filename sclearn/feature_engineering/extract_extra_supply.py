import pandas as pd


def extract_extra_supply(x_data):
    abilities = [
        ('(1020) - BuildCommandCenter', 15),
        ('(1021) - BuildSupplyDepot', 8),
        ('(1540) - BuildNexus', 15),
        ('(1541) - BuildPylon', 8),
        ('(16E0) - BuildHatchery', 6),
        ('(1822) - MorphOverlord', 8),
        ('(1BA0) - MorphToOverseer', 2),
    ]

    p0_extra_supply = sum([x_data[f'p0_ability_{v[0]}'] * v[1] for v in abilities])
    p1_extra_supply = sum([x_data[f'p1_ability_{v[0]}'] * v[1] for v in abilities])
    delta_extra_supply = p0_extra_supply - p1_extra_supply

    p0_extra_supply.name = 'p0_extra_supply'
    p1_extra_supply.name = 'p1_extra_supply'
    delta_extra_supply.name = 'delta_extra_supply'
    
    return pd.concat([p0_extra_supply, p1_extra_supply, delta_extra_supply], axis='columns', copy=False)
