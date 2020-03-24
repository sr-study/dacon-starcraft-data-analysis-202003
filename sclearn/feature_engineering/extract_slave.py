import pandas as pd


def extract_slave(x_data):
    abilities = ['(1360) - TrainSCV', '(15E0) - TrainProbe', '(1820) - MorphDrone']

    p0_slave = x_data[[f'p0_ability_{ability}' for ability in abilities]].sum(axis='columns')
    p1_slave = x_data[[f'p1_ability_{ability}' for ability in abilities]].sum(axis='columns')
    delta_slave = p0_slave - p1_slave

    p0_slave.name = 'p0_slave'
    p1_slave.name = 'p1_slave'
    delta_slave.name = 'delta_slave'
    
    return pd.concat([p0_slave, p1_slave, delta_slave], axis='columns', copy=False)
