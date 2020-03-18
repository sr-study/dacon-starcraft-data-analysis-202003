def get_swap_table(x_data):
    return x_data['p0_species'] > x_data['p1_species']

def swap_x_data(x_data, swap_table):
    def _swap_x_data(x_data):
        df = x_data.copy()
        columns = df.columns

        p0_columns = columns[columns.str.startswith('p0_')]
        p1_columns = columns[columns.str.startswith('p1_')]
        delta_columns = columns[columns.str.startswith('delta_')]

        df[p0_columns], df[p1_columns] = df[p1_columns], df[p0_columns]
        df[delta_columns] = -df[delta_columns]
        return df

    result = x_data.copy()
    swapped_x_data = _swap_x_data(x_data)
    result.loc[swap_table] = swapped_x_data.loc[swap_table]
    return result

def swap_y_data(y_data, swap_table):
    def _swap_y_data(y_data):
        return 1 - y_data

    result = y_data.copy()
    swapped_y_data = _swap_y_data(y_data)
    result[swap_table] = swapped_y_data[swap_table]
    return result
