import pandas as pd


def _species_converter(string):
    if string == 'T':
        return 0
    elif string == 'P':
        return 1
    elif string == 'Z':
        return 2
    else:
        raise ValueError

def extract_species(df):
    species = df.groupby(['game_id', 'player'])['species'].first()

    species_df = species.unstack(level=-1)
    species_df.columns = species_df.columns.map(lambda x: f'p{x}_species')
    species_df.columns.name = None

    species_df = species_df.applymap(_species_converter)

    cat_dtype = pd.api.types.CategoricalDtype(
        categories=[0, 1, 2], ordered=True)
    species_df = species_df.astype(cat_dtype)

    return species_df
