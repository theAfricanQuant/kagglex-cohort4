#
#
# Ricky Macharm, MScFE
# www.SisengAI.com
# © 2024 SisengAI. All rights reserved.
#
#

import pandas as pd
import numpy as np
import re
from datetime import datetime

def shrink_ints(df):
    mapping = {}
    for col in df.dtypes[df.dtypes == 'int64[pyarrow]'].index:
        max_ = df[col].max()
        min_ = df[col].min()
        if min_ < 0:
            continue
        if max_ < 255:
            mapping[col] = 'uint8[pyarrow]'
        elif max_ < 65_535:
            mapping[col] = 'uint16[pyarrow]'
        elif max_ < 4294967295:
            mapping[col] = 'uint32[pyarrow]'
    return df.astype(mapping)

def prep_data(df):
    current_year = datetime.now().year

    def extract_hp(engine):
        match = re.search(r'(\d+\.?\d*)HP', engine)
        return float(match.group(1)) if match else np.nan

    def extract_cylinders(engine):
        match = re.search(r'(\d+) Cylinder', engine)
        return int(match.group(1)) if match else np.nan

    def fill_cylinders(row):
        if pd.isna(row['cylinders']):
            if row['fuel_type'] == 'Electric':
                return 0
            else:
                return 4
        return row['cylinders']

    return (df
        # Drop the id column
        .drop(columns=['id'])

        # Convert string columns to category
        .assign(**df.select_dtypes('string').astype('category'))
        
        # Shrink integer columns
        .pipe(shrink_ints)
        
        # Convert model year to age
        .assign(age=lambda x: current_year - x['model_year'])
        .drop(columns=['model_year'])
        
        # Extract horsepower and cylinders from engine
        .assign(
            horsepower=lambda x: x['engine'].apply(extract_hp),
            cylinders=lambda x: x['engine'].apply(extract_cylinders)
        )
        .drop(columns=['engine'])
        
        # Fill missing horsepower with 150 HP
        .assign(horsepower=lambda x: x['horsepower'].fillna(150))
        
        # Fill missing cylinders based on fuel type
        .assign(cylinders=lambda x: x.apply(fill_cylinders, axis=1))
        
        # Simplify transmission types
        .assign(transmission=lambda x: x['transmission'].apply(lambda val: 'Automatic' if 'A/T' in val else 'Manual' if 'M/T' in val else 'Other'))
        
        # Encode accident and clean title as binary
        .assign(
            accident=lambda x: x['accident'].apply(lambda val: 1 if 'accident' in val.lower() else 0),
            clean_title=lambda x: x['clean_title'].apply(lambda val: 1 if val == 'Yes' else 0)
        )
        
        # Log transform the target variable
        .assign(price=lambda x: np.log1p(x['price']))
    )


