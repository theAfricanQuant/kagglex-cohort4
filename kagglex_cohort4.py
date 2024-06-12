#
# Ricky Macharm, MScFE
# www.SisengAI.com
#

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def shrink_ints(df):
    mapping = {}
    for col in df.dtypes[df.dtypes=='int64[pyarrow]'].index:
        max_ = df[col].max()
        min_ = df[col].min()
        if min_ < 0:
            continue
        if max_ < 255:
            mapping[col] = 'uint8[pyarrow]'
        elif max_ < 65_535:
            mapping[col] = 'uint16[pyarrow]'
        elif max_ <  4294967295:
            mapping[col] = 'uint32[pyarrow]'
    return df.astype(mapping)


def clean_housing(df):
    return (df
     .assign(**df.select_dtypes('string').astype('category'),)
     .pipe(shrink_ints)
    )    


def hyperparameter_search(X_train, y_train, estimator, param_grid, search_strategy='grid', n_iter=10):
    if search_strategy == 'grid':
        search = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
    elif search_strategy == 'random':
        search = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid, n_iter=n_iter, scoring='neg_root_mean_squared_error', cv=3, verbose=1, random_state=42, n_jobs=-1)
    else:
        raise ValueError("search_strategy must be either 'grid' or 'random'")
    
    # Step 3: Execute the Search
    search.fit(X_train, y_train)
    
    # Step 4: Evaluate and Iterate
    print(f"Best parameters found: {search.best_params_}")
    print(f"Best score (negative RMSE): {-search.best_score_}")
    
    return search.best_estimator_