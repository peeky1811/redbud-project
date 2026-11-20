
"""MLayalee: Data-quality sensitivity analysis."""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parent

# Load the fixed cleaned dataset and development IDs.
data = pd.read_csv(ROOT / 'redbud_cleaned_candidates.csv')
dev_ids = pd.read_csv(ROOT / 'development_ids.csv')

# Reconstruct the same development dataset.
train = dev_ids.merge(
    data,
    on='id',
    how='left',
    validate='one_to_one',
    indicator=True
)

assert train['_merge'].eq('both').all()
train = train.drop(columns='_merge')

assert len(train) == 809
assert train['id'].is_unique

# Check the data types and values before filtering.
print('Cultivated values:')
print(train['captive_cultivated'].value_counts(dropna=False))

print('\nQuality grades:')
print(train['quality_grade'].value_counts(dropna=False))

# Convert the cultivated flag safely.
cultivated = (
    train['captive_cultivated']
    .astype('string')
    .str.strip()
    .str.lower()
)

assert cultivated.isin(['true', 'false']).all()

train['is_cultivated'] = cultivated.eq('true')

# Define the three dataset versions.
datasets = {
    'all_observations': train,
    'exclude_cultivated': train.loc[
        ~train['is_cultivated']
    ].copy(),
    'research_only': train.loc[
        train['quality_grade'] == 'research'
    ].copy()
}

# Compare class counts and class proportions.
summary = []

for name, subset in datasets.items():

    flowering = int((subset['flowering'] == 1).sum())
    no_flowers = int((subset['flowering'] == 0).sum())

    summary.append({
        'dataset': name,
        'total': len(subset),
        'flowering': flowering,
        'no_flowers_fruits': no_flowers,
        'flowering_percentage':
            round(100 * flowering / len(subset), 2)
    })

summary_df = pd.DataFrame(summary)

print('\nData-quality filtering summary:')
print(summary_df.to_string(index=False))

summary_df.to_csv(
    ROOT / 'data_quality_summary.csv',
    index=False
)

print('\nSaved data_quality_summary.csv')
print('Original cleaned dataset and split files unchanged.')

# Data-quality sensitivity experiment.
# Filter training folds only; keep validation folds identical.

features = ['day_sin', 'day_cos', 'latitude', 'longitude']
seeds = [42, 123, 2026]

pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('model', LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

param_grid = {
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__class_weight': [None, 'balanced']
}

results = []

for seed in seeds:
    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    for fold, (train_idx, val_idx) in enumerate(
        outer_cv.split(train[features], train['flowering']),
        start=1
    ):
        outer_train = train.iloc[train_idx].copy()
        validation = train.iloc[val_idx].copy()

        # Apply filtering only to outer training observations.
        training_sets = {
            'all_observations': outer_train,
            'exclude_cultivated': outer_train.loc[
                ~outer_train['is_cultivated']
            ].copy(),
            'research_only': outer_train.loc[
                outer_train['quality_grade'] == 'research'
            ].copy()
        }

        X_val = validation[features]
        y_val = validation['flowering']

        for name, subset in training_sets.items():
            X_train = subset[features]
            y_train = subset['flowering']

            inner_cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=seed
            )

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring='balanced_accuracy',
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
                error_score='raise'
            )

            search.fit(X_train, y_train)

            predictions = search.predict(X_val)
            probabilities = search.predict_proba(X_val)[:, 1]

            results.append({
                'dataset': name,
                'seed': seed,
                'fold': fold,
                'training_size': len(subset),
                'validation_size': len(validation),
                'best_C': search.best_params_['model__C'],
                'best_class_weight':
                    search.best_params_['model__class_weight'],
                'balanced_accuracy': balanced_accuracy_score(
                    y_val, predictions
                ),
                'accuracy': accuracy_score(
                    y_val, predictions
                ),
                'precision': precision_score(
                    y_val, predictions, zero_division=0
                ),
                'recall': recall_score(
                    y_val, predictions, zero_division=0
                ),
                'f1': f1_score(
                    y_val, predictions, zero_division=0
                ),
                'roc_auc': roc_auc_score(
                    y_val, probabilities
                )
            })

        print(f'Completed seed {seed}, fold {fold}')

# Save individual fold results.
results_df = pd.DataFrame(results)

results_df.to_csv(
    ROOT / 'data_quality_cv_results.csv',
    index=False
)

metrics = [
    'balanced_accuracy', 'accuracy',
    'precision', 'recall', 'f1', 'roc_auc'
]

# Summarize the three training strategies.
summary = results_df.groupby('dataset')[metrics].agg(
    ['mean', 'std']
)

summary.to_csv(
    ROOT / 'data_quality_cv_summary.csv'
)

print('\nData-quality sensitivity results:')

for name in training_sets:
    subset_results = results_df[
        results_df['dataset'] == name
    ]

    print('\n', name)

    for metric in metrics:
        values = subset_results[metric]

        print(
            f'  {metric}: '
            f'{values.mean():.3f} +/- '
            f'{values.std(ddof=1):.3f}'
        )

# Paired comparison against the unfiltered model.
paired = results_df.pivot(
    index=['seed', 'fold'],
    columns='dataset',
    values=metrics
)

differences = pd.DataFrame(index=paired.index)

for filtered_name in ['exclude_cultivated', 'research_only']:
    for metric in metrics:
        differences[f'{filtered_name}_{metric}'] = (
            paired[(metric, filtered_name)]
            - paired[(metric, 'all_observations')]
        )

differences.to_csv(
    ROOT / 'data_quality_cv_differences.csv'
)

print('\nBalanced accuracy differences vs all observations:')

for name in ['exclude_cultivated', 'research_only']:
    values = differences[f'{name}_balanced_accuracy']

    print(
        f'  {name}: '
        f'{values.mean():+.4f} +/- '
        f'{values.std(ddof=1):.4f}'
    )

print('\nData-quality sensitivity analysis completed.')
print('Reserved test set remains untouched.')
