
"""MLayalee: Logistic regression feature ablation study."""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent

# Load cleaned data and the fixed development IDs.
data = pd.read_csv(ROOT / 'redbud_cleaned_candidates.csv')
dev_ids = pd.read_csv(ROOT / 'development_ids.csv')

# Preserve the saved development observation order.
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

print('Development dataset shape:', train.shape)
print('Class counts:\n', train.flowering.value_counts())

# Define target and feature sets.
y = train['flowering']

feature_sets = {
    'seasonal_only': ['day_sin', 'day_cos'],
    'seasonal_geography': [
        'day_sin', 'day_cos', 'latitude', 'longitude'
    ]
}

for name, features in feature_sets.items():
    print('\nFeature set:', name)
    print('Features:', features)
    print('Input shape:', train[features].shape)

# Logistic regression pipeline.
pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('model', LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Use the same tuning grid for both feature sets.
param_grid = {
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__class_weight': [None, 'balanced']
}

seeds = [42, 123, 2026]

scoring = {
    'balanced_accuracy': 'balanced_accuracy',
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Compare both feature sets using nested cross-validation.
ablation_results = []

for name, features in feature_sets.items():

    X = train[features]

    print('\nRunning experiment:', name)

    for seed in seeds:

        # Outer CV measures model performance.
        outer_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed
        )

        # Inner CV selects the best hyperparameters.
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
            refit=True
        )

        scores = cross_validate(
            search,
            X,
            y,
            cv=outer_cv,
            scoring=scoring,
            return_estimator=True,
            error_score='raise'
        )

        # Save scores and selected parameters for each fold.
        for fold in range(5):

            best_params = (
                scores['estimator'][fold].best_params_
            )

            result = {
                'feature_set': name,
                'seed': seed,
                'fold': fold + 1,
                'best_C': best_params['model__C'],
                'best_class_weight':
                    best_params['model__class_weight']
            }

            for metric in scoring:
                result[metric] = scores[
                    'test_' + metric
                ][fold]

            ablation_results.append(result)

        print('Completed seed:', seed)

# Convert results into a DataFrame.
ablation_df = pd.DataFrame(ablation_results)

# Save individual fold results.
ablation_df.to_csv(
    ROOT / 'logistic_ablation_results.csv',
    index=False
)

# Calculate mean and standard deviation.
ablation_summary = ablation_df.groupby(
    'feature_set'
)[list(scoring.keys())].agg(['mean', 'std'])

ablation_summary.to_csv(
    ROOT / 'logistic_ablation_summary.csv'
)

print('\nLogistic regression ablation results:')

for name in feature_sets:

    model_results = ablation_df[
        ablation_df['feature_set'] == name
    ]

    print('\n', name)

    for metric in scoring:

        values = model_results[metric]

        print(
            f'  {metric}: '
            f'{values.mean():.3f} +/- '
            f'{values.std(ddof=1):.3f}'
        )

# Compare results on matching seeds and folds.
seasonal = ablation_df[
    ablation_df['feature_set'] == 'seasonal_only'
].set_index(['seed', 'fold'])

geography = ablation_df[
    ablation_df['feature_set'] == 'seasonal_geography'
].set_index(['seed', 'fold'])

assert seasonal.index.equals(geography.index)

# Positive differences indicate improved performance
# after adding latitude and longitude.
differences = (
    geography[list(scoring.keys())]
    - seasonal[list(scoring.keys())]
)

differences.to_csv(
    ROOT / 'logistic_ablation_differences.csv'
)

print('\nEffect of adding geographic features:')

for metric in scoring:

    print(
        f'  {metric}: '
        f'{differences[metric].mean():+.4f} +/- '
        f'{differences[metric].std(ddof=1):.4f}'
    )

print('\nAblation results saved successfully.')
print('Test set remains untouched.')
