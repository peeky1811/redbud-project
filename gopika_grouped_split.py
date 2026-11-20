
"""MLayalee: Observer-grouped validation for logistic regression."""

from pathlib import Path
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
ROOT = Path(__file__).resolve().parent

# Load the cleaned dataset and fixed development IDs.
data = pd.read_csv(ROOT / 'redbud_cleaned_candidates.csv')
dev_ids = pd.read_csv(ROOT / 'development_ids.csv')

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
assert train['user_id'].notna().all()

# Use the same logistic regression features as the main model.
features = ['day_sin', 'day_cos', 'latitude', 'longitude']

X = train[features]
y = train['flowering']
groups = train['user_id']

print('Development observations:', len(train))
print('Unique observers:', groups.nunique())
print('\nClass counts:')
print(y.value_counts())

print('\nObservations per observer:')
print(groups.value_counts().describe())

# Compare standard and observer-grouped cross-validation.

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

for method in ['standard', 'observer_grouped']:

    print('\nRunning:', method)

    for seed in seeds:

        if method == 'standard':
            outer_cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=seed
            )
            splits = outer_cv.split(X, y)
        else:
            outer_cv = StratifiedGroupKFold(
                n_splits=5,
                shuffle=True,
                random_state=seed
            )
            splits = outer_cv.split(X, y, groups)

        for fold, (train_idx, val_idx) in enumerate(
            splits, start=1
        ):

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            train_groups = groups.iloc[train_idx]
            val_groups = groups.iloc[val_idx]

            if method == 'observer_grouped':

                # Verify no observer is in both partitions.
                assert set(train_groups).isdisjoint(
                    set(val_groups)
                )

                inner_cv = StratifiedGroupKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=seed
                )

            else:
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

            # Group information is required for grouped tuning.
            if method == 'observer_grouped':
                search.fit(
                    X_train,
                    y_train,
                    groups=train_groups
                )
            else:
                search.fit(X_train, y_train)

            predictions = search.predict(X_val)
            probabilities = search.predict_proba(X_val)[:, 1]

            results.append({
                'method': method,
                'seed': seed,
                'fold': fold,
                'training_size': len(train_idx),
                'validation_size': len(val_idx),
                'training_observers': train_groups.nunique(),
                'validation_observers': val_groups.nunique(),
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
    ROOT / 'grouped_split_results.csv',
    index=False
)

metrics = [
    'balanced_accuracy',
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc'
]

summary = results_df.groupby('method')[metrics].agg(
    ['mean', 'std']
)

summary.to_csv(
    ROOT / 'grouped_split_summary.csv'
)

print('\nGrouped-split comparison:')

for method in ['standard', 'observer_grouped']:

    subset = results_df[
        results_df['method'] == method
    ]

    print('\n', method)

    for metric in metrics:
        values = subset[metric]

        print(
            f'  {metric}: '
            f'{values.mean():.3f} +/- '
            f'{values.std(ddof=1):.3f}'
        )

# Compare average balanced accuracy.
standard_mean = results_df.loc[
    results_df['method'] == 'standard',
    'balanced_accuracy'
].mean()

grouped_mean = results_df.loc[
    results_df['method'] == 'observer_grouped',
    'balanced_accuracy'
].mean()

print(
    '\nGrouped minus standard balanced accuracy:',
    f'{grouped_mean - standard_mean:+.4f}'
)

print('\nSaved grouped_split_results.csv')
print('Saved grouped_split_summary.csv')
print('Reserved test set remains untouched.')
