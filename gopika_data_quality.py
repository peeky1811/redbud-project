
"""MLayalee: Data-quality sensitivity analysis."""

from pathlib import Path
import pandas as pd

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
