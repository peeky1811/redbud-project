"""MLayalee: reproducible data audit, EDA, majority baseline, logistic regression.
Run: python gopika_data_eda_baseline.py
Place the two original iNaturalist ZIP exports in the same directory as this script.
"""
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

ROOT = Path(__file__).resolve().parent
FILES = [("texas-easternredbuds-flowering.csv.zip", 1), ("texas-easternredbuds-nonflowering.csv.zip", 0)]
frames=[]
for filename,label in FILES:
    with zipfile.ZipFile(ROOT / filename) as z:
        csv_name=next(n for n in z.namelist() if n.endswith('.csv'))
        df=pd.read_csv(z.open(csv_name))
    df['flowering']=label
    frames.append(df)
data=pd.concat(frames, ignore_index=True)
print('Raw shape:',data.shape)
print('Duplicate IDs:',data.id.duplicated().sum())
print('Class counts:\n',data.flowering.value_counts())
print('Quality grades:\n',pd.crosstab(data.quality_grade,data.flowering))
print('Cultivated:\n',pd.crosstab(data.captive_cultivated,data.flowering))

# Preserve original exports. Restrict to essential fields and valid dates/coordinates.
cols=['id','observed_on','user_id','quality_grade','captive_cultivated',
      'latitude','longitude','place_county_name','scientific_name','url','image_url','flowering']
data=data[cols].copy()
data['observed_on']=pd.to_datetime(data.observed_on,errors='coerce')
data=data.dropna(subset=['observed_on','latitude','longitude','user_id'])
data=data[data.latitude.between(25,37) & data.longitude.between(-107,-93)]
# Do not silently choose labels if an observation appears in both exports.
conflicts=data.groupby('id').flowering.nunique()
if (conflicts>1).any():
    raise ValueError('Conflicting observation labels found; investigate IDs manually.')
data=data.drop_duplicates(subset='id').copy()
data['month']=data.observed_on.dt.month
data['day_of_year']=data.observed_on.dt.dayofyear
data['day_sin']=np.sin(2*np.pi*data.day_of_year/365.25)
data['day_cos']=np.cos(2*np.pi*data.day_of_year/365.25)

# Keep observations in a consistent order.
data = data.sort_values('id').reset_index(drop=True)

data.to_csv(ROOT/'redbud_cleaned_candidates.csv',index=False)
print('Cleaned candidate shape:',data.shape)
print('Month vs class:\n',pd.crosstab(data.month,data.flowering))

monthly=pd.crosstab(data.month,data.flowering).reindex(range(1,13),fill_value=0)
monthly.columns=['No flowers/fruits','Flowering']
ax=monthly.plot(kind='bar',figsize=(9,4))
ax.set(xlabel='Observation month',ylabel='Number of observations',title='iNaturalist redbud annotations by month')
plt.tight_layout()
plt.savefig(ROOT/'monthly_flowering_counts.png',dpi=160)
plt.close()

class MonthBaseline(ClassifierMixin, BaseEstimator):
    """Predict the majority class for each month."""

    def fit(self, X, y):
        months = np.asarray(X["month"])
        labels = np.asarray(y, dtype=int)

        self.classes_ = np.array([0, 1])
        self.default_class_ = int(labels.mean() >= 0.5)
        self.month_predictions_ = {}

        for month in range(1, 13):
            month_labels = labels[months == month]

            if len(month_labels) > 0:
                self.month_predictions_[month] = int(
                    month_labels.mean() >= 0.5
                )

        return self

    def predict(self, X):
        months = np.asarray(X["month"])

        return np.array([
            self.month_predictions_.get(
                int(month), self.default_class_
            )
            for month in months
        ])

    def predict_proba(self, X):
        predictions = self.predict(X)
        return np.column_stack([
            1 - predictions, predictions
        ]).astype(float)
    
# One untouched test split, fixed for every model and ablation.
# Stratification maintains class proportions. Observer leakage is a limitation;
# evaluate group-based splitting as an additional robustness experiment later.
train_idx,test_idx=train_test_split(np.arange(len(data)),test_size=0.2,
    stratify=data.flowering,random_state=42)
train=data.iloc[train_idx].copy(); test=data.iloc[test_idx].copy()
print('Development/test sizes:',len(train),len(test))

# Verify dataset order and split.
assert data['id'].is_monotonic_increasing
assert set(train.id).isdisjoint(set(test.id))
assert len(train) + len(test) == len(data)

print('Dataset order and split checks passed.')

# Save the fixed split for both team members.
train[['id']].to_csv(
    ROOT / 'development_ids.csv',
    index=False
)

test[['id']].to_csv(
    ROOT / 'test_ids.csv',
    index=False
)

print('Saved development and test observation IDs.')

features=['day_sin','day_cos','latitude','longitude']

X=train[features]
y=train.flowering
X_month = train[['month']]

scoring={'balanced_accuracy':'balanced_accuracy','f1':'f1','accuracy':'accuracy','roc_auc':'roc_auc'}

models = {
    'majority_baseline': (
        DummyClassifier(strategy='most_frequent'),
        X
    ),

    'month_baseline': (
        MonthBaseline(),
        X_month
    ),

    'logistic_regression': (
        Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),
        X
    )
}

seeds = [42, 123, 2026]

cv_results = []
for name, (model, X_model) in models.items():
    all_scores = {metric: [] for metric in scoring}

    for seed in seeds:
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed
        )

        scores = cross_validate(
            model,
            X_model,
            y,
            cv=cv,
            scoring=scoring
        )

        for metric in scoring:
            all_scores[metric].extend(
                scores['test_' + metric]
            )
        for fold in range(5):
            cv_results.append({
                'model': name,
                'seed': seed,
                'fold': fold + 1,
                'balanced_accuracy':
                    scores['test_balanced_accuracy'][fold],
                'f1': scores['test_f1'][fold],
                'accuracy': scores['test_accuracy'][fold],
                'roc_auc': scores['test_roc_auc'][fold]
            })

    print('\n', name)

    for metric, values in all_scores.items():
        values = np.array(values)

        print(
            f'  {metric}: '
            f'{values.mean():.3f} +/- '
            f'{values.std(ddof=1):.3f}'
        )

# Save all 15 validation results per model.
cv_df = pd.DataFrame(cv_results)

cv_df.to_csv(
    ROOT / 'cross_validation_results.csv',
    index=False
)

# Calculate the mean and standard deviation
# for each model and metric.
cv_summary = cv_df.groupby('model')[
    list(scoring.keys())
].agg(['mean', 'std'])

cv_summary.to_csv(
    ROOT / 'cross_validation_summary.csv'
)

print('\nSaved cross-validation results and summary.')

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
# Hyperparameter tuning is done ONLY on the development set.
search = GridSearchCV(models['logistic_regression'][0],
    {'model__C':[0.01,0.1,1,10,100], 'model__class_weight':[None,'balanced']},
    scoring='balanced_accuracy',cv=cv,n_jobs=-1,refit=True)

search.fit(X,y)
print('\nBest logistic regression settings:',search.best_params_)
print('Best development CV balanced accuracy:',round(search.best_score_,3))
# Do NOT evaluate on the held-out test set yet. Keep it untouched until
# decision tree, random forest and ablation choices are finalized.
print('Test set reserved; no test metrics calculated yet.')
