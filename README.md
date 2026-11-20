# Texas Eastern Redbud Flowering Prediction

**CS 6375 Machine Learning  
Final Project**

**Team:** MLayalee  
**Members:** Gopika Murali and Ivin Issac Georgi

## Project Overview

This project investigates whether the flowering status of Eastern redbud (_Cercis canadensis_) observations in Texas can be predicted using observation date and geographic location.

We use iNaturalist observations annotated as either flowering (1) or no flowers or fruits (0). The current repository contains data preparation, exploratory data analysis, baseline models, logistic regression, feature ablation, and robustness experiments.

Decision tree, random forest, and final test-set evaluation components will be incorporated as the remaining team experiments are completed.

## Dataset

**Source:** iNaturalist (https://www.inaturalist.org/)

The dataset contains 1,012 observations:

| Class                | Observations |
| -------------------- | -----------: |
| Flowering            |          580 |
| No flowers or fruits |          432 |
| Total                |        1,012 |

The original iNaturalist CSV exports are included as ZIP files:

- `texas-easternredbuds-flowering.csv.zip`
- `texas-easternredbuds-nonflowering.csv.zip`

The exports were obtained for Eastern redbud observations in Texas using plant phenology annotations, with photographs included. The flowering and no-flowers/fruits annotations were exported separately.

The preprocessing script combines these exports, checks duplicate IDs and conflicting labels, validates required date and location information, and generates seasonal features.

The dataset is divided into:

- Development set: 809 observations
- Reserved test set: 203 observations

The split uses stratification and random seed 42. The saved observation IDs are available in `development_ids.csv` and `test_ids.csv`.

## Installation

Install Python and the required packages:

```bash
python -m pip install pandas numpy matplotlib scikit-learn geopandas
```

The improved Texas map also requires internet access to retrieve a US states GeoJSON boundary file from the PublicaMundi MappingAPI repository.

## Running the Experiments

Run the scripts in the following order:

```bash
python gopika_data_eda_baseline.py
python gopika_eda_visualizations.py
python gopika_data_quality.py
python gopika_logistic_ablation.py
python gopika_grouped_split.py
```

All scripts read files from their own directory and save their generated outputs in the same directory.

### Script Descriptions

| Script                         | Purpose                                                                                |
| ------------------------------ | -------------------------------------------------------------------------------------- |
| `gopika_data_eda_baseline.py`  | Data cleaning, fixed split, monthly counts, baselines, and initial logistic regression |
| `gopika_eda_visualizations.py` | Monthly flowering proportion and Texas geographic distribution figures                 |
| `gopika_data_quality.py`       | Data-quality filtering and logistic regression sensitivity analysis                    |
| `gopika_logistic_ablation.py`  | Seasonal-only versus seasonal-plus-geography logistic regression                       |
| `gopika_grouped_split.py`      | Standard versus observer-grouped cross-validation                                      |

## Experimental Setup

The main classification features are:

- Sine and cosine transformations of day of year
- Latitude
- Longitude

The month-based baseline uses observation month.

Experiments use five-fold cross-validation with random seeds 42, 123, and 2026. Logistic regression tuning considers regularization strengths of 0.01, 0.1, 1, 10, and 100, with and without balanced class weights.

Balanced accuracy is the primary metric. Other reported metrics include accuracy, precision, recall, F1 score, and ROC-AUC.

The feature-ablation, data-quality sensitivity, and grouped-split experiments use nested cross-validation for logistic regression hyperparameter selection.

The grouped-split experiment keeps observations from the same iNaturalist user together in both outer evaluation and inner tuning folds.

## Reproducing Figures

Run `gopika_data_eda_baseline.py` and `gopika_eda_visualizations.py` to generate:

- `monthly_flowering_counts.png`
- `monthly_flowering_proportion.png`
- `texas_redbud_distribution.png`
- `texas_redbud_distribution_improved.png`

## Reproducing Result Tables

The scripts generate:

| Output                          | Description                                                    |
| ------------------------------- | -------------------------------------------------------------- |
| `cross_validation_summary.csv`  | Majority, month-based, and initial logistic regression results |
| `logistic_ablation_summary.csv` | Logistic regression feature-ablation results                   |
| `data_quality_summary.csv`      | Development-set counts after quality filtering                 |
| `data_quality_cv_summary.csv`   | Logistic regression data-quality sensitivity results           |
| `grouped_split_summary.csv`     | Standard and observer-grouped validation results               |

The corresponding detailed CSV files contain individual fold results.

## Reproducibility Notes

The original development/test split is fixed using random seed 42. Repeated cross-validation uses seeds 42, 123, and 2026.

The reserved test set is not used for training, hyperparameter tuning, or the development experiments described above. Final test-set results will be added after model selection is complete.

## Team Contributions

**Gopika Murali:** Dataset collection and preprocessing, exploratory figures, majority and month-based baselines, logistic regression experiments, and logistic regression ablation and robustness checks.

**Ivin Issac Georgi:** Decision tree and random forest experiments, hyperparameter tuning and sensitivity analysis, and final test-set evaluation with bootstrap confidence intervals.

**Shared responsibilities:** Ablation analysis, grouped-split robustness evaluation, error analysis, final report, and presentation.

This contribution section will be updated to reflect the completed work of both members before final submission.

## References and Acknowledgments

- iNaturalist: https://www.inaturalist.org/
- PublicaMundi MappingAPI (Texas boundary visualization): https://github.com/PublicaMundi/MappingAPI

Additional references, software acknowledgments, and AI-tool usage disclosures will be finalized according to the course syllabus and project report requirements.
