# Comprehensive Data Science Workflow

This README provides an in-depth overview of our end-to-end data science workflow, which includes initial data analysis, data preparation, feature engineering, model training, evaluation, and submission. The workflow is implemented through a series of 13 Python scripts, each handling a specific part of the process.

## Workflow Diagram

```
                           ┌─────────────────┐
                           │  Start Workflow │
                           └────────┬────────┘
                                    │
                            ┌───────▼────────┐
                            │ 1. Analyze     │
                            │    Dataset     │
                            └───────┬────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
        ┌────────▼─────────┐ ┌──────▼─────────┐ ┌──────▼─────────┐
        │ 2. Incorporate   │ │ 3. Repair      │ │ 4. Data Drift  │
        │    Dataset       │ │    Dataset     │ │    Correction  │
        └────────┬─────────┘ └──────┬─────────┘ └──────┬─────────┘
                 │                  │                  │
                 └──────────────────┼──────────────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
        ┌────────▼─────────┐ ┌──────▼─────────┐ ┌──────▼─────────┐
        │ 5. Manual Feature│ │ 6. Historical  │ │ 7. Random      │
        │    Engineering   │ │    Feature Eng │ │    Forest Feat │
        └────────┬─────────┘ └──────┬─────────┘ └──────┬─────────┘
                 │                  │                  │
                 └──────────────────┼──────────────────┘
                                    │
                            ┌───────▼────────┐
                            │ 8. Killer      │
                            │    Canaries    │
                            └───────┬────────┘
                                    │
                            ┌───────▼────────┐
                            │ 9. Training    │
                            │    Strategy    │
                            └───────┬────────┘
                                    │
                            ┌───────▼────────┐
                            │ 10. Final      │
                            │     Models     │
                            └───────┬────────┘
                                    │
                            ┌───────▼────────┐
                            │ 11. Scoring    │
                            └───────┬────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
        ┌────────▼─────────┐ ┌──────▼─────────┐
        │ 12. Evaluate     │ │ 13. Evaluate   │
        │     with Class   │ │     Kaggle     │
        └────────┬─────────┘ └──────┬─────────┘
                 │                  │
                 └──────────────────┼──────────────────┘
                                    │
                           ┌────────▼────────┐
                           │   End Workflow  │
                           └─────────────────┘
```

## Detailed Script Descriptions

1. **Analyze Dataset (505.py)**:
   - Input: Raw dataset (CSV or similar)
   - Output: Various PDF plots, analysis results
   - Operations:
     - Calculates ratios of zeros and NAs for each field
     - Computes and visualizes averages for each field
     - Generates plots for data distribution and potential issues
   - Key Libraries: Polars, Matplotlib
   - Purpose: This initial analysis helps understand the dataset structure, identify potential issues, and inform subsequent data preparation steps.

2. **Incorporate Dataset (1101.py)**:
   - Input: Raw dataset (CSV or similar)
   - Output: Processed dataset (dataset.csv.gz), metadata (dataset_metadata.yml)
   - Operations:
     - Reads the initial dataset
     - Verifies field names and primary key consistency
     - Sorts the dataset by the primary key
     - Saves the processed dataset and metadata
   - Key Libraries: Polars, PyYAML

3. **Repair Dataset (1201.py)**:
   - Input: Processed dataset (dataset.csv.gz)
   - Output: Repaired dataset (dataset.csv.gz)
   - Operations:
     - Implements MICE (Multiple Imputation by Chained Equations) for missing data
     - Applies interpolation for specific fields and time periods
     - Corrects known data issues based on predefined rules
   - Key Libraries: Polars, Scikit-learn

4. **Data Drift Correction (1401.py)**:
   - Input: Repaired dataset (dataset.csv.gz)
   - Output: Drift-corrected dataset (dataset.csv.gz)
   - Operations:
     - Applies various drift correction methods (e.g., normalization, ranking)
     - Uses financial indices for monetary value corrections
     - Handles different drift correction strategies based on configuration
   - Key Libraries: Polars, NumPy

5. **Manual Feature Engineering (1301.py)**:
   - Input: Drift-corrected dataset (dataset.csv.gz)
   - Output: Dataset with new manual features (dataset.csv.gz)
   - Operations:
     - Creates new features based on domain knowledge
     - Combines information from multiple fields (e.g., Mastercard and Visa)
     - Implements custom calculations and transformations
   - Key Libraries: Polars

6. **Historical Feature Engineering (1501.py)**:
   - Input: Dataset with manual features (dataset.csv.gz)
   - Output: Dataset with historical features (dataset.csv.gz)
   - Operations:
     - Generates lag features for specified columns
     - Calculates trends and moving averages over different time windows
     - Uses Numba for optimized performance on historical calculations
   - Key Libraries: Polars, Numba, NumPy

7. **Random Forest Feature Engineering (1311.py)**:
   - Input: Dataset with historical features (dataset.csv.gz)
   - Output: Dataset with Random Forest features (dataset.csv.gz)
   - Operations:
     - Trains a LightGBM Random Forest model
     - Generates new features based on leaf node assignments
     - Creates features for each time period in the dataset
   - Key Libraries: Polars, LightGBM

8. **Killer Canaries (1601.py)**:
   - Input: Dataset with all engineered features (dataset.csv.gz)
   - Output: Dataset with selected features (dataset.csv.gz)
   - Operations:
     - Adds random "canary" features to the dataset
     - Trains a LightGBM model to assess feature importance
     - Removes features that are less important than the canary features
   - Key Libraries: Polars, LightGBM

9. **Training Strategy (2101.py)**:
   - Input: Dataset with selected features (dataset.csv.gz)
   - Output: Training, validation, test, and future datasets
   - Operations:
     - Splits the data into different sets based on time periods
     - Applies undersampling for imbalanced classes if specified
     - Prepares datasets for final model training and evaluation
   - Key Libraries: Polars, NumPy

10. **Final Models (2301.py)**:
    - Input: Training dataset
    - Output: Trained LightGBM models (.model files)
    - Operations:
      - Trains multiple LightGBM models using the best hyperparameters
      - Generates models with different random seeds for ensemble purposes
      - Saves the trained models and their feature importances
    - Key Libraries: Polars, LightGBM

11. **Scoring (2401.py)**:
    - Input: Future dataset, trained models
    - Output: Predictions file (tb_future_prediccion.txt)
    - Operations:
      - Loads the trained models
      - Applies each model to the future dataset
      - Generates and saves predictions for each model and seed combination
    - Key Libraries: Polars, LightGBM

12. **Evaluate with Class (2501.py)**:
    - Input: Predictions file, true class labels
    - Output: Evaluation metrics, gain plots (PDF)
    - Operations:
      - Calculates performance metrics for each model
      - Generates gain curves and identifies optimal cutoff points
      - Logs results to MLflow for experiment tracking
    - Key Libraries: Polars, Matplotlib, MLflow

13. **Evaluate Kaggle (2601.py)**:
    - Input: Predictions file
    - Output: Kaggle submission files, evaluation results
    - Operations:
      - Prepares submission files for Kaggle
      - Submits predictions to Kaggle competition (if configured)
      - Evaluates model performance based on Kaggle scores
      - Generates plots and logs results for different submission thresholds
    - Key Libraries: Polars, Matplotlib, MLflow, Kaggle API

## Workflow Execution

The scripts should be executed in the order presented above, starting with 505.py and then following the numerical order (1101, 1201, 1301, etc.). Each script relies on the output of the previous steps. The workflow is designed to be modular, allowing for easy modifications or additions to specific steps without affecting the entire pipeline.

[The rest of the README remains the same as in the previous version, including sections on Configuration, Dependencies, Output Files, MLflow Tracking, Kaggle Integration, Customization and Extension, and Conclusion.]