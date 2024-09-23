import os
import yaml
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime
import wandb

# Set up directories
root_dir = os.path.expanduser("~/buckets/b1/")
exp_name = "HT4220-OPTUNA"
exp_dir = os.path.join(root_dir, "exp", exp_name)
os.makedirs(exp_dir, exist_ok=True)

# Set up logging
log_file = os.path.join(exp_dir, f"{exp_name}.txt")
log_best_file = os.path.join(exp_dir, f"{exp_name}_mejor.txt")

def log_message(message, file=log_file):
    timestamp = datetime.now().strftime("%Y%m%d %H%M%S")
    with open(file, "a") as f:
        f.write(f"{timestamp}\t{message}\n")

# Load configuration
with open(os.path.join(root_dir, "miAmbiente.yml"), 'r') as file:
    mi_ambiente = yaml.safe_load(file)

# Load data
dataset = pd.read_csv(mi_ambiente['dataset_pequeno'])
log_message(f"Loaded dataset from {mi_ambiente['dataset_pequeno']}")

# Prepare data for training
training_months = [202107]  # Adjust if needed
dataset['clase01'] = np.where(dataset['clase_ternaria'] == "CONTINUA", 0, 1)

train_data = dataset[dataset['foto_mes'].isin(training_months)]
features = [col for col in dataset.columns if col not in ['clase_ternaria', 'clase01', 'foto_mes']]

X_train = train_data[features]
y_train = train_data['clase01']

# Initialize wandb
wandb.init(project="HT4220-OPTUNA", name=exp_name, config={"exp_dir": exp_dir})

# Define the objective function
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    cv_results = lgb.cv(params, train_data, num_boost_round=1000, nfold=5, 
                        stratified=True, early_stopping_rounds=50, verbose_eval=False)
    
    avg_auc = np.mean(cv_results['auc-mean'])
    
    # Log each trial
    log_message(f"Trial {trial.number}: AUC = {avg_auc:.4f}, Params = {trial.params}")
    
    return avg_auc

# Set up the Optuna study with wandb callback
wandb_callback = WeightsAndBiasesCallback(metric_name="avg_auc", wandb_kwargs={"project": "HT4220-OPTUNA"})

study = optuna.create_study(direction='maximize', 
                            study_name=exp_name,
                            storage=f"sqlite:///{os.path.join(exp_dir, 'optuna.db')}",
                            load_if_exists=True)

study.optimize(objective, n_trials=100, callbacks=[wandb_callback])

# Log best results
best_trial = study.best_trial
log_message(f"Best trial: AUC = {best_trial.value:.4f}, Params = {best_trial.params}", log_best_file)
wandb.log({"best_auc": best_trial.value, "best_params": best_trial.params})

# Save Optuna visualizations
plt.figure(figsize=(12, 8))
optuna.visualization.plot_optimization_history(study)
plt.title('Optimization History')
plt.savefig(os.path.join(exp_dir, "optimization_history.png"))
wandb.log({"optimization_history": wandb.Image(os.path.join(exp_dir, "optimization_history.png"))})
plt.close()

plt.figure(figsize=(12, 8))
optuna.visualization.plot_param_importances(study)
plt.title('Parameter Importances')
plt.savefig(os.path.join(exp_dir, "param_importances.png"))
wandb.log({"param_importances": wandb.Image(os.path.join(exp_dir, "param_importances.png"))})
plt.close()

# Train the final model with the best parameters
best_params = study.best_params
final_model = lgb.train(best_params, lgb.Dataset(X_train, label=y_train))

# Save the final model
model_file = os.path.join(exp_dir, "best_lightgbm_model.txt")
final_model.save_model(model_file)
log_message(f"Best model saved to '{model_file}'", log_best_file)
wandb.save(model_file)

# Save study for later analysis
study_file = os.path.join(exp_dir, f"{exp_name}.pkl")
optuna.study.save_study(study, study_file)
log_message(f"Study saved to '{study_file}'", log_best_file)

print(f"Experiment completed. Results saved in {exp_dir}")

# Finish wandb run
wandb.finish()