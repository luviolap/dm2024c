from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import yaml
import os
import gc

# Clean memory
gc.collect()

# Load required libraries (pandas and numpy are Python equivalents for data.table)
# rpart is replaced by sklearn's DecisionTreeClassifier
# yaml is available in Python

# Experiment parameters
PARAM: Dict[str, Any] = {
    "experimento": 3610,
    "rpart": pd.DataFrame({
        "cp": [-1],
        "min_samples_split": [77],
        "min_samples_leaf": [23],
        "max_depth": [7]
    }),
    "feature_fraction": 0.5,
    "num_trees_max": 512
}

# Set working directory
os.chdir("~/buckets/b1/")

# Load environment
with open("~/buckets/b1/miAmbiente.yml", 'r') as file:
    mi_ambiente = yaml.safe_load(file)

# Load data
dataset = pd.read_csv(mi_ambiente['dataset_pequeno'])

# Create experiment folder
os.makedirs(f"./exp/KA{PARAM['experimento']}", exist_ok=True)
os.chdir(f"./exp/KA{PARAM['experimento']}")

# Sizes of ensemble to save
grabar: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Define train and test datasets
dtrain = dataset[dataset['foto_mes'] == 202107]
dapply = dataset[dataset['foto_mes'] == 202109]

# Clear clase_ternaria in test set
dapply['clase_ternaria'] = np.nan

# Remove unused data
del dataset
gc.collect()

# Define usable fields for prediction
campos_buenos = [col for col in dtrain.columns if col != "clase_ternaria"]

# Generate outputs
for icorrida in range(len(PARAM['rpart'])):
    print(f"Corrida {icorrida} ; ", end="")

    # Initialize accumulated probability
    dapply['prob_acumulada'] = 0

    # Get rpart parameters for this run
    param_rpart = PARAM['rpart'].iloc[icorrida]

    np.random.seed(mi_ambiente['semilla_primigenia'])

    for arbolito in range(1, PARAM['num_trees_max'] + 1):
        qty_campos_a_utilizar = int(len(campos_buenos) * PARAM['feature_fraction'])
        campos_random = np.random.choice(campos_buenos, qty_campos_a_utilizar, replace=False)

        # Create and train the decision tree
        modelo = DecisionTreeClassifier(
            random_state=mi_ambiente['semilla_primigenia'],
            max_depth=param_rpart['max_depth'],
            min_samples_split=param_rpart['min_samples_split'],
            min_samples_leaf=param_rpart['min_samples_leaf'],
            ccp_alpha=-param_rpart['cp']  # Note: sklearn uses ccp_alpha instead of cp
        )
        
        X_train = dtrain[campos_random]
        y_train = dtrain['clase_ternaria']
        modelo.fit(X_train, y_train)

        # Apply the model to the test data
        X_apply = dapply[campos_random]
        prediccion = modelo.predict_proba(X_apply)
        dapply['prob_acumulada'] += prediccion[:, 1]  # Assuming "BAJA+2" is the positive class

        if arbolito in grabar:
            # Generate Kaggle submission
            umbral_corte = (1 / 40) * arbolito
            entrega = pd.DataFrame({
                "numero_de_cliente": dapply['numero_de_cliente'],
                "Predicted": (dapply['prob_acumulada'] > umbral_corte).astype(int)
            })

            nom_arch_kaggle = f"KA{PARAM['experimento']}_{icorrida}_{arbolito:03d}.csv"

            # Save the file
            entrega.to_csv(nom_arch_kaggle, index=False)

            # Prepare for Kaggle submit
            comentario = (f"'trees={arbolito} cp={param_rpart['cp']} "
                          f"min_samples_split={param_rpart['min_samples_split']} "
                          f"min_samples_leaf={param_rpart['min_samples_leaf']} "
                          f"max_depth={param_rpart['max_depth']}'")

            comando = (f"~/install/proc_kaggle_submit.sh "
                       f"TRUE {mi_ambiente['modalidad']} {nom_arch_kaggle} {comentario}")

            # Run the command and capture output
            import subprocess
            ganancia = subprocess.check_output(comando, shell=True, text=True).strip()

            with open("tb_ganancias.txt", "a") as f:
                f.write(f"{ganancia}\t{nom_arch_kaggle}\n")

        print(f"{arbolito} ", end="")
    print()

# Copy results
os.system("~/install/repobrutalcopy.sh")

# Shut down the virtual machine
os.system("~/install/apagar-vm.sh")