#!/usr/bin/env python3

# Workflow Feature Engineering intrames hojas de Random Forest

import os
import sys
import gc
from typing import List, Dict, Any
import polars as pl
import yaml
from datetime import datetime
import lightgbm as lgb
import numpy as np

def action_initialize() -> None:
    global envg
    envg: Dict[str, Any] = {
        'PARAM': {},
        'OUTPUT': {'dataset': {}, 'time': {}}
    }
    # Load parameters from YAML file
    with open('parameters.yml', 'r') as file:
        envg['PARAM'] = yaml.safe_load(file)

def action_verify_file(file_path: str) -> None:
    if not os.path.exists(file_path):
        sys.exit(f"File not found: {file_path}")

def save_output() -> None:
    with open('output.yml', 'w') as file:
        yaml.dump(envg['OUTPUT'], file)

def agrega_var_random_forest(dataset: pl.DataFrame) -> pl.DataFrame:
    print("inicio AgregaVarRandomForest()")
    gc.collect()

    clase_column = envg['PARAM']['dataset_metadata']['clase']
    periodo_column = envg['PARAM']['dataset_metadata']['periodo']

    dataset = dataset.with_columns(
        pl.when(pl.col(clase_column).is_in(envg['PARAM']['train']['clase01_valor1']))
        .then(1)
        .otherwise(0)
        .alias("clase01")
    )

    campos_buenos = [col for col in dataset.columns if col not in ["clase_ternaria", "clase01"]]

    dataset = dataset.with_columns(
        pl.when(pl.col(periodo_column).is_in(envg['PARAM']['train']['training']))
        .then(1)
        .otherwise(0)
        .alias("entrenamiento")
    )

    train_data = dataset.filter(pl.col("entrenamiento") == 1)
    X_train = train_data.select(campos_buenos).to_numpy()
    y_train = train_data.select("clase01").to_numpy().flatten()

    train_dataset = lgb.Dataset(X_train, label=y_train)

    modelo = lgb.train(
        params=envg['PARAM']['lgb_param'],
        train_set=train_dataset,
        verbose=-100
    )

    print("Fin construccion RandomForest")
    # Save the model
    modelo.save_model("modelo.model")

    qarbolitos = envg['PARAM']['lgb_param']['num_iterations']

    periodos = dataset[periodo_column].unique().sort().to_list()

    for periodo in periodos:
        print(f"periodo = {periodo}")
        periodo_data = dataset.filter(pl.col(periodo_column) == periodo)
        X_periodo = periodo_data.select(campos_buenos).to_numpy()

        print("Inicio prediccion")
        prediccion = modelo.predict(X_periodo, pred_leaf=True)
        print("Fin prediccion")

        for arbolito in range(qarbolitos):
            print(f"{arbolito} ", end="")
            hojas_arbol = np.unique(prediccion[:, arbolito])

            for nodo_id in hojas_arbol:
                column_name = f"rf_{arbolito:03d}_{nodo_id:03d}"
                new_column = (prediccion[:, arbolito] == nodo_id).astype(int)
                dataset = dataset.with_columns(
                    pl.when(pl.col(periodo_column) == periodo)
                    .then(pl.Series(new_column))
                    .otherwise(0)
                    .alias(column_name)
                )

        print()
        gc.collect()

    gc.collect()
    
    # Remove clase01 column
    dataset = dataset.drop("clase01")

    return dataset

def main() -> None:
    print("z1311_FE_rfatributes.py START")
    
    action_initialize()
    
    envg['PARAM']['lgb_param']['seed'] = envg['PARAM']['semilla']
    
    # Load dataset
    envg['PARAM']['dataset'] = f"./{envg['PARAM']['input']}/dataset.csv.gz"
    with open(f"./{envg['PARAM']['input']}/dataset_metadata.yml", 'r') as file:
        envg['PARAM']['dataset_metadata'] = yaml.safe_load(file)
    
    print("Reading dataset")
    action_verify_file(envg['PARAM']['dataset'])
    print("Starting dataset reading")
    dataset = pl.read_csv(envg['PARAM']['dataset'])
    print("Finished dataset reading")
    
    save_output()
    
    # Sort dataset
    print("Sorting dataset")
    dataset = dataset.sort(envg['PARAM']['dataset_metadata']['periodo'])
    
    # Add Random Forest variables
    envg['OUTPUT']['AgregaVarRandomForest'] = {}
    envg['OUTPUT']['AgregaVarRandomForest']['ncol_antes'] = len(dataset.columns)
    
    dataset = agrega_var_random_forest(dataset)
    
    envg['OUTPUT']['AgregaVarRandomForest']['ncol_despues'] = len(dataset.columns)
    save_output()
    gc.collect()
    
    # Sort dataset by primary key
    dataset = dataset.sort(envg['PARAM']['dataset_metadata']['primarykey'])
    
    # Save dataset
    print("Saving dataset")
    dataset.write_csv("dataset.csv.gz", compression="gzip")
    print("Finished saving dataset")
    
    # Save metadata
    print("Saving metadata")
    with open("dataset_metadata.yml", "w") as file:
        yaml.dump(envg['PARAM']['dataset_metadata'], file)
    
    # Save field information
    tb_campos = pl.DataFrame({
        "pos": range(1, len(dataset.columns) + 1),
        "campo": dataset.columns,
        "tipo": [str(dataset[col].dtype) for col in dataset.columns],
        "nulos": dataset.null_count().to_list(),
        "ceros": [(dataset[col] == 0).sum() for col in dataset.columns]
    })
    tb_campos.write_csv("dataset.campos.txt", separator="\t")
    
    # Update output information
    envg['OUTPUT']['dataset'] = {
        'ncol': len(dataset.columns),
        'nrow': len(dataset)
    }
    envg['OUTPUT']['time']['end'] = datetime.now().strftime("%Y%m%d %H%M%S")
    save_output()
    
    # Finalize
    if all(os.path.exists(f) for f in ["dataset.csv.gz", "dataset_metadata.yml"]):
        print("z1311_FE_rfatributes.py END")
    else:
        sys.exit("Missing output files")

if __name__ == "__main__":
    # Clean memory
    gc.collect()
    
    # Get command line arguments
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    
    main()