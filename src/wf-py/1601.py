#!/usr/bin/env python3

# Workflow Canaritos Asesinos

import os
import sys
import gc
from typing import List, Dict, Any
import polars as pl
import yaml
from datetime import datetime
import numpy as np
import lightgbm as lgb

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

VPOS_CORTE = []

def fganancia_lgbm_meseta(preds, train_data):
    global VPOS_CORTE
    labels = train_data.get_label()
    weights = train_data.get_weight()

    df = pl.DataFrame({
        "prob": preds,
        "gan": np.where((labels == 1) & (weights > 1), envg['PARAM']['train']['gan1'], envg['PARAM']['train']['gan0'])
    })

    df = df.sort("prob", descending=True)
    df = df.with_columns(pl.arange(1, len(df) + 1).alias("posicion"))
    df = df.with_columns(pl.col("gan").cumsum().alias("gan_acum"))
    df = df.sort("gan_acum", descending=True)

    gan = df["gan_acum"][:500].mean()
    pos_meseta = int(df["posicion"][:500].median())
    VPOS_CORTE.append(pos_meseta)

    return "ganancia", gan, True

GVEZ = 1

def CanaritosAsesinos(dataset: pl.DataFrame, canaritos_ratio: float, canaritos_desvios: float, canaritos_semilla: int) -> pl.DataFrame:
    global GVEZ
    print("inicio CanaritosAsesinos()")
    gc.collect()

    clase_column = envg['PARAM']['dataset_metadata']['clase']
    periodo_column = envg['PARAM']['dataset_metadata']['periodo']

    dataset = dataset.with_columns(
        pl.when(pl.col(clase_column).is_in(envg['PARAM']['train']['clase01_valor1']))
        .then(1)
        .otherwise(0)
        .alias("clase01")
    )

    np.random.seed(canaritos_semilla)
    for i in range(1, int(dataset.shape[1] * canaritos_ratio) + 1):
        dataset = dataset.with_columns(pl.Series(f"canarito{i}", np.random.rand(len(dataset))))

    campos_buenos = [col for col in dataset.columns if col not in campitos + ["clase01"]]

    azar = np.random.rand(len(dataset))

    dataset = dataset.with_columns(
        pl.when((pl.col(periodo_column).is_in(envg['PARAM']['train']['training'])) &
                ((pl.col("clase01") == 1) | (pl.Series(azar) < envg['PARAM']['train']['undersampling'])))
        .then(1)
        .otherwise(0)
        .alias("entrenamiento")
    )

    train_data = dataset.filter(pl.col("entrenamiento") == 1)
    X_train = train_data.select(campos_buenos).to_numpy()
    y_train = train_data["clase01"].to_numpy()
    w_train = np.where(train_data[clase_column].is_in(envg['PARAM']['train']['positivos']), 1.0000001, 1.0)

    valid_data = dataset.filter(pl.col(periodo_column).is_in(envg['PARAM']['train']['validation']))
    X_valid = valid_data.select(campos_buenos).to_numpy()
    y_valid = valid_data["clase01"].to_numpy()
    w_valid = np.where(valid_data[clase_column].is_in(envg['PARAM']['train']['positivos']), 1.0000001, 1.0)

    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train, free_raw_data=False)
    dvalid = lgb.Dataset(X_valid, label=y_valid, weight=w_valid, free_raw_data=False)

    param = {
        "objective": "binary",
        "metric": "custom",
        "first_metric_only": True,
        "boost_from_average": True,
        "feature_pre_filter": False,
        "verbosity": -100,
        "seed": canaritos_semilla,
        "max_depth": -1,
        "min_gain_to_split": 0.0,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "max_bin": 31,
        "num_iterations": 9999,
        "force_row_wise": True,
        "learning_rate": 0.065,
        "feature_fraction": 1.0,
        "min_data_in_leaf": 260,
        "num_leaves": 60,
        "early_stopping_rounds": 200,
        "num_threads": 1
    }

    np.random.seed(canaritos_semilla)
    modelo = lgb.train(param, dtrain, valid_sets=[dvalid], feval=fganancia_lgbm_meseta, verbose_eval=False)

    importancia = modelo.feature_importance(importance_type='gain')
    tb_importancia = pl.DataFrame({
        "Feature": campos_buenos,
        "Importance": importancia,
        "pos": pl.arange(1, len(campos_buenos) + 1)
    }).sort("Importance", descending=True)

    tb_importancia.write_csv(f"impo_{GVEZ}.txt", separator="\t")
    GVEZ += 1

    umbral = tb_importancia.filter(pl.col("Feature").str.contains("canarito")).select(
        (pl.col("pos").median() + canaritos_desvios * pl.col("pos").std())
    ).item()

    col_utiles = tb_importancia.filter((pl.col("pos") < umbral) & (~pl.col("Feature").str.contains("canarito")))["Feature"].to_list()
    col_utiles = list(set(col_utiles + campitos + ["mes"]))

    col_inutiles = [col for col in dataset.columns if col not in col_utiles]
    dataset = dataset.drop(col_inutiles)

    print("fin CanaritosAsesinos()")
    return dataset

def main() -> None:
    print("z1601_CN_canaritos_asesinos.py START")
    
    action_initialize()
    
    envg['PARAM']['CanaritosAsesinos']['semilla'] = envg['PARAM']['semilla']
    
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
    
    # Define important columns
    global campitos
    campitos = list(set([
        envg['PARAM']['dataset_metadata']['primarykey'],
        envg['PARAM']['dataset_metadata']['entity_id'],
        envg['PARAM']['dataset_metadata']['periodo'],
        envg['PARAM']['dataset_metadata']['clase']
    ]))

    cols_lagueables = [col for col in dataset.columns if col not in envg['PARAM']['dataset_metadata']]
    
    # Sort dataset
    print("Sorting dataset")
    dataset = dataset.sort(envg['PARAM']['dataset_metadata']['primarykey'])
    
    # Feature selection using Killer Canaries
    envg['OUTPUT']['CanaritosAsesinos'] = {'ncol_antes': len(dataset.columns)}
    
    dataset = CanaritosAsesinos(
        dataset,
        canaritos_ratio=envg['PARAM']['CanaritosAsesinos']['ratio'],
        canaritos_desvios=envg['PARAM']['CanaritosAsesinos']['desvios'],
        canaritos_semilla=envg['PARAM']['CanaritosAsesinos']['semilla']
    )
    
    envg['OUTPUT']['CanaritosAsesinos']['ncol_despues'] = len(dataset.columns)
    save_output()
    
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
        print("z1601_CN_canaritos_asesinos.py END")
    else:
        sys.exit("Missing output files")

if __name__ == "__main__":
    # Clean memory
    gc.collect()
    
    # Get command line arguments
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    
    main()