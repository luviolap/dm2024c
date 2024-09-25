#!/usr/bin/env python3

# Workflow Scoring

import os
import sys
import gc
from typing import List, Dict, Any
import polars as pl
import yaml
from datetime import datetime
import lightgbm as lgb

def action_initialize() -> None:
    global envg
    envg: Dict[str, Any] = {
        'PARAM': {},
        'OUTPUT': {'status': {}, 'time': {}}
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

def main() -> None:
    print("z2401_SC_scoring.py START")
    
    action_initialize()
    save_output()
    
    # Read the model table
    arch_tb_modelos = f"./{envg['PARAM']['input'][0]}/tb_modelos.txt"
    action_verify_file(arch_tb_modelos)
    tb_modelos = pl.read_csv(arch_tb_modelos, separator="\t")
    
    # Read dataset metadata
    with open(f"./{envg['PARAM']['input'][1]}/dataset_metadata.yml", 'r') as file:
        envg['PARAM']['dataset_metadata'] = yaml.safe_load(file)
    
    # Read future dataset
    arch_future = f"./{envg['PARAM']['input'][1]}/dataset_future.csv.gz"
    print("Reading dataset_future.csv.gz")
    dfuture = pl.read_csv(arch_future)
    
    campos_buenos = [col for col in dfuture.columns if col not in [envg['PARAM']['dataset_metadata']['clase'], "clase01"]]
    
    if os.path.exists("tb_future_prediccion.txt"):
        tb_future_prediccion = pl.read_csv("tb_future_prediccion.txt", separator="\t")
    else:
        tb_future_prediccion = dfuture.select([
            envg['PARAM']['dataset_metadata']['primarykey'],
            envg['PARAM']['dataset_metadata']['clase']
        ])
    
    qpred = len(tb_modelos)
    dfuture_matriz = dfuture.select(campos_buenos).to_numpy()
    
    for ipred in range(qpred):
        mod = tb_modelos[ipred]
        print(f"\nmodelo_rank: {mod['rank']}, isem: {mod['isem']}")
        envg['OUTPUT']['status']['modelo_rank'] = mod['rank']
        envg['OUTPUT']['status']['modelo_isem'] = mod['isem']
        
        modelo_final = lgb.Booster(model_file=f"./{envg['PARAM']['input'][0]}/{mod['archivo']}")
        
        # Generate prediction, Scoring
        print("Creating prediction")
        prediccion = modelo_final.predict(dfuture_matriz)
        
        campo_pred = f"m_{mod['rank']}_{mod['isem']}"
        tb_future_prediccion = tb_future_prediccion.with_columns(pl.Series(name=campo_pred, values=prediccion))
        
        tb_modelos = tb_modelos.with_columns(
            pl.when(pl.col("rank") == mod['rank'])
            .when(pl.col("isem") == mod['isem'])
            .then(pl.lit(campo_pred))
            .otherwise(pl.col("campo"))
            .alias("campo")
        )
        
        tb_modelos = tb_modelos.with_columns(
            pl.when(pl.col("rank") == mod['rank'])
            .when(pl.col("isem") == mod['isem'])
            .then(pl.lit("tb_future_prediccion.txt"))
            .otherwise(pl.col("archivo_pred"))
            .alias("archivo_pred")
        )
        
        del prediccion
        del modelo_final
        gc.collect()
    
    tb_modelos.select(['rank', 'iteracion_bayesiana', 'isem', 'semilla', 'campo', 'archivo_pred']).write_csv("tb_predicciones.txt", separator="\t")
    tb_future_prediccion.write_csv("tb_future_prediccion.txt", separator="\t")
    
    # Save metadata
    print("Saving metadata")
    with open("dataset_metadata.yml", "w") as file:
        yaml.dump(envg['PARAM']['dataset_metadata'], file)
    
    envg['OUTPUT']['time']['end'] = datetime.now().strftime("%Y%m%d %H%M%S")
    save_output()
    
    print("z2401_SC_scoring.py END")

if __name__ == "__main__":
    # Clean memory
    gc.collect()
    
    # Get command line arguments
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    
    main()