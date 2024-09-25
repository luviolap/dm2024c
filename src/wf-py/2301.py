#!/usr/bin/env python3

# Workflow final_models

import os
import sys
import gc
from typing import List, Dict, Any
import polars as pl
import yaml
from datetime import datetime
import numpy as np
from sympy import prime, randprime
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

def generate_primes(min_val: int, max_val: int) -> List[int]:
    return [prime(i) for i in range(prime(min_val), prime(max_val)) if min_val <= prime(i) <= max_val]

def grabar_importancia(modelo_final: lgb.Booster, modelo_rank: int, iteracion_bayesiana: int) -> None:
    tb_importancia = pl.DataFrame(lgb.importance(modelo_final))
    tb_importancia.write_csv(
        f"impo_{modelo_rank:02d}_{iteracion_bayesiana:03d}.txt",
        separator="\t"
    )

def main() -> None:
    print("z2301_FM_final_models_lightgbm.py START")
    
    action_initialize()
    
    # Generate seeds
    primes = generate_primes(100000, 1000000)
    np.random.seed(envg['PARAM']['semilla'])
    envg['PARAM']['semillas'] = np.random.choice(primes, envg['PARAM']['qsemillas'], replace=False).tolist()
    
    save_output()
    
    # Read Bayesian Optimization log
    arch_log = f"./{envg['PARAM']['input'][0]}/BO_log.txt"
    action_verify_file(arch_log)
    tb_log = pl.read_csv(arch_log, separator="\t")
    tb_log = tb_log.sort(envg['PARAM']['metrica'], descending=(envg['PARAM']['metrica_order'] == -1))
    
    # Read final training dataset
    arch_dataset = f"./{envg['PARAM']['input'][1]}/dataset_train_final.csv.gz"
    print("Reading dataset_train_final.csv.gz")
    action_verify_file(arch_dataset)
    dataset = pl.read_csv(arch_dataset)
    with open(f"./{envg['PARAM']['input'][1]}/dataset_metadata.yml", 'r') as file:
        envg['PARAM']['dataset_metadata'] = yaml.safe_load(file)
    
    campos_buenos = [col for col in dataset.columns if col not in [envg['PARAM']['dataset_metadata']['clase'], "clase01"]]
    
    dataset = dataset.with_columns(
        pl.when(pl.col(envg['PARAM']['dataset_metadata']['clase']).is_in(envg['PARAM']['train']['clase01_valor1']))
        .then(1)
        .otherwise(0)
        .alias("clase01")
    )
    
    if os.path.exists("tb_modelos.txt"):
        tb_modelos = pl.read_csv("tb_modelos.txt", separator="\t")
    else:
        tb_modelos = pl.DataFrame({
            "rank": [],
            "iteracion_bayesiana": [],
            "semilla": [],
            "isem": [],
            "archivo": []
        })
    
    with open("z-Rcanresume.txt", "a") as f:
        f.write(f"{datetime.now().strftime('%Y%m%d %H%M%S')}\n")
    
    imodelo = 0
    for modelo_rank in envg['PARAM']['modelos_rank']:
        imodelo += 1
        print(f"\nmodelo_rank: {modelo_rank}, semillas: ", end="")
        envg['OUTPUT']['status']['modelo_rank'] = modelo_rank
        
        parametros = tb_log[modelo_rank - 1].to_dict()
        iteracion_bayesiana = parametros['iteracion_bayesiana']
        
        print("Creating lgb.Dataset")
        dtrain = lgb.Dataset(
            data=dataset.select(campos_buenos).to_numpy(),
            label=dataset["clase01"].to_numpy(),
            free_raw_data=False
        )
        
        ganancia = parametros['ganancia']
        
        # Remove non-LightGBM parameters
        for param in ['experimento', 'cols', 'rows', 'fecha', 'estimulos', 'ganancia', 'metrica', 'iteracion_bayesiana']:
            parametros.pop(param, None)
        
        sem = 0
        for vsemilla in envg['PARAM']['semillas']:
            sem += 1
            print(f"{sem} ", end="")
            envg['OUTPUT']['status']['sem'] = sem
            save_output()
            
            parametros['seed'] = vsemilla
            
            nombre_raiz = f"{modelo_rank:02d}_{iteracion_bayesiana:03d}_s{parametros['seed']}"
            arch_modelo = f"modelo_{nombre_raiz}.model"
            
            if not os.path.exists(arch_modelo):
                print(f"\nTraining model = {sem}  .", end="")
                np.random.seed(parametros['seed'])
                modelo_final = lgb.train(
                    params=parametros,
                    train_set=dtrain,
                    verbose=-100
                )
                print(" ...End.")
                
                # Save the model
                modelo_final.save_model(arch_modelo)
                
                # Create and save variable importance for the first seed
                if sem == 1:
                    with open("z-Rcanresume.txt", "a") as f:
                        f.write(f"{datetime.now().strftime('%Y%m%d %H%M%S')}\n")
                    grabar_importancia(modelo_final, modelo_rank, iteracion_bayesiana)
                
                # Add to tb_modelos
                new_row = pl.DataFrame({
                    "rank": [modelo_rank],
                    "iteracion_bayesiana": [iteracion_bayesiana],
                    "semilla": [vsemilla],
                    "isem": [sem],
                    "archivo": [arch_modelo]
                })
                tb_modelos = pl.concat([tb_modelos, new_row])
                tb_modelos.write_csv("tb_modelos.txt", separator="\t")
    
    # Save metadata
    print("Saving metadata")
    with open("dataset_metadata.yml", "w") as file:
        yaml.dump(envg['PARAM']['dataset_metadata'], file)
    
    envg['OUTPUT']['time']['end'] = datetime.now().strftime("%Y%m%d %H%M%S")
    save_output()
    
    print("z2301_FM_final_models_lightgbm.py END")

if __name__ == "__main__":
    # Clean memory
    gc.collect()
    
    # Get command line arguments
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    
    main()