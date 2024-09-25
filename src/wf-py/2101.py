#!/usr/bin/env python3

# Workflow Training Strategy

import os
import sys
import gc
from typing import List, Dict, Any
import polars as pl
import yaml
from datetime import datetime
import numpy as np
from sympy import prime, randprime

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

def generate_primes(min_val: int, max_val: int) -> List[int]:
    return [prime(i) for i in range(prime(min_val), prime(max_val)) if min_val <= prime(i) <= max_val]

def main() -> None:
    print("z2101_TS_training_strategy.py START")
    
    action_initialize()
    
    # Generate seeds
    primes = generate_primes(100000, 1000000)
    np.random.seed(envg['PARAM']['semilla'])
    envg['PARAM']['semillas'] = np.random.choice(primes, 2, replace=False).tolist()
    
    envg['PARAM']['train']['semilla'] = envg['PARAM']['semillas'][0]
    envg['PARAM']['final_train']['semilla'] = envg['PARAM']['semillas'][1]
    
    # Load dataset
    envg['PARAM']['dataset'] = f"./{envg['PARAM']['input']}/dataset.csv.gz"
    with open(f"./{envg['PARAM']['input']}/dataset_metadata.yml", 'r') as file:
        envg['PARAM']['dataset_metadata'] = yaml.safe_load(file)
    
    print("Reading dataset")
    action_verify_file(envg['PARAM']['dataset'])
    print("Starting dataset reading")
    dataset = pl.read_csv(envg['PARAM']['dataset')
    print("Finished dataset reading")
    
    print("Sorting dataset")
    dataset = dataset.sort(envg['PARAM']['dataset_metadata']['primarykey'])
    
    archivos_salida = []
    
    if "future" in envg['PARAM']:
        print("Starting to save future dataset")
        future_data = dataset.filter(pl.col(envg['PARAM']['dataset_metadata']['periodo']).is_in(envg['PARAM']['future']))
        future_data.write_csv("dataset_future.csv.gz", compression="gzip")
        print("Finished saving future dataset")
        archivos_salida.append("dataset_future.csv.gz")
    
    if "final_train" in envg['PARAM']:
        print("Starting to save final train dataset")
        np.random.seed(envg['PARAM']['final_train']['semilla'])
        final_train_data = dataset.filter(pl.col(envg['PARAM']['dataset_metadata']['periodo']).is_in(envg['PARAM']['final_train']['training']))
        final_train_data = final_train_data.with_columns(pl.Series("azar", np.random.rand(len(final_train_data))))
        final_train_data = final_train_data.filter(
            (pl.col("azar") <= envg['PARAM']['final_train']['undersampling']) |
            (pl.col(envg['PARAM']['dataset_metadata']['clase']).is_in(envg['PARAM']['final_train']['clase_minoritaria']))
        )
        final_train_data = final_train_data.drop("azar")
        final_train_data.write_csv("dataset_train_final.csv.gz", compression="gzip")
        print("Finished saving final train dataset")
        archivos_salida.append("dataset_train_final.csv.gz")
    
    if "train" in envg['PARAM']:
        print("Starting to save training datasets")
        np.random.seed(envg['PARAM']['train']['semilla'])
        train_data = dataset.filter(pl.col(envg['PARAM']['dataset_metadata']['periodo']).is_in(envg['PARAM']['train']['training']))
        train_data = train_data.with_columns(pl.Series("azar", np.random.rand(len(train_data))))
        
        train_data = train_data.with_columns(
            pl.when((pl.col("azar") <= envg['PARAM']['train']['undersampling']) |
                    (pl.col(envg['PARAM']['dataset_metadata']['clase']).is_in(envg['PARAM']['train']['clase_minoritaria'])))
            .then(1)
            .otherwise(0)
            .alias("fold_train")
        )
        
        dataset = dataset.with_columns(
            pl.when(pl.col(envg['PARAM']['dataset_metadata']['periodo']).is_in(envg['PARAM']['train']['validation']))
            .then(1)
            .otherwise(0)
            .alias("fold_validate"),
            
            pl.when(pl.col(envg['PARAM']['dataset_metadata']['periodo']).is_in(envg['PARAM']['train']['testing']))
            .then(1)
            .otherwise(0)
            .alias("fold_test")
        )
        
        training_data = dataset.filter(pl.col("fold_train") + pl.col("fold_validate") + pl.col("fold_test") >= 1)
        training_data = training_data.drop("azar")
        training_data.write_csv("dataset_training.csv.gz", compression="gzip")
        print("Finished saving training datasets")
        archivos_salida.append("dataset_training.csv.gz")
    
    # Save metadata
    print("Saving metadata")
    with open("dataset_metadata.yml", "w") as file:
        yaml.dump(envg['PARAM']['dataset_metadata'], file)
    
    # Save field information
    tb_campos = pl.DataFrame({
        "pos": range(1, len(dataset.columns) + 1),
        "campo": dataset.columns,
        "tipo": [str(dataset[col].dtype) for col in dataset.columns],
        "nulos": dataset.filter(pl.col("fold_train") + pl.col("fold_validate") + pl.col("fold_test") >= 1).null_count().to_list(),
        "ceros": [(dataset.filter(pl.col("fold_train") + pl.col("fold_validate") + pl.col("fold_test") >= 1)[col] == 0).sum() for col in dataset.columns]
    })
    tb_campos.write_csv("dataset_training.campos.txt", separator="\t")
    
    # Update output information
    envg['OUTPUT']['dataset_train'] = {
        'ncol': len(dataset.filter(pl.col("fold_train") > 0).columns),
        'nrow': len(dataset.filter(pl.col("fold_train") > 0)),
        'periodos': dataset.filter(pl.col("fold_train") > 0)[envg['PARAM']['dataset_metadata']['periodo']].n_unique()
    }
    envg['OUTPUT']['dataset_validate'] = {
        'ncol': len(dataset.filter(pl.col("fold_validate") > 0).columns),
        'nrow': len(dataset.filter(pl.col("fold_validate") > 0)),
        'periodos': dataset.filter(pl.col("fold_validate") > 0)[envg['PARAM']['dataset_metadata']['periodo']].n_unique()
    }
    envg['OUTPUT']['dataset_test'] = {
        'ncol': len(dataset.filter(pl.col("fold_test") > 0).columns),
        'nrow': len(dataset.filter(pl.col("fold_test") > 0)),
        'periodos': dataset.filter(pl.col("fold_test") > 0)[envg['PARAM']['dataset_metadata']['periodo']].n_unique()
    }
    envg['OUTPUT']['dataset_future'] = {
        'ncol': len(dataset.filter(pl.col(envg['PARAM']['dataset_metadata']['periodo']).is_in(envg['PARAM']['future'])).columns),
        'nrow': len(dataset.filter(pl.col(envg['PARAM']['dataset_metadata']['periodo']).is_in(envg['PARAM']['future']))),
        'periodos': dataset.filter(pl.col(envg['PARAM']['dataset_metadata']['periodo']).is_in(envg['PARAM']['future']))[envg['PARAM']['dataset_metadata']['periodo']].n_unique()
    }
    envg['OUTPUT']['dataset_finaltrain'] = {
        'ncol': len(dataset.filter(pl.col(envg['PARAM']['dataset_metadata']['periodo']).is_in(envg['PARAM']['final_train'])).columns),
        'nrow': len(dataset.filter(pl.col(envg['PARAM']['dataset_metadata']['periodo']).is_in(envg['PARAM']['final_train']))),
        'periodos': dataset.filter(pl.col(envg['PARAM']['dataset_metadata']['periodo']).is_in(envg['PARAM']['final_train']))[envg['PARAM']['dataset_metadata']['periodo']].n_unique()
    }
    
    envg['OUTPUT']['time']['end'] = datetime.now().strftime("%Y%m%d %H%M%S")
    save_output()
    
    # Finalize
    if all(os.path.exists(f) for f in archivos_salida):
        print("z2101_TS_training_strategy.py END")
    else:
        sys.exit("Missing output files")

if __name__ == "__main__":
    # Clean memory
    gc.collect()
    
    # Get command line arguments
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    
    main()