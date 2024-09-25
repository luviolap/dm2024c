#!/usr/bin/env python3
# Workflow Incorporate Dataset

import os
import sys
import gc
from typing import Dict, List, Any
import polars as pl
import yaml
from datetime import datetime

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

def action_abort(message: str) -> None:
    sys.exit(f"Error: {message}")

def save_output() -> None:
    with open('output.yml', 'w') as file:
        yaml.dump(envg['OUTPUT'], file)

def main() -> None:
    print("z1101_DT_incorporar_dataset.py START")
    
    # Initialize environment
    action_initialize()
    
    # Load dataset
    print("Reading dataset")
    arch: str = envg['PARAM']['archivo']
    action_verify_file(arch)
    print("Starting file reading")
    dataset: pl.DataFrame = pl.read_csv(arch)
    print("Finished file reading")
    
    # Verify field names
    print("Verifying field names")
    campos: List[str] = dataset.columns
    campitos: set = set(envg['PARAM']['primarykey'] + [envg['PARAM']['entity_id'], envg['PARAM']['periodo'], envg['PARAM']['clase']])
    for vcampo in campitos:
        if vcampo not in campos:
            action_abort(f"Field does not exist: {vcampo}")
    
    save_output()
    
    # Verify primary key
    print("Verifying primary key")
    pk_qty: int = dataset.select(envg['PARAM']['primarykey']).n_unique()
    if pk_qty != len(dataset):
        action_abort("Inconsistent Primary Key")
    
    # Sort dataset
    print("Sorting dataset")
    dataset = dataset.sort(envg['PARAM']['primarykey'])
    
    # Save dataset
    print("Saving dataset")
    print("Starting dataset save")
    dataset.write_csv("dataset.csv.gz", compression="gzip")
    print("Finished dataset save")
    
    # Save metadata
    print("Saving metadata")
    dataset_metadata: Dict[str, Any] = envg['PARAM'].copy()
    dataset_metadata.pop('archivo', None)
    dataset_metadata.pop('semilla', None)
    dataset_metadata['cols'] = dataset.columns
    with open("dataset_metadata.yml", "w") as file:
        yaml.dump(dataset_metadata, file)
    
    # Save field information
    tb_campos: pl.DataFrame = pl.DataFrame({
        "pos": range(1, len(dataset.columns) + 1),
        "campo": dataset.columns,
        "tipo": [str(dataset[col].dtype) for col in dataset.columns],
        "nulos": dataset.null_count().to_list(),
        "ceros": [(dataset[col] == 0).sum() for col in dataset.columns]
    })
    tb_campos.write_csv("dataset.campos.txt", separator="\t")
    
    # Update output information
    envg['OUTPUT']['dataset']['ncol'] = len(dataset.columns)
    envg['OUTPUT']['dataset']['nrow'] = len(dataset)
    envg['OUTPUT']['time']['end'] = datetime.now().strftime("%Y%m%d %H%M%S")
    save_output()
    
    # Finalize
    if all(os.path.exists(f) for f in ["dataset.csv.gz", "dataset_metadata.yml"]):
        print("z1101_DT_incorporar_dataset.py END")
    else:
        action_abort("Missing output files")

if __name__ == "__main__":
    # Clean memory
    gc.collect()
    
    # Get command line arguments
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    
    main()