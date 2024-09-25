#!/usr/bin/env python3

# Workflow Feature Engineering intrames manual artesanal

import os
import sys
import gc
from typing import List, Dict, Any
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

def save_output() -> None:
    with open('output.yml', 'w') as file:
        yaml.dump(envg['OUTPUT'], file)

def atributos_presentes(patributos: List[str], dataset: pl.DataFrame) -> bool:
    atributos = set(patributos)
    comun = atributos.intersection(set(dataset.columns))
    return len(atributos) == len(comun)

def agregar_variables_intrames(dataset: pl.DataFrame) -> pl.DataFrame:
    print("inicio AgregarVariables_IntraMes()")
    gc.collect()

    # INICIO de la seccion donde se deben hacer cambios con variables nuevas

    # el mes 1,2, ..12
    if atributos_presentes(["foto_mes"], dataset):
        dataset = dataset.with_columns(pl.col("foto_mes").mod(100).alias("kmes"))

    # creo un ctr_quarter que tenga en cuenta cuando los clientes hace 3 menos meses que estan
    if atributos_presentes(["ctrx_quarter"], dataset):
        dataset = dataset.with_columns(pl.col("ctrx_quarter").cast(pl.Float64).alias("ctrx_quarter_normalizado"))

    if atributos_presentes(["ctrx_quarter", "cliente_antiguedad"], dataset):
        dataset = dataset.with_columns(
            pl.when(pl.col("cliente_antiguedad") == 1)
            .then(pl.col("ctrx_quarter") * 5)
            .when(pl.col("cliente_antiguedad") == 2)
            .then(pl.col("ctrx_quarter") * 2)
            .when(pl.col("cliente_antiguedad") == 3)
            .then(pl.col("ctrx_quarter") * 1.2)
            .otherwise(pl.col("ctrx_quarter_normalizado"))
            .alias("ctrx_quarter_normalizado")
        )

    # variable extraida de una tesis de maestria de Irlanda
    if atributos_presentes(["mpayroll", "cliente_edad"], dataset):
        dataset = dataset.with_columns((pl.col("mpayroll") / pl.col("cliente_edad")).alias("mpayroll_sobre_edad"))

    # se crean los nuevos campos para MasterCard y Visa, teniendo en cuenta los NA's
    if atributos_presentes(["Master_status", "Visa_status"], dataset):
        dataset = dataset.with_columns([
            pl.max_horizontal("Master_status", "Visa_status").alias("vm_status01"),
            (pl.col("Master_status") + pl.col("Visa_status")).alias("vm_status02"),
            pl.max_horizontal(
                pl.when(pl.col("Master_status").is_null()).then(10).otherwise(pl.col("Master_status")),
                pl.when(pl.col("Visa_status").is_null()).then(10).otherwise(pl.col("Visa_status"))
            ).alias("vm_status03"),
            (pl.when(pl.col("Master_status").is_null()).then(10).otherwise(pl.col("Master_status")) +
             pl.when(pl.col("Visa_status").is_null()).then(10).otherwise(pl.col("Visa_status"))).alias("vm_status04"),
            (pl.when(pl.col("Master_status").is_null()).then(10).otherwise(pl.col("Master_status")) +
             100 * pl.when(pl.col("Visa_status").is_null()).then(10).otherwise(pl.col("Visa_status"))).alias("vm_status05"),
            pl.when(pl.col("Visa_status").is_null())
            .then(pl.when(pl.col("Master_status").is_null()).then(10).otherwise(pl.col("Master_status")))
            .otherwise(pl.col("Visa_status")).alias("vm_status06"),
            pl.when(pl.col("Master_status").is_null())
            .then(pl.when(pl.col("Visa_status").is_null()).then(10).otherwise(pl.col("Visa_status")))
            .otherwise(pl.col("Master_status")).alias("mv_status07")
        ])

    # combino MasterCard y Visa
    if atributos_presentes(["Master_mfinanciacion_limite", "Visa_mfinanciacion_limite"], dataset):
        dataset = dataset.with_columns(
            (pl.col("Master_mfinanciacion_limite") + pl.col("Visa_mfinanciacion_limite")).alias("vm_mfinanciacion_limite")
        )

    # ... (Continue with the rest of the variable combinations)

    # Aqui debe usted agregar sus propias nuevas variables

    # valvula de seguridad para evitar valores infinitos
    infinitos = dataset.select(pl.all().is_infinite().sum()).to_series().sum()
    if infinitos > 0:
        print(f"ATENCION, hay {infinitos} valores infinitos en tu dataset. Seran pasados a NA")
        dataset = dataset.with_columns(pl.all().map_elements(lambda x: None if x == float('inf') else x))

    # valvula de seguridad para evitar valores NaN que es 0/0
    nans = dataset.select(pl.all().is_nan().sum()).to_series().sum()
    if nans > 0:
        print(f"ATENCION, hay {nans} valores NaN 0/0 en tu dataset. Seran pasados arbitrariamente a 0")
        print("Si no te gusta la decision, modifica a gusto el programa!")
        dataset = dataset.with_columns(pl.all().map_elements(lambda x: 0 if x != x else x))

    print("fin AgregarVariables_IntraMes()")
    return dataset

def main() -> None:
    print("z1301_FE_intrames_manual.py START")
    
    action_initialize()
    
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
    
    # Add manual variables
    print("variables intra mes")
    dataset = agregar_variables_intrames(dataset)
    
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
    envg['OUTPUT']['dataset']['ncol'] = len(dataset.columns)
    envg['OUTPUT']['dataset']['nrow'] = len(dataset)
    envg['OUTPUT']['time']['end'] = datetime.now().strftime("%Y%m%d %H%M%S")
    save_output()
    
    # Finalize
    if all(os.path.exists(f) for f in ["dataset.csv.gz", "dataset_metadata.yml"]):
        print("z1301_FE_intrames_manual.py END")
    else:
        sys.exit("Missing output files")

if __name__ == "__main__":
    # Clean memory
    gc.collect()
    
    # Get command line arguments
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    
    main()