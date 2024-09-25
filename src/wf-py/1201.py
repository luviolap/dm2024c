#!/usr/bin/env python3

# Workflow Catastrophe Analysis

import os
import sys
import gc
from typing import Dict, List, Any, Tuple
import polars as pl
import yaml
from datetime import datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
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

def corregir_mice(dataset: pl.DataFrame, pcampo: str, pmeses: List[int]) -> pl.DataFrame:
    mask = dataset['foto_mes'].is_in(pmeses)
    imp = IterativeImputer(random_state=7, max_iter=5)
    data = dataset.select([pcampo]).to_numpy()
    imputed_data = imp.fit_transform(data)
    dataset = dataset.with_columns(
        pl.when(mask)
        .then(pl.Series(imputed_data.flatten()))
        .otherwise(pl.col(pcampo))
        .alias(pcampo)
    )
    return dataset

def corregir_interpolar(dataset: pl.DataFrame, pcampo: str, pmeses: List[int]) -> pl.DataFrame:
    entity_id: str = envg['PARAM']['dataset_metadata']['entity_id']
    tbl = dataset.select(
        [entity_id, 'foto_mes', pcampo]
    ).sort(entity_id, 'foto_mes')
    
    tbl = tbl.with_columns([
        pl.col(pcampo).shift(1).over(entity_id).alias('v1'),
        pl.col(pcampo).shift(-1).over(entity_id).alias('v2')
    ])
    
    tbl = tbl.with_columns(
        ((pl.col('v1') + pl.col('v2')) / 2).alias('promedio')
    )
    
    dataset = dataset.join(
        tbl.select([entity_id, 'foto_mes', 'promedio']),
        on=[entity_id, 'foto_mes'],
        how='left'
    )
    
    dataset = dataset.with_columns(
        pl.when(pl.col('foto_mes').is_in(pmeses))
        .then(pl.col('promedio'))
        .otherwise(pl.col(pcampo))
        .alias(pcampo)
    )
    
    dataset = dataset.drop('promedio')
    return dataset

def asignar_na_campomeses(dataset: pl.DataFrame, pcampo: str, pmeses: List[int]) -> pl.DataFrame:
    if pcampo in dataset.columns:
        dataset = dataset.with_columns(
            pl.when(pl.col('foto_mes').is_in(pmeses))
            .then(None)
            .otherwise(pl.col(pcampo))
            .alias(pcampo)
        )
    return dataset

def corregir_atributo(dataset: pl.DataFrame, pcampo: str, pmeses: List[int], pmetodo: str) -> Tuple[pl.DataFrame, int]:
    if pcampo not in dataset.columns:
        return dataset, 1
    
    if pmetodo == "MachineLearning":
        dataset = asignar_na_campomeses(dataset, pcampo, pmeses)
    elif pmetodo == "EstadisticaClasica":
        dataset = corregir_interpolar(dataset, pcampo, pmeses)
    elif pmetodo == "MICE":
        dataset = corregir_mice(dataset, pcampo, pmeses)
    
    return dataset, 0

def corregir_rotas(dataset: pl.DataFrame, pmetodo: str) -> pl.DataFrame:
    gc.collect()
    print("inicio Corregir_Rotas()")
    
    corrections: List[Tuple[str, List[int]]] = [
        ("active_quarter", [202006]),
        ("internet", [202006]),
        ("mrentabilidad", [201905, 201910, 202006]),
        ("mrentabilidad_annual", [201905, 201910, 202006]),
        ("mcomisiones", [201905, 201910, 202006]),
        ("mactivos_margen", [201905, 201910, 202006]),
        ("mpasivos_margen", [201905, 201910, 202006]),
        ("mcuentas_saldo", [202006]),
        ("ctarjeta_debito_transacciones", [202006]),
        ("mautoservicio", [202006]),
        ("ctarjeta_visa_transacciones", [202006]),
        ("mtarjeta_visa_consumo", [202006]),
        ("ctarjeta_master_transacciones", [202006]),
        ("mtarjeta_master_consumo", [202006]),
        ("ctarjeta_visa_debitos_automaticos", [201904]),
        ("mttarjeta_visa_debitos_automaticos", [201904]),
        ("ccajeros_propios_descuentos", [201910, 202002, 202006, 202009, 202010, 202102]),
        ("mcajeros_propios_descuentos", [201910, 202002, 202006, 202009, 202010, 202102]),
        ("ctarjeta_visa_descuentos", [201910, 202002, 202006, 202009, 202010, 202102]),
        ("mtarjeta_visa_descuentos", [201910, 202002, 202006, 202009, 202010, 202102]),
        ("ctarjeta_master_descuentos", [201910, 202002, 202006, 202009, 202010, 202102]),
        ("mtarjeta_master_descuentos", [201910, 202002, 202006, 202009, 202010, 202102]),
        ("ccomisiones_otras", [201905, 201910, 202006]),
        ("mcomisiones_otras", [201905, 201910, 202006]),
        ("cextraccion_autoservicio", [202006]),
        ("mextraccion_autoservicio", [202006]),
        ("ccheques_depositados", [202006]),
        ("mcheques_depositados", [202006]),
        ("ccheques_emitidos", [202006]),
        ("mcheques_emitidos", [202006]),
        ("ccheques_depositados_rechazados", [202006]),
        ("mcheques_depositados_rechazados", [202006]),
        ("ccheques_emitidos_rechazados", [202006]),
        ("mcheques_emitidos_rechazados", [202006]),
        ("tcallcenter", [202006]),
        ("ccallcenter_transacciones", [202006]),
        ("thomebanking", [202006]),
        ("chomebanking_transacciones", [201910, 202006]),
        ("ccajas_transacciones", [202006]),
        ("ccajas_consultas", [202006]),
        ("ccajas_depositos", [202006, 202105]),
        ("ccajas_extracciones", [202006]),
        ("ccajas_otras", [202006]),
        ("catm_trx", [202006]),
        ("matm", [202006]),
        ("catm_trx_other", [202006]),
        ("matm_other", [202006]),
    ]
    
    for pcampo, pmeses in corrections:
        dataset, _ = corregir_atributo(dataset, pcampo, pmeses, pmetodo)
    
    print("fin Corregir_rotas()")
    return dataset

def eliminar_atributo(dataset: pl.DataFrame, patributo: str) -> pl.DataFrame:
    if patributo in dataset.columns:
        dataset = dataset.drop(patributo)
    return dataset

def main() -> None:
    print("z1201_CA_reparar_dataset.py START")
    
    action_initialize()
    
    # Load dataset
    envg['PARAM']['dataset'] = f"./{envg['PARAM']['input']}/dataset.csv.gz"
    with open(f"./{envg['PARAM']['input']}/dataset_metadata.yml", 'r') as file:
        envg['PARAM']['dataset_metadata'] = yaml.safe_load(file)
    
    print("Reading dataset")
    action_verify_file(envg['PARAM']['dataset'])
    print("Starting dataset reading")
    dataset: pl.DataFrame = pl.read_csv(envg['PARAM']['dataset'])
    print("Finished dataset reading")
    
    # Remove attributes
    for atributo in envg['PARAM']['atributos_eliminar']:
        dataset = eliminar_atributo(dataset, atributo)
    
    save_output()
    
    # Sort dataset
    dataset = dataset.sort(envg['PARAM']['dataset_metadata']['primarykey'])
    
    # Apply correction method
    if envg['PARAM']['metodo'] in ["MachineLearning", "EstadisticaClasica", "MICE"]:
        dataset = corregir_rotas(dataset, envg['PARAM']['metodo'])
    
    # Save dataset
    print("Saving dataset")
    dataset.write_csv("dataset.csv.gz", compression="gzip")
    print("Finished saving dataset")
    
    # Save metadata
    print("Saving metadata")
    with open("dataset_metadata.yml", "w") as file:
        yaml.dump(envg['PARAM']['dataset_metadata'], file)
    
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
        print("z1201_CA_reparar_dataset.py END")
    else:
        sys.exit("Missing output files")

if __name__ == "__main__":
    # Clean memory
    gc.collect()
    
    # Get command line arguments
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    
    main()