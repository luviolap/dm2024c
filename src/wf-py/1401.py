#!/usr/bin/env python3

# Workflow Data Drifting repair

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

# Financial indices
vfoto_mes = [
    201901, 201902, 201903, 201904, 201905, 201906,
    201907, 201908, 201909, 201910, 201911, 201912,
    202001, 202002, 202003, 202004, 202005, 202006,
    202007, 202008, 202009, 202010, 202011, 202012,
    202101, 202102, 202103, 202104, 202105, 202106,
    202107, 202108, 202109
]

vIPC = [1.9903030878, 1.9174403544, 1.8296186587, 1.7728862972, 1.7212488323, 1.6776304408,
        1.6431248196, 1.5814483345, 1.4947526791, 1.4484037589, 1.3913580777, 1.3404220402,
        1.3154288912, 1.2921698342, 1.2472681797, 1.2300475145, 1.2118694724, 1.1881073259,
        1.1693969743, 1.1375456949, 1.1065619600, 1.0681100000, 1.0370000000, 1.0000000000,
        0.9680542110, 0.9344152616, 0.8882274350, 0.8532444140, 0.8251880213, 0.8003763543,
        0.7763107219, 0.7566381305, 0.7289384687]

vdolar_blue = [39.045455, 38.402500, 41.639474, 44.274737, 46.095455, 45.063333,
               43.983333, 54.842857, 61.059524, 65.545455, 66.750000, 72.368421,
               77.477273, 78.191667, 82.434211, 101.087500, 126.236842, 125.857143,
               130.782609, 133.400000, 137.954545, 170.619048, 160.400000, 153.052632,
               157.900000, 149.380952, 143.615385, 146.250000, 153.550000, 162.000000,
               178.478261, 180.878788, 184.357143]

vdolar_oficial = [38.430000, 39.428000, 42.542105, 44.354211, 46.088636, 44.955000,
                  43.751429, 54.650476, 58.790000, 61.403182, 63.012632, 63.011579,
                  62.983636, 63.580556, 65.200000, 67.872000, 70.047895, 72.520952,
                  75.324286, 77.488500, 79.430909, 83.134762, 85.484737, 88.181667,
                  91.474000, 93.997778, 96.635909, 98.526000, 99.613158, 100.619048,
                  101.619048, 102.569048, 103.781818]

vUVA = [2.001408838932958, 1.950325472789153, 1.89323032351521, 1.8247220405493787, 1.746027787673673, 1.6871348409529485,
        1.6361678865622313, 1.5927529755859773, 1.5549162794128493, 1.4949100586391746, 1.4197729500774545, 1.3678188186372326,
        1.3136508617223726, 1.2690535173062818, 1.2381595983200178, 1.211656735577568, 1.1770808941405335, 1.1570338657445522,
        1.1388769475653255, 1.1156993751209352, 1.093638313080772, 1.0657171590878205, 1.0362173587708712, 1.0,
        0.9669867858358365, 0.9323750098728378, 0.8958202912590305, 0.8631993702994263, 0.8253893405524657, 0.7928918905364516,
        0.7666323845128089, 0.7428976357662823, 0.721615762047849]

def drift_UVA(dataset: pl.DataFrame, campos_monetarios: List[str], tb_indices: pl.DataFrame) -> pl.DataFrame:
    print("inicio drift_UVA()")
    periodo_column = envg['PARAM']['dataset_metadata']['periodo']
    for campo in campos_monetarios:
        dataset = dataset.join(
            tb_indices.select([periodo_column, "UVA"]),
            on=periodo_column,
            how="left"
        ).with_columns(
            (pl.col(campo) * pl.col("UVA")).alias(campo)
        ).drop("UVA")
    print("fin drift_UVA()")
    return dataset

def drift_dolar_oficial(dataset: pl.DataFrame, campos_monetarios: List[str], tb_indices: pl.DataFrame) -> pl.DataFrame:
    print("inicio drift_dolar_oficial()")
    periodo_column = envg['PARAM']['dataset_metadata']['periodo']
    for campo in campos_monetarios:
        dataset = dataset.join(
            tb_indices.select([periodo_column, "dolar_oficial"]),
            on=periodo_column,
            how="left"
        ).with_columns(
            (pl.col(campo) / pl.col("dolar_oficial")).alias(campo)
        ).drop("dolar_oficial")
    print("fin drift_dolar_oficial()")
    return dataset

def drift_dolar_blue(dataset: pl.DataFrame, campos_monetarios: List[str], tb_indices: pl.DataFrame) -> pl.DataFrame:
    print("inicio drift_dolar_blue()")
    periodo_column = envg['PARAM']['dataset_metadata']['periodo']
    for campo in campos_monetarios:
        dataset = dataset.join(
            tb_indices.select([periodo_column, "dolar_blue"]),
            on=periodo_column,
            how="left"
        ).with_columns(
            (pl.col(campo) / pl.col("dolar_blue")).alias(campo)
        ).drop("dolar_blue")
    print("fin drift_dolar_blue()")
    return dataset

def drift_deflacion(dataset: pl.DataFrame, campos_monetarios: List[str], tb_indices: pl.DataFrame) -> pl.DataFrame:
    print("inicio drift_deflacion()")
    periodo_column = envg['PARAM']['dataset_metadata']['periodo']
    for campo in campos_monetarios:
        dataset = dataset.join(
            tb_indices.select([periodo_column, "IPC"]),
            on=periodo_column,
            how="left"
        ).with_columns(
            (pl.col(campo) * pl.col("IPC")).alias(campo)
        ).drop("IPC")
    print("fin drift_deflacion()")
    return dataset

def drift_rank_simple(dataset: pl.DataFrame, campos_drift: List[str]) -> pl.DataFrame:
    print("inicio drift_rank_simple()")
    periodo_column = envg['PARAM']['dataset_metadata']['periodo']
    for campo in campos_drift:
        print(campo, end=" ")
        dataset = dataset.with_columns(
            pl.col(campo).rank(method="random").over(periodo_column).sub(1).div(
                pl.col(campo).count().over(periodo_column).sub(1)
            ).alias(f"{campo}_rank")
        ).drop(campo)
    print("\nfin drift_rank_simple()")
    return dataset

def drift_rank_cero_fijo(dataset: pl.DataFrame, campos_drift: List[str]) -> pl.DataFrame:
    print("inicio drift_rank_cero_fijo()")
    periodo_column = envg['PARAM']['dataset_metadata']['periodo']
    for campo in campos_drift:
        print(campo, end=" ")
        dataset = dataset.with_columns(
            pl.when(pl.col(campo) == 0).then(0)
            .when(pl.col(campo) > 0).then(
                pl.col(campo).rank(method="random").over([periodo_column, pl.col(campo).gt(0)]).div(
                    pl.col(campo).filter(pl.col(campo) > 0).count().over(periodo_column)
                )
            )
            .when(pl.col(campo) < 0).then(
                -pl.col(campo).rank(method="random").over([periodo_column, pl.col(campo).lt(0)]).div(
                    pl.col(campo).filter(pl.col(campo) < 0).count().over(periodo_column)
                )
            )
            .alias(f"{campo}_rank")
        ).drop(campo)
    print("\nfin drift_rank_cero_fijo()")
    return dataset

def drift_estandarizar(dataset: pl.DataFrame, campos_drift: List[str]) -> pl.DataFrame:
    print("inicio drift_estandarizar()")
    periodo_column = envg['PARAM']['dataset_metadata']['periodo']
    for campo in campos_drift:
        print(campo, end=" ")
        dataset = dataset.with_columns(
            ((pl.col(campo) - pl.col(campo).mean().over(periodo_column)) /
             pl.col(campo).std().over(periodo_column)).alias(f"{campo}_normal")
        ).drop(campo)
    print("\nfin drift_estandarizar()")
    return dataset

def main() -> None:
    print("z1401_DR_corregir_drifting.py START")
    
    action_initialize()
    
    # Load dataset
    envg['PARAM']['dataset'] = f"./{envg['PARAM']['input']}/dataset.csv.gz"
    with open(f"./{envg['PARAM']['input']}/dataset_metadata.yml", 'r') as file:
        envg['PARAM']['dataset_metadata'] = yaml.safe_load(file)
    
    # Financial indices table
    tb_indices = pl.DataFrame({
        "IPC": vIPC,
        "dolar_blue": vdolar_blue,
        "dolar_oficial": vdolar_oficial,
        "UVA": vUVA,
        envg['PARAM']['dataset_metadata']['periodo']: vfoto_mes
    })
    
    print("Reading dataset")
    action_verify_file(envg['PARAM']['dataset'])
    print("Starting dataset reading")
    dataset = pl.read_csv(envg['PARAM']['dataset'])
    print("Finished dataset reading")
    
    save_output()
    
    # Sort dataset
    dataset = dataset.sort(envg['PARAM']['dataset_metadata']['primarykey'])
    
    # Monetary fields
    campos_monetarios = [col for col in dataset.columns if col.startswith(("m", "Visa_m", "Master_m", "vm_m"))]
    
    # Apply data drift correction method
    metodo = envg['PARAM']['metodo']
    if metodo == "ninguno":
        print("No hay correccion del data drifting")
    elif metodo == "rank_simple":
        dataset = drift_rank_simple(dataset, campos_monetarios)
    elif metodo == "rank_cero_fijo":
        dataset = drift_rank_cero_fijo(dataset, campos_monetarios)
    elif metodo == "deflacion":
        dataset = drift_deflacion(dataset, campos_monetarios, tb_indices)
    elif metodo == "dolar_blue":
        dataset = drift_dolar_blue(dataset, campos_monetarios, tb_indices)
    elif metodo == "dolar_oficial":
        dataset = drift_dolar_oficial(dataset, campos_monetarios, tb_indices)
    elif metodo == "UVA":
        dataset = drift_UVA(dataset, campos_monetarios, tb_indices)
    elif metodo == "estandarizar":
        dataset = drift_estandarizar(dataset, campos_monetarios)
    
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
        print("z1401_DR_corregir_drifting.py END")
    else:
        sys.exit("Missing output files")

if __name__ == "__main__":
    # Clean memory
    gc.collect()
    
    # Get command line arguments
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    
    main()