#!/usr/bin/env python3

# Workflow Feature Engineering historico

import os
import sys
import gc
from typing import List, Dict, Any
import polars as pl
import yaml
from datetime import datetime
import numpy as np
from numba import jit

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

@jit(nopython=True)
def fhistC(pcolumna, pdesde):
    n = len(pcolumna)
    out = np.zeros(5 * n)

    for i in range(n):
        # lag
        if pdesde[i] - 1 < i:
            out[i + 4 * n] = pcolumna[i - 1]
        else:
            out[i + 4 * n] = np.nan

        libre = 0
        xvalor = 1
        x = np.zeros(100)
        y = np.zeros(100)

        for j in range(pdesde[i] - 1, i + 1):
            a = pcolumna[j]
            if not np.isnan(a):
                y[libre] = a
                x[libre] = xvalor
                libre += 1
            xvalor += 1

        if libre > 1:
            xsum = np.sum(x[:libre])
            ysum = np.sum(y[:libre])
            xysum = np.sum(x[:libre] * y[:libre])
            xxsum = np.sum(x[:libre] ** 2)
            vmin = np.min(y[:libre])
            vmax = np.max(y[:libre])

            out[i] = (libre * xysum - xsum * ysum) / (libre * xxsum - xsum * xsum)
            out[i + n] = vmin
            out[i + 2 * n] = vmax
            out[i + 3 * n] = ysum / libre
        else:
            out[i] = np.nan
            out[i + n] = np.nan
            out[i + 2 * n] = np.nan
            out[i + 3 * n] = np.nan

    return out

def TendenciaYmuchomas(dataset: pl.DataFrame, cols: List[str], ventana: int = 6, tendencia: bool = True,
                       minimo: bool = True, maximo: bool = True, promedio: bool = True,
                       ratioavg: bool = False, ratiomax: bool = False) -> pl.DataFrame:
    gc.collect()
    ventana_regresion = ventana
    last = len(dataset)

    entity_id_col = envg['PARAM']['dataset_metadata']['entity_id']
    vector_ids = dataset[entity_id_col].to_numpy()

    vector_desde = np.arange(-ventana_regresion + 2, last - ventana_regresion + 2)
    vector_desde[:ventana_regresion] = 1

    for i in range(1, last):
        if vector_ids[i - 1] != vector_ids[i]:
            vector_desde[i] = i + 1
    
    for i in range(1, last):
        if vector_desde[i] < vector_desde[i - 1]:
            vector_desde[i] = vector_desde[i - 1]

    for campo in cols:
        nueva_col = fhistC(dataset[campo].to_numpy(), vector_desde)

        new_columns = []
        if tendencia:
            new_columns.append(pl.Series(name=f"{campo}_tend{ventana}", values=nueva_col[0:last]))
        if minimo:
            new_columns.append(pl.Series(name=f"{campo}_min{ventana}", values=nueva_col[last:2*last]))
        if maximo:
            new_columns.append(pl.Series(name=f"{campo}_max{ventana}", values=nueva_col[2*last:3*last]))
        if promedio:
            new_columns.append(pl.Series(name=f"{campo}_avg{ventana}", values=nueva_col[3*last:4*last]))
        if ratioavg:
            new_columns.append(pl.Series(name=f"{campo}_ratioavg{ventana}", values=dataset[campo] / nueva_col[3*last:4*last]))
        if ratiomax:
            new_columns.append(pl.Series(name=f"{campo}_ratiomax{ventana}", values=dataset[campo] / nueva_col[2*last:3*last]))

        dataset = dataset.with_columns(new_columns)

    return dataset

def main() -> None:
    print("z1501_FE_historia.py START")
    
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
    
    # Columns that can be lagged
    campitos = set(envg['PARAM']['dataset_metadata']['primarykey'] +
                   [envg['PARAM']['dataset_metadata']['entity_id'],
                    envg['PARAM']['dataset_metadata']['periodo'],
                    envg['PARAM']['dataset_metadata']['clase']])
    
    cols_lagueables = [col for col in dataset.columns if col not in campitos]
    
    # Sort dataset
    dataset = dataset.sort(envg['PARAM']['dataset_metadata']['primarykey'])
    
    # Lag 1
    if envg['PARAM']['lag1']:
        print("Inicio lag1")
        envg['OUTPUT']['lag1'] = {'ncol_antes': len(dataset.columns)}
        
        for col in cols_lagueables:
            dataset = dataset.with_columns([
                pl.col(col).shift(1).over(envg['PARAM']['dataset_metadata']['entity_id']).alias(f"{col}_lag1"),
                (pl.col(col) - pl.col(col).shift(1).over(envg['PARAM']['dataset_metadata']['entity_id'])).alias(f"{col}_delta1")
            ])
        
        envg['OUTPUT']['lag1']['ncol_despues'] = len(dataset.columns)
        save_output()
        print("Fin lag1")
    
    # Lag 2
    if envg['PARAM']['lag2']:
        print("Inicio lag2")
        envg['OUTPUT']['lag2'] = {'ncol_antes': len(dataset.columns)}
        
        for col in cols_lagueables:
            dataset = dataset.with_columns([
                pl.col(col).shift(2).over(envg['PARAM']['dataset_metadata']['entity_id']).alias(f"{col}_lag2"),
                (pl.col(col) - pl.col(col).shift(2).over(envg['PARAM']['dataset_metadata']['entity_id'])).alias(f"{col}_delta2")
            ])
        
        envg['OUTPUT']['lag2']['ncol_despues'] = len(dataset.columns)
        save_output()
        print("Fin lag2")
    
    # Lag 3
    if envg['PARAM']['lag3']:
        print("Inicio lag3")
        envg['OUTPUT']['lag3'] = {'ncol_antes': len(dataset.columns)}
        
        for col in cols_lagueables:
            dataset = dataset.with_columns([
                pl.col(col).shift(3).over(envg['PARAM']['dataset_metadata']['entity_id']).alias(f"{col}_lag3"),
                (pl.col(col) - pl.col(col).shift(3).over(envg['PARAM']['dataset_metadata']['entity_id'])).alias(f"{col}_delta3")
            ])
        
        envg['OUTPUT']['lag3']['ncol_despues'] = len(dataset.columns)
        save_output()
        print("Fin lag3")
    
    # Tendencias
    if envg['PARAM']['Tendencias1']['run']:
        envg['OUTPUT']['TendenciasYmuchomas1'] = {'ncol_antes': len(dataset.columns)}
        dataset = TendenciaYmuchomas(dataset,
                                     cols=cols_lagueables,
                                     ventana=envg['PARAM']['Tendencias1']['ventana'],
                                     tendencia=envg['PARAM']['Tendencias1']['tendencia'],
                                     minimo=envg['PARAM']['Tendencias1']['minimo'],
                                     maximo=envg['PARAM']['Tendencias1']['maximo'],
                                     promedio=envg['PARAM']['Tendencias1']['promedio'],
                                     ratioavg=envg['PARAM']['Tendencias1']['ratioavg'],
                                     ratiomax=envg['PARAM']['Tendencias1']['ratiomax'])
        envg['OUTPUT']['TendenciasYmuchomas1']['ncol_despues'] = len(dataset.columns)
        save_output()
    
    if envg['PARAM']['Tendencias2']['run']:
        envg['OUTPUT']['TendenciasYmuchomas2'] = {'ncol_antes': len(dataset.columns)}
        dataset = TendenciaYmuchomas(dataset,
                                     cols=cols_lagueables,
                                     ventana=envg['PARAM']['Tendencias2']['ventana'],
                                     tendencia=envg['PARAM']['Tendencias2']['tendencia'],
                                     minimo=envg['PARAM']['Tendencias2']['minimo'],
                                     maximo=envg['PARAM']['Tendencias2']['maximo'],
                                     promedio=envg['PARAM']['Tendencias2']['promedio'],
                                     ratioavg=envg['PARAM']['Tendencias2']['ratioavg'],
                                     ratiomax=envg['PARAM']['Tendencias2']['ratiomax'])
        envg['OUTPUT']['TendenciasYmuchomas2']['ncol_despues'] = len(dataset.columns)
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
        print("z1501_FE_historia.py END")
    else:
        sys.exit("Missing output files")

if __name__ == "__main__":
    # Clean memory
    gc.collect()
    
    # Get command line arguments
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    
    main()