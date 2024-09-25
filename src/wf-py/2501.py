#!/usr/bin/env python3

# Workflow EV_evaluate_conclase

import os
import sys
import gc
from typing import List, Dict, Any
import polars as pl
import yaml
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import mlflow

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

def mlog_table_hash(df: pl.DataFrame) -> str:
    return hashlib.md5(df.to_pandas().to_csv().encode()).hexdigest()

def mlog_log(data: Dict, arch: str, parentreplicate: bool = False) -> None:
    with open(arch, 'a') as f:
        f.write('\t'.join(map(str, data.values())) + '\n')
    
    if parentreplicate:
        mlflow.log_metrics(data)

def main() -> None:
    print("z2501_EV_evaluate_conclase.py START")
    
    action_initialize()
    save_output()
    
    # Read future predictions
    arch_future_prediccion = f"./{envg['PARAM']['input'][0]}/tb_future_prediccion.txt"
    action_verify_file(arch_future_prediccion)
    tb_future_prediccion = pl.read_csv(arch_future_prediccion, separator="\t")
    
    # Read predictions table
    arch_tb_predicciones = f"./{envg['PARAM']['input'][0]}/tb_predicciones.txt"
    action_verify_file(arch_tb_predicciones)
    tb_predicciones = pl.read_csv(arch_tb_predicciones, separator="\t")
    
    # Read dataset metadata
    with open(f"./{envg['PARAM']['input'][0]}/dataset_metadata.yml", 'r') as file:
        envg['PARAM']['dataset_metadata'] = yaml.safe_load(file)
    
    # Initialize MLFlow
    mlflow.set_experiment(f"/futu-{mlog_table_hash(tb_future_prediccion.select(envg['PARAM']['dataset_metadata']['primarykey']))}")
    mlflow.start_run(run_name=envg['PARAM']['experimento'])
    
    tb_ganancias = pl.DataFrame({"envios": range(1, len(tb_future_prediccion) + 1)})
    
    ranks = tb_predicciones["rank"].unique().sort()
    
    ganancia_mejor = float('-inf')
    vganancias_suavizadas = []
    
    for irank in ranks:
        gan_sum = f"gan_sum_{irank}"
        tb_ganancias = tb_ganancias.with_columns(pl.lit(0).alias(gan_sum))
        
        isems = tb_predicciones.filter(pl.col("rank") == irank)["isem"].unique().sort()
        
        for vsem in isems:
            print(irank, vsem)
            envg['OUTPUT']['status']['rank'] = irank
            envg['OUTPUT']['status']['isem'] = vsem
            save_output()
            
            campito = tb_predicciones.filter((pl.col("rank") == irank) & (pl.col("isem") == vsem))["campo"][0]
            temp_pred = tb_future_prediccion.select([campito, envg['PARAM']['dataset_metadata']['clase']])
            temp_pred = temp_pred.sort(campito, descending=True)
            
            temp_pred = temp_pred.with_columns(
                pl.when(pl.col(envg['PARAM']['dataset_metadata']['clase']).is_in(envg['PARAM']['train']['positivos']))
                .then(envg['PARAM']['train']['gan1'])
                .otherwise(envg['PARAM']['train']['gan0'])
                .alias("ganancia")
            )
            
            temp_pred = temp_pred.with_columns(pl.col("ganancia").cumsum().alias("ganancia_acum"))
            
            temp_pred = temp_pred.with_columns(
                pl.col("ganancia_acum")
                .rolling_mean(window_size=envg['PARAM']['graficar']['ventana_suavizado'], center=True)
                .alias("gan_suavizada")
            )
            
            ganancia_suavizada_max = temp_pred["gan_suavizada"].max()
            corte_mejor = temp_pred["gan_suavizada"].arg_max()
            
            tb_ganancias = tb_ganancias.with_columns(
                (pl.col(gan_sum) + temp_pred["ganancia_acum"]).alias(gan_sum),
                temp_pred["ganancia_acum"].alias(campito)
            )
            
            # MLFlow logging
            linea = {
                "rank": irank,
                "iteracion_bayesiana": tb_predicciones.filter((pl.col("rank") == irank) & (pl.col("isem") == vsem))["iteracion_bayesiana"][0],
                "qsemillas": 1,
                "semilla": tb_predicciones.filter((pl.col("rank") == irank) & (pl.col("isem") == vsem))["semilla"][0],
                "corte": corte_mejor,
                "ganancia": ganancia_suavizada_max,
                "metrica": ganancia_suavizada_max
            }
            mlog_log(linea, arch="ganancias_log.txt")
            
            del temp_pred
            gc.collect()
        
        tb_ganancias = tb_ganancias.with_columns((pl.col(gan_sum) / len(isems)).alias(gan_sum))
        
        tb_ganancias = tb_ganancias.with_columns(
            pl.col(gan_sum)
            .rolling_mean(window_size=envg['PARAM']['graficar']['ventana_suavizado'], center=True)
            .alias("gan_suavizada")
        )
        
        ganancia_suavizada_max = tb_ganancias["gan_suavizada"].max()
        corte_mejor = tb_ganancias["gan_suavizada"].arg_max()
        
        # MLFlow logging
        linea = {
            "rank": irank,
            "iteracion_bayesiana": tb_predicciones.filter(pl.col("rank") == irank)["iteracion_bayesiana"].min(),
            "qsemillas": len(isems),
            "semilla": -1,
            "corte": corte_mejor,
            "ganancia": ganancia_suavizada_max,
            "metrica": ganancia_suavizada_max
        }
        
        superacion = ganancia_suavizada_max > ganancia_mejor
        if superacion:
            ganancia_mejor = ganancia_suavizada_max
        
        mlog_log(linea, arch="ganancias_log.txt", parentreplicate=superacion)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        
        for s in range(len(isems)):
            campo = tb_predicciones.filter((pl.col("rank") == irank) & (pl.col("isem") == s+1))["campo"][0]
            plt.plot(tb_ganancias["envios"], tb_ganancias[campo], color="gray", alpha=0.5)
        
        plt.plot(tb_ganancias["envios"], tb_ganancias[gan_sum], color="red", linewidth=2)
        
        plt.xlim(envg['PARAM']['graficar']['envios_desde'], envg['PARAM']['graficar']['envios_hasta'])
        plt.title(f"Mejor gan prom = {int(ganancia_suavizada_max)}")
        plt.xlabel("Envios")
        plt.ylabel("Ganancia")
        
        plt.savefig(f"modelo_{irank:02d}_{tb_predicciones.filter(pl.col('rank') == irank)['iteracion_bayesiana'].min():03d}.pdf")
        plt.close()
        
        # Save gains
        tb_ganancias.write_csv(f"ganancias_{irank:02d}_{tb_predicciones.filter(pl.col('rank') == irank)['iteracion_bayesiana'].min():03d}.txt", separator="\t")
        
        vganancias_suavizadas.append(ganancia_suavizada_max)
    
    envg['OUTPUT']['ganancias_suavizadas'] = vganancias_suavizadas
    
    envg['OUTPUT']['time']['end'] = datetime.now().strftime("%Y%m%d %H%M%S")
    save_output()
    
    mlflow.end_run()
    
    print("z2501_EV_evaluate_conclase.py END")

if __name__ == "__main__":
    # Clean memory
    gc.collect()
    
    # Get command line arguments
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    
    main()