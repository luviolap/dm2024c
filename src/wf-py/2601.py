#!/usr/bin/env python3

# Workflow KA_evaluate_kaggle

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
import subprocess
import time

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

def graficar_ganancias(tb_ganancias_local: pl.DataFrame, irank: int, ibayesiana: int, qsemillas: int) -> None:
    campos_buenos = [col for col in tb_ganancias_local.columns if col not in ["envios", "rank"]]
    ymin = tb_ganancias_local.select(campos_buenos).min().min() * 0.9
    ymax = tb_ganancias_local.select(campos_buenos).max().max() * 1.1

    plt.figure(figsize=(10, 6))
    plt.grid(True)

    for s in range(1, qsemillas + 1):
        plt.plot(tb_ganancias_local["envios"], tb_ganancias_local[f"m{s}"], color="gray", alpha=0.5)

    plt.plot(tb_ganancias_local["envios"], tb_ganancias_local["gan_sum"], color="red", linewidth=2)

    plt.ylim(ymin, ymax)
    plt.title(f"Mejor gan prom = {tb_ganancias_local['gan_sum'].max():.2f}")
    plt.xlabel("Envios")
    plt.ylabel("Ganancia")

    plt.savefig(f"modelo_{irank:02d}_{ibayesiana:03d}.pdf")
    plt.close()

def main() -> None:
    print("z2601_KA_evaluate_kaggle.py START")
    
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
    
    cortes = range(envg['PARAM']['envios_desde'], envg['PARAM']['envios_hasta'] + 1, envg['PARAM']['envios_salto'])
    
    ranks = tb_predicciones["rank"].unique().sort()
    
    ganancia_mejor = float('-inf')
    vganancias = []
    
    for irank in ranks:
        tb_ganancias_local = pl.DataFrame({"envios": cortes, "rank": irank, "gan_sum": 0})
        
        isems = tb_predicciones.filter(pl.col("rank") == irank)["isem"].unique().sort()
        isem = [i for i in isems if i in envg['PARAM']['isems_submit']]
        
        for vsem in isems:
            print(irank, vsem)
            envg['OUTPUT']['status']['rank'] = irank
            envg['OUTPUT']['status']['isem'] = vsem
            save_output()
            
            nombre_raiz = f"{irank:02d}_{tb_predicciones.filter((pl.col('rank') == irank) & (pl.col('isem') == vsem))['iteracion_bayesiana'][0]:03d}_s{tb_predicciones.filter((pl.col('rank') == irank) & (pl.col('isem') == vsem))['semilla'][0]}"
            
            campito = tb_predicciones.filter((pl.col("rank") == irank) & (pl.col("isem") == vsem))["campo"][0]
            temp_pred = tb_future_prediccion.select([envg['PARAM']['dataset_metadata']['entity_id'], campito])
            temp_pred = temp_pred.sort(campito, descending=True)
            
            for icorte in cortes:
                temp_pred = temp_pred.with_columns(
                    pl.when(pl.col(envg['PARAM']['dataset_metadata']['entity_id']).rank() <= icorte)
                    .then(1)
                    .otherwise(0)
                    .alias("Predicted")
                )
                
                nom_submit = f"{envg['PARAM']['experimento']}_{nombre_raiz}_{icorte:05d}.csv"
                
                print("write prediccion Kaggle")
                print("Columnas del dataset: ", temp_pred.columns)
                temp_pred.select([envg['PARAM']['dataset_metadata']['entity_id'], "Predicted"]).write_csv(nom_submit)
                print("written prediccion Kaggle")
                
                # Submit to Kaggle
                submitear = True
                if "rango_submit" in envg['PARAM']:
                    if vsem not in envg['PARAM']['rango_submit']:
                        submitear = False
                
                if "competition" in envg['PARAM'] and submitear:
                    submit_script = f"""#!/bin/bash
source ~/.venv/bin/activate
kaggle competitions submit -c {envg['PARAM']['competition']} -f {nom_submit} -m "{envg['PARAM']['experimento']}, {nom_submit}"
deactivate
"""
                    with open("subir.sh", "w") as f:
                        f.write(submit_script)
                    os.chmod("subir.sh", 0o744)
                    
                    res = subprocess.run(["./subir.sh"], capture_output=True, text=True)
                    time.sleep(3)  # Wait to avoid saturation
                    
                    if "Successfully" in res.stdout:
                        res = subprocess.run(["~/install/list", nom_submit], capture_output=True, text=True)
                        print("res= ", res.stdout)
                        score = float(res.stdout.strip())
                        tb_ganancias_local = tb_ganancias_local.with_columns(
                            pl.when(pl.col("envios") == icorte)
                            .then(pl.lit(score))
                            .otherwise(pl.col(f"m{vsem}"))
                            .alias(f"m{vsem}")
                        )
                        tb_ganancias_local = tb_ganancias_local.with_columns(
                            pl.when(pl.col("envios") == icorte)
                            .then(pl.col("gan_sum") + score)
                            .otherwise(pl.col("gan_sum"))
                            .alias("gan_sum")
                        )
                        
                        # MLFlow logging
                        linea = {
                            "rank": irank,
                            "iteracion_bayesiana": tb_predicciones.filter((pl.col("rank") == irank) & (pl.col("isem") == vsem))["iteracion_bayesiana"][0],
                            "qsemillas": 1,
                            "semilla": tb_predicciones.filter((pl.col("rank") == irank) & (pl.col("isem") == vsem))["semilla"][0],
                            "corte": icorte,
                            "ganancia": score,
                            "metrica": score
                        }
                        mlog_log(linea, arch="ganancias_log.txt")
            
            del temp_pred
            gc.collect()
        
        tb_ganancias_local = tb_ganancias_local.with_columns((pl.col("gan_sum") / len(isems)).alias("gan_sum"))
        vganancias.append(tb_ganancias_local["gan_sum"].max())
        
        graficar_ganancias(
            tb_ganancias_local,
            irank,
            ibayesiana=tb_predicciones.filter(pl.col("rank") == irank)["iteracion_bayesiana"].min(),
            qsemillas=len(isems)
        )
        
        # MLFlow logging for average gains
        for icorte in cortes:
            ganancia_media = tb_ganancias_local.filter(pl.col("envios") == icorte)["gan_sum"][0]
            linea = {
                "rank": irank,
                "iteracion_bayesiana": tb_predicciones.filter(pl.col("rank") == irank)["iteracion_bayesiana"].min(),
                "qsemillas": len(isems),
                "semilla": -1,
                "corte": icorte,
                "ganancia": ganancia_media,
                "metrica": ganancia_media
            }
            
            superacion = ganancia_media > ganancia_mejor
            if superacion:
                ganancia_mejor = ganancia_media
            
            mlog_log(linea, arch="ganancias_log.txt", parentreplicate=superacion)
        
        # Accumulate gains
        if "tb_ganancias" not in locals():
            tb_ganancias = tb_ganancias_local
        else:
            tb_ganancias = pl.concat([tb_ganancias, tb_ganancias_local])
        
        tb_ganancias.write_csv("tb_ganancias.txt", separator="\t")
    
    envg['OUTPUT']['ganancias_suavizadas'] = vganancias
    
    envg['OUTPUT']['time']['end'] = datetime.now().strftime("%Y%m%d %H%M%S")
    save_output()
    
    mlflow.end_run()
    
    print("z2601_KA_evaluate_kaggle.py END")

if __name__ == "__main__":
    # Clean memory
    gc.collect()
    
    # Get command line arguments
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    
    main()