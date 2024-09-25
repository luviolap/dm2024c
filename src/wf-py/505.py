import os
import gc
from typing import List, Dict
import csv
from datetime import datetime
import yaml
import matplotlib.pyplot as plt

# Clean memory
gc.collect()

# Parameters
PARAM: Dict[str, str] = {
    "experimento": "CA5050"
}

# Set working directory
os.chdir(os.path.expanduser("~/buckets/b1/"))

# Load environment
with open("miAmbiente.yml", "r") as file:
    mi_ambiente: Dict[str, str] = yaml.safe_load(file)

# Load dataset
def load_dataset(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        return list(reader)

dataset: List[Dict[str, str]] = load_dataset(mi_ambiente["dataset_competencia"])

# Create experiment folder
os.makedirs(f"./exp/{PARAM['experimento']}/", exist_ok=True)
os.chdir(f"./exp/{PARAM['experimento']}/")

# Sort dataset
dataset.sort(key=lambda x: (x["foto_mes"], x["numero_de_cliente"]))

campos_buenos: List[str] = [col for col in dataset[0].keys() if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]

def calculate_ratio(dataset: List[Dict[str, str]], field: str, condition_func) -> Dict[str, float]:
    ratio_by_month = {}
    for row in dataset:
        month = row["foto_mes"]
        if month not in ratio_by_month:
            ratio_by_month[month] = {"count": 0, "total": 0}
        ratio_by_month[month]["total"] += 1
        if condition_func(row[field]):
            ratio_by_month[month]["count"] += 1
    
    return {month: data["count"] / data["total"] for month, data in ratio_by_month.items()}

def plot_ratio(ratio_data: Dict[str, float], title: str, y_label: str, output_file: str):
    months = sorted(ratio_data.keys())
    ratios = [ratio_data[month] for month in months]
    
    plt.figure(figsize=(12, 6))
    plt.plot(months, ratios, 'o-')
    plt.title(title)
    plt.xlabel("Periodo")
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    
    for i in [0, 12, 24]:
        plt.axvline(x=months[i], color='green', linestyle='-', linewidth=1)
    for i in [6, 18, 30]:
        plt.axvline(x=months[i], color='green', linestyle=':', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Zeroes ratio
for campo in campos_buenos:
    zeroes_ratio = calculate_ratio(dataset, campo, lambda x: x == "0")
    plot_ratio(zeroes_ratio, f"Zeroes ratio - {campo}", "Zeroes ratio", f"zeroes_ratio_{campo}.pdf")

# NAs ratio
for campo in campos_buenos:
    nas_ratio = calculate_ratio(dataset, campo, lambda x: x == "")
    plot_ratio(nas_ratio, f"NAs ratio - {campo}", "NAs ratio", f"nas_ratio_{campo}.pdf")

# Averages
def calculate_average(dataset: List[Dict[str, str]], field: str, condition_func=None) -> Dict[str, float]:
    avg_by_month = {}
    for row in dataset:
        month = row["foto_mes"]
        value = row[field]
        if condition_func is None or condition_func(value):
            if month not in avg_by_month:
                avg_by_month[month] = {"sum": 0, "count": 0}
            avg_by_month[month]["sum"] += float(value)
            avg_by_month[month]["count"] += 1
    
    return {month: data["sum"] / data["count"] for month, data in avg_by_month.items()}

for campo in campos_buenos:
    avg_data = calculate_average(dataset, campo)
    zeroes_ratio = calculate_ratio(dataset, campo, lambda x: x == "0")
    
    plt.figure(figsize=(12, 6))
    months = sorted(avg_data.keys())
    averages = [avg_data[month] for month in months]
    
    plt.plot(months, averages, 'o-')
    plt.title(f"Promedios - {campo}")
    plt.xlabel("Periodo")
    plt.ylabel("Promedio")
    plt.xticks(rotation=45)
    
    for i in [0, 12, 24]:
        plt.axvline(x=months[i], color='green', linestyle='-', linewidth=1)
    for i in [6, 18, 30]:
        plt.axvline(x=months[i], color='green', linestyle=':', linewidth=1)
    
    median_zero_ratio = sorted(zeroes_ratio.values())[len(zeroes_ratio) // 2]
    for i, month in enumerate(months):
        if zeroes_ratio[month] > 0.99 and median_zero_ratio < 0.99:
            plt.axvline(x=month, color='red', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f"promedios_{campo}.pdf")
    plt.close()

# Averages (non-zero)
for campo in campos_buenos:
    avg_data = calculate_average(dataset, campo, lambda x: x != "0")
    zeroes_ratio = calculate_ratio(dataset, campo, lambda x: x == "0")
    
    plt.figure(figsize=(12, 6))
    months = sorted(avg_data.keys())
    averages = [avg_data[month] for month in months]
    
    plt.plot(months, averages, 'o-')
    plt.title(f"Promedios NO cero - {campo}")
    plt.xlabel("Periodo")
    plt.ylabel("Promedio valores no cero")
    plt.xticks(rotation=45)
    
    for i in [0, 12, 24]:
        plt.axvline(x=months[i], color='green', linestyle='-', linewidth=1)
    for i in [6, 18, 30]:
        plt.axvline(x=months[i], color='green', linestyle=':', linewidth=1)
    
    median_zero_ratio = sorted(zeroes_ratio.values())[len(zeroes_ratio) // 2]
    for i, month in enumerate(months):
        if zeroes_ratio[month] > 0.99 and median_zero_ratio < 0.99:
            plt.axvline(x=month, color='red', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f"promedios_nocero_{campo}.pdf")
    plt.close()

# Final mark
with open("zRend.txt", "a") as file:
    file.write(f"{datetime.now().strftime('%Y%m%d %H%M%S')}\n")