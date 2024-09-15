from scipy import stats
import numpy as np

def perform_wilcoxon_test(gains1, gains2):
    result = stats.wilcoxon(gains1, gains2)
    print(f"Number of samples: {len(gains1)}")
    print(f"p-value: {result.pvalue:.5f}")
    print("Significant" if result.pvalue < 0.05 else "Not significant")
    print()

# 1 gain
gains1 = [41470000]
gains2 = [37790000]
perform_wilcoxon_test(gains1, gains2)

# 2 gains
gains1 = [41470000, 48480000]
gains2 = [37790000, 43660000]
perform_wilcoxon_test(gains1, gains2)

# 3 gains
gains1 = [41470000, 48480000, 50610000]
gains2 = [37790000, 43660000, 47840000]
perform_wilcoxon_test(gains1, gains2)

# 4 gains
gains1 = [41470000, 48480000, 50610000, 45580000]
gains2 = [37790000, 43660000, 47840000, 44490000]
perform_wilcoxon_test(gains1, gains2)

# 5 gains
gains1 = [41470000, 48480000, 50610000, 45580000, 52780000]
gains2 = [37790000, 43660000, 47840000, 44490000, 46750000]
perform_wilcoxon_test(gains1, gains2)

# 6 gains
gains1 = [41470000, 48480000, 50610000, 45580000, 52780000, 49970000]
gains2 = [37790000, 43660000, 47840000, 44490000, 46750000, 44300000]
perform_wilcoxon_test(gains1, gains2)

# 7 gains
gains1 = [41470000, 48480000, 50610000, 45580000, 52780000, 49970000, 52810000]
gains2 = [37790000, 43660000, 47840000, 44490000, 46750000, 44300000, 42840000]
perform_wilcoxon_test(gains1, gains2)

# 8 gains
gains1 = [41470000, 48480000, 50610000, 45580000, 52780000, 49970000, 52810000, 43060000]
gains2 = [37790000, 43660000, 47840000, 44490000, 46750000, 44300000, 42840000, 37300000]
perform_wilcoxon_test(gains1, gains2)

# 9 gains
gains1 = [41470000, 48480000, 50610000, 45580000, 52780000, 49970000, 52810000, 43060000, 49660000]
gains2 = [37790000, 43660000, 47840000, 44490000, 46750000, 44300000, 42840000, 37300000, 43730000]
perform_wilcoxon_test(gains1, gains2)