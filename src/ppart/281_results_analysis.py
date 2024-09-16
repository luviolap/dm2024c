```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the data
df = pd.read_csv('~/buckets/b1/exp/HT2810/gridsearch.txt', sep='\t')

# Display basic information about the dataset
print(df.info())
print(df.describe())

# Sort by ganancia_mean descending
df_sorted = df.sort_values('ganancia_mean', ascending=False)

print("Top 10 parameter combinations:")
print(df_sorted.head(10))

# Function to plot parameter impact
def plot_parameter_impact(df, param):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=param, y='ganancia_mean', data=df)
    plt.title(f'Impact of {param} on ganancia_mean')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.violinplot(x=param, y='ganancia_mean', data=df)
    plt.title(f'Distribution of ganancia_mean for different {param} values')
    plt.xticks(rotation=45)
    plt.show()

# Analyze the impact of each parameter on ganancia_mean
for param in ['cp', 'maxdepth', 'minsplit', 'minbucket']:
    plot_parameter_impact(df, param)

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[['cp', 'maxdepth', 'minsplit', 'minbucket', 'ganancia_mean']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Parameters and ganancia_mean')
plt.show()

# 3D scatter plot for top 3 most correlated parameters
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['cp'], df['maxdepth'], df['minsplit'], c=df['ganancia_mean'], cmap='viridis')
ax.set_xlabel('cp')
ax.set_ylabel('maxdepth')
ax.set_zlabel('minsplit')
plt.colorbar(scatter, label='ganancia_mean')
plt.title('3D Scatter Plot of cp, maxdepth, minsplit, and ganancia_mean')
plt.show()

# Function to get best combinations excluding certain parameters
def best_combinations_excluding(df, exclude_params):
    remaining_params = [p for p in ['cp', 'maxdepth', 'minsplit', 'minbucket'] if p not in exclude_params]
    return df.groupby(remaining_params)['ganancia_mean'].mean().sort_values(ascending=False).head(10)

print("Best combinations excluding cp:")
print(best_combinations_excluding(df, ['cp']))

print("\nBest combinations excluding maxdepth:")
print(best_combinations_excluding(df, ['maxdepth']))

print("\nBest combinations excluding minsplit and minbucket:")
print(best_combinations_excluding(df, ['minsplit', 'minbucket']))

# Analyze parameter interactions
plt.figure(figsize=(15, 10))
sns.pairplot(df[['cp', 'maxdepth', 'minsplit', 'minbucket', 'ganancia_mean']], hue='maxdepth', plot_kws={'alpha': 0.6})
plt.suptitle('Pairwise Relationships between Parameters and ganancia_mean', y=1.02)
plt.show()

# Identify regions of high performance
high_performance = df[df['ganancia_mean'] > df['ganancia_mean'].quantile(0.9)]
print("High-performance parameter ranges:")
for param in ['cp', 'maxdepth', 'minsplit', 'minbucket']:
    print(f"{param}: {high_performance[param].min()} to {high_performance[param].max()}")

# Identify regions of low performance
low_performance = df[df['ganancia_mean'] < df['ganancia_mean'].quantile(0.1)]
print("\nLow-performance parameter ranges:")
for param in ['cp', 'maxdepth', 'minsplit', 'minbucket']:
    print(f"{param}: {low_performance[param].min()} to {low_performance[param].max()}")

# PCA to visualize parameter importance
scaler = StandardScaler()
pca = PCA()
pca_result = pca.fit_transform(scaler.fit_transform(df[['cp', 'maxdepth', 'minsplit', 'minbucket']]))

plt.figure(figsize=(10, 6))
plt.bar(range(1, 5), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA: Explained Variance Ratio by Principal Component')
plt.show()

print("PCA Component Loadings:")
print(pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=['cp', 'maxdepth', 'minsplit', 'minbucket']))

# Analyzing the stability of top performers
top_10_percent = df_sorted.head(int(len(df) * 0.1))
print("\nStability of top 10% performers:")
for param in ['cp', 'maxdepth', 'minsplit', 'minbucket']:
    print(f"{param} - Most common value: {top_10_percent[param].mode().values[0]}, Frequency: {top_10_percent[param].value_counts().iloc[0] / len(top_10_percent):.2%}")

# Suggestion for next iteration of grid search
print("\nSuggested ranges for next iteration of grid search:")
for param in ['cp', 'maxdepth', 'minsplit', 'minbucket']:
    lower = top_10_percent[param].quantile(0.1)
    upper = top_10_percent[param].quantile(0.9)
    print(f"{param}: {lower} to {upper}")

# Efficiency analysis
efficiency = df.copy()
efficiency['rank'] = efficiency['ganancia_mean'].rank(ascending=False)
efficiency['efficiency'] = efficiency['ganancia_mean'] / efficiency['rank']

print("\nTop 10 most efficient parameter combinations:")
print(efficiency.sort_values('efficiency', ascending=False).head(10)[['cp', 'maxdepth', 'minsplit', 'minbucket', 'ganancia_mean', 'efficiency']])

# Visualize efficiency
plt.figure(figsize=(12, 6))
sns.scatterplot(x='rank', y='efficiency', hue='maxdepth', size='ganancia_mean', data=efficiency)
plt.title('Efficiency of Parameter Combinations')
plt.xlabel('Rank (based on ganancia_mean)')
plt.ylabel('Efficiency (ganancia_mean / rank)')
plt.show()
```