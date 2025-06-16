import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class Visualizer:
    def __init__(self, df):
        self.df = df.copy()

    def plot_histogram(self, column, bins=50):
        plt.figure(figsize=(8, 4))
        sns.histplot(data= self.df, x = column, bins=bins, kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

    def plot_boxplot(self, column):
        plt.figure(figsize=(8, 4))
        sns.boxplot(data= self.df, y=self.df[column].dropna())
        plt.title(f"Boxplot of {column}")
        plt.xlabel(column)
        plt.show()

    def plot_loss_ratio_by_group(self, group_col, min_threshold=1000):
        grouped = self.df.groupby(group_col).agg({
            'TotalClaims': 'sum',
            'TotalPremium': 'sum',
            group_col: 'count'
        })
        grouped = grouped[grouped[group_col] >= min_threshold]  # avoid small group noise
        grouped['LossRatio'] = grouped['TotalClaims'] / grouped['TotalPremium']
        grouped = grouped.sort_values('LossRatio', ascending=False)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=grouped.index, y=grouped['LossRatio'])
        plt.xticks(rotation=45)
        plt.title(f"Loss Ratio by {group_col}")
        plt.ylabel("Loss Ratio")
        plt.xlabel(group_col)
        plt.tight_layout()
        plt.show()


    def plot_log_histogram(self, column, clip_upper=0.99):
        data = self.df[column].dropna()
        data = data[data > 0]  # log can't handle zero or negative
        clipped = np.clip(data, a_min=None, a_max=data.quantile(clip_upper))
        log_values = np.log1p(clipped)  # log(x + 1)

        plt.figure(figsize=(8, 4))
        sns.histplot(log_values, bins=50, kde=True)
        plt.title(f"Log-Transformed Histogram of {column}")
        plt.xlabel(f"log({column} + 1)")
        plt.show()

    def plot_log_boxplot(self, column, clip_upper=0.99):
        data = self.df[column].dropna()
        data = data[data > 0]
        clipped = np.clip(data, a_min=None, a_max=data.quantile(clip_upper))
        log_values = np.log1p(clipped)

        plt.figure(figsize=(8, 4))
        sns.boxplot(x=log_values)
        plt.title(f"Log-Transformed Boxplot of {column}")
        plt.xlabel(f"log({column} + 1)")
        plt.show()
