import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, df):
        self.df = df

    def plot_histogram(self, column, bins=50):
        plt.figure(figsize=(8, 4))
        sns.histplot(self.df[column].dropna(), bins=bins, kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

    def plot_boxplot(self, column):
        plt.figure(figsize=(8, 4))
        sns.boxplot(y=self.df[column].dropna())
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
