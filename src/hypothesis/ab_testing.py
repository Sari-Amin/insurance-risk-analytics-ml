import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

class HypothesisTester:
    def __init__(self, df):
        self.df = df.copy()
        self.df['HasClaim'] = self.df['TotalClaims'] > 0
        self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
        self.df['NegativeMargin'] = self.df['Margin'] < 0
        self.df['HighSeverity'] = self.df['TotalClaims'] > self.df['TotalClaims'].median()

    def _validate_two_groups(self, series):
        unique_vals = series.dropna().unique()
        if len(unique_vals) != 2:
            raise ValueError("Exactly 2 groups are required for A/B testing.")
        return unique_vals

    # === T-TESTS ===

    def test_claim_severity_ttest(self, group_col):

        df = self.df[self.df['TotalClaims'] > 0][[group_col, 'TotalClaims']].dropna()
        groups = self._validate_two_groups(df[group_col])

        a = df[df[group_col] == groups[0]]['TotalClaims']
        b = df[df[group_col] == groups[1]]['TotalClaims']

        if a.var() == 0 or b.var() == 0:
            return {"method": "t-test", "stat": None, "p": None, "error": "No variance in one group"}

        stat, p = ttest_ind(a, b, equal_var=False)
        return {"method": "t-test", "target": "Claim Severity", "groups": groups.tolist(), "stat": stat, "p": p}

    def test_margin_difference_ttest(self, group_col):

        df = self.df[[group_col, 'Margin']].dropna()
        groups = self._validate_two_groups(df[group_col])

        a = df[df[group_col] == groups[0]]['Margin']
        b = df[df[group_col] == groups[1]]['Margin']

        if a.var() == 0 or b.var() == 0:
            return {"method": "t-test", "stat": None, "p": None, "error": "No variance in one group"}

        stat, p = ttest_ind(a, b, equal_var=False)
        return {"method": "t-test", "target": "Margin", "groups": groups.tolist(), "stat": stat, "p": p}

    def test_total_premium_ttest(self, group_col):

        df = self.df[[group_col, 'TotalPremium']].dropna()
        groups = self._validate_two_groups(df[group_col])

        a = df[df[group_col] == groups[0]]['TotalPremium']
        b = df[df[group_col] == groups[1]]['TotalPremium']

        if a.var() == 0 or b.var() == 0:
            return {"method": "t-test", "stat": None, "p": None, "error": "No variance in one group"}

        stat, p = ttest_ind(a, b, equal_var=False)
        return {"method": "t-test", "target": "Total Premium", "groups": groups.tolist(), "stat": stat, "p": p}

    # === CHI-SQUARED TESTS ===

    def test_claim_frequency_chi2(self, group_col):

        df = self.df[[group_col, 'HasClaim']].dropna()
        groups = self._validate_two_groups(df[group_col])
        contingency = pd.crosstab(df[group_col], df['HasClaim'])

        if contingency.min().min() == 0:
            return {"method": "chi-squared", "stat": None, "p": None, "error": "Empty contingency cell"}

        stat, p, _, _ = chi2_contingency(contingency)
        return {"method": "chi-squared", "target": "Has Claim", "groups": groups.tolist(), "stat": stat, "p": p}

    def test_high_severity_chi2(self, group_col):

        df = self.df[self.df['TotalClaims'] > 0][[group_col, 'HighSeverity']].dropna()
        groups = self._validate_two_groups(df[group_col])
        contingency = pd.crosstab(df[group_col], df['HighSeverity'])

        if contingency.min().min() == 0:
            return {"method": "chi-squared", "stat": None, "p": None, "error": "Empty contingency cell"}

        stat, p, _, _ = chi2_contingency(contingency)
        return {"method": "chi-squared", "target": "High Severity", "groups": groups.tolist(), "stat": stat, "p": p}

    def test_negative_margin_chi2(self, group_col):
        
        df = self.df[[group_col, 'NegativeMargin']].dropna()
        groups = self._validate_two_groups(df[group_col])
        contingency = pd.crosstab(df[group_col], df['NegativeMargin'])

        if contingency.min().min() == 0:
            return {"method": "chi-squared", "stat": None, "p": None, "error": "Empty contingency cell"}

        stat, p, _, _ = chi2_contingency(contingency)
        return {"method": "chi-squared", "target": "Negative Margin", "groups": groups.tolist(), "stat": stat, "p": p}
