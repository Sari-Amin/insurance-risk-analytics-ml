import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()

    def convert_transaction_month(self):
        """Convert TransactionMonth to datetime format."""
        self.df['TransactionMonth'] = pd.to_datetime(
            self.df['TransactionMonth'], errors='coerce'
        )

    def clean_gender(self):
        """Fill missing genders with 'NotSpecified'."""
        if 'Gender' in self.df.columns:
            self.df['Gender'] = self.df['Gender'].fillna('NotSpecified')

    def clean_province(self):
        """Fill missing provinces with 'Unknown'."""
        if 'Province' in self.df.columns:
            self.df['Province'] = self.df['Province'].fillna('Unknown')

    def clean_marital_status(self):
        """Fill missing marital status with 'Unknown'."""
        if 'MaritalStatus' in self.df.columns:
            self.df['MaritalStatus'] = self.df['MaritalStatus'].fillna('Unknown')

    def fill_crossborder(self):
        """Assume missing CrossBorder means 'No'."""
        if 'CrossBorder' in self.df.columns:
            self.df['CrossBorder'] = self.df['CrossBorder'].fillna('No')

    def handle_numeric_outliers(self, column, lower=0):
        
        if column in self.df.columns:
            upper = self.df[column][self.df[column] > 0].quantile(0.99)
            # Only clip values > 0
            mask = self.df[column] > 0
            self.df.loc[mask, column] = self.df.loc[mask, column].clip(lower=lower, upper=upper)

    def run_all(self):
        """Run all cleaning steps in order."""
        self.convert_transaction_month()
        self.clean_gender()
        self.clean_province()
        self.clean_marital_status()
        self.fill_crossborder()

        # Remove negative or extreme values from monetary fields
        self.handle_numeric_outliers('TotalPremium', lower=0)
        self.handle_numeric_outliers('TotalClaims', lower=0)

        return self.df
