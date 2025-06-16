import pandas as pd


class DataLoader:
    def __init__(self, path, sep = None):
        self.filePath = path
        self.sep = sep
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.filePath, sep=self.sep)
            return self.df
        except Exception as e:
            print(f"File failed to load:{e}")
    
    def basic_info(self):
        if self.df is not None:
            print("Data loaded!")
            print("shape:", self.df.shape)
            print("Columns:", list(self.df.columns))
            self.df.info()
        else:
            print("Data doesn't loaded properly!")
    

    def describe_data(self):

        if self.df is not None:
            return self.df.describe()
        else:
            raise ValueError("load data first!")
        

    def missing_value(self):

        if self.df is not None:
            return self.df.isnull().sum().sort_values(ascending=False)
        else:
            raise ValueError("load data first!")