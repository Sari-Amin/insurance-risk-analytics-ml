import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

class ClaimSeverityModel:
    def __init__(self, df):
        self.df = df[df['TotalClaims'] > 0].copy()  # only with claims
        self.model = None

    def prepare_features(self):

        self.df = self.df.dropna(subset=['TotalClaims'])  # safety
        features = [
            'Gender', 'Province', 'VehicleType', 'Make', 'Cubiccapacity',
            'Kilowatts', 'Bodytype', 'RegistrationYear'
        ]
        # Filter only available features
        features = [col for col in features if col in self.df.columns]
        self.df = self.df.dropna(subset=features)

        X = pd.get_dummies(self.df[features], drop_first=True)
        y = self.df['TotalClaims']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_features()
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {"rmse": rmse, "r2": r2, "model": self.model}
