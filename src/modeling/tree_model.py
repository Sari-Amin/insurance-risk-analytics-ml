import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class ClaimSeverityTreeModel:
    def __init__(self, df):
        self.df = df[df['TotalClaims'] > 0].copy()
        self.model = None

    def prepare_data(self):
        features = [
            'Gender', 'Province', 'VehicleType', 'Make', 'Cubiccapacity',
            'Kilowatts', 'Bodytype', 'RegistrationYear'
        ]
        features = [f for f in features if f in self.df.columns]
        self.df = self.df.dropna(subset=features)

        X = pd.get_dummies(self.df[features], drop_first=True)
        y = self.df['TotalClaims']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        return {
            "rmse": rmse,
            "r2": r2,
            "model": self.model,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred
        }
