import shap
import matplotlib.pyplot as plt

class ModelExplainer:
    def __init__(self, model, X, model_type="tree"):
        self.model = model
        self.X = X

        if model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        else:
            self.explainer = shap.Explainer(model, X)

        self.shap_values = self.explainer(self.X)

    def summary_plot(self, max_display=10):
        shap.summary_plot(self.shap_values, self.X, plot_type="bar", max_display=max_display)

    def full_summary_plot(self):
        shap.summary_plot(self.shap_values, self.X)

    def force_plot(self, index=0):
        shap.plots.force(self.explainer.expected_value, self.shap_values[index], self.X.iloc[index])
