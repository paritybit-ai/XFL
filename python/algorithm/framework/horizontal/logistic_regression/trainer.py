from algorithm.core.horizontal.template.torch.fedavg.trainer import TemplateTrainer
from algorithm.model.logistic_regression import LogisticRegression


class HorizontalLogisticRegressionTrainer(TemplateTrainer):
    def set_model(self):
        model_config = self.model_info.get("config")
        model = LogisticRegression(input_dim=model_config["input_dim"],
                                   bias=model_config["bias"])
        return model
