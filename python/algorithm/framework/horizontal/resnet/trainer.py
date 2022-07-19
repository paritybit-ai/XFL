import torch
from algorithm.model.resnet import ResNet50
from algorithm.core.horizontal.template.torch.fedavg.trainer import TemplateTrainer
from torch.utils.data import TensorDataset

class HorizontalResnetTrainer(TemplateTrainer):
    def set_model(self):
        model_config = self.model_info.get("config")
        model = ResNet50(num_classes=model_config["num_classes"])
        return model
    
    def set_dataset(self):
        ''' Define default self.train_dataloader and self.val_dataloader '''
        train_data = self.get_data(self.input_trainset)
        trainset, valset = None, None
        
        if train_data is not None:
            trainset = TensorDataset(torch.tensor(train_data.features(), dtype=torch.float32).to(self.device),
                                     torch.tensor(train_data.label(), dtype=torch.long).to(self.device))
        
        val_data = self.get_data(self.input_valset)
        if val_data is not None:
            valset = TensorDataset(torch.tensor(val_data.features(), dtype=torch.float32).to(self.device),
                                   torch.tensor(val_data.label(), dtype=torch.long).to(self.device))
        return trainset, valset