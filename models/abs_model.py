from abc import ABC, abstractmethod
import torch.nn as nn

class ABSmodule(nn.Module, ABC):

    @abstractmethod
    def infer_text(self, batch, **kwargs):
        raise NotImplemented
    
    @abstractmethod
    def forward(self, batch: dict) -> dict:
        '''
        Input: 
            batch: dictionary
        Output:
            A dictionary containing the outputs of the model
        '''
        raise NotImplemented

