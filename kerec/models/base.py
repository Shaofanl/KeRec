from abc import ABC, abstractmethod


class BaseModel(ABC):
    """The base class for all learning models

    Methods:
        _check_hparam:  check the hyper-parameters
        compile:    link model with optimizer/loss/metrics
        load:       load the model from files
        save:       save the model to files
        predict:  use the trained model to inference

    Abstract Methods:
        _build:     build the model
        fit:      fitthe model

    Attributes:
        hparam_default_list:    default values fro hyper-parameters 
        hparam_check_dict:  checklist for hyper-parameters    
    """
    def __init__(self, hparam):
        self.hparam = hparam
        self._check_hparam()
        self._build()

    @abstractmethod
    def _build(self):
        """Build the model
        """
        raise NotImplementedError

    def compile(self, optimizer, loss, metrics):
        """Link optimizer, loss, metrics with the model

            To extend it to other framework (e.g. PyTorch), we can simply
            store these variables in `self` and retrive them during training
        """
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    @abstractmethod
    def fit(self, **kwargs):
        """Training the model
        """
        raise NotImplementedError

    def predict(self, **kwargs):
        """Inference the rating/ranking/output

        It's an optional method
        """
        raise NotImplementedError

    def load(self, path, model_name):
        """Retrive a model and its optimizer/hyper-parameters from files 
        """
        raise NotImplementedError

    def save(self, path, model_name):
        """Save a model and its optimizer/hyper-parameters to files 
        """
        raise NotImplementedError

    # check hparam
    def _check_hparam(self):
        """Check the completeness of hyper-parameters
        """
        self.hparam.check(
            self.hparam_check_list,
            self.hparam_default_dict
        )

    @property
    def hparam_check_list(self):
        """hyper-parameters necessary fields 
        """
        return []
        # raise NotImplementedError

    @property
    def hparam_default_dict(self):
        """hyper-parameters default values
        """
        return {}
        # raise NotImplementedError
