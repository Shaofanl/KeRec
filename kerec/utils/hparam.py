class HyperParameterSet(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError('Cannot find hyper-parameter with name: '+name)

    def __setattr__(self, name, value):
        self[name] = value

    def check(self, check_list, default_dict={}):
        """Check the hyper-parameters
        """
        for param_name in check_list:
            if param_name not in self:
                if param_name in default_dict:
                    self[param_name] = default_dict[param_name]
                else:
                    raise AttributeError("Hyper-parameters `{}` is required but not provided".format(param_name))
