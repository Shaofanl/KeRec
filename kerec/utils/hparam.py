import json


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

    def serialize(self):
        return json.dumps(self)

    def save(self, filename):
        raise NotImplementedError
        # TODO: object cannot be serialized to string
        # with open(filename, 'w') as f:
        #     f.write(self.serialize())


    def deserialize(self, ser):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError

