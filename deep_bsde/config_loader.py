"""  Config loader that stores the pre-defined params of the PDE  """


class BaseConfigLoader:
    """ Base class to load pre-define configuration  """
    name = 'base_config'

    @staticmethod
    def get_loader_from_flags(config_name):
        """ Find out correct derived class loader """
        loader_cls = None
        for sub_loader_cls in BaseConfigLoader.__subclasses__():
            if sub_loader_cls.name == config_name:
                loader_cls = sub_loader_cls

        if loader_cls is None:
            raise RuntimeError('Unknown config name:' + config_name)

        return loader_cls()


class BSConfigLoader(BaseConfigLoader):
    name = 'BlackScholes'

    equation_dim = 1
    sigma = 0.2
    mu = 0.05
    spot = 100
    strike = 95
    init_y_range = [10, 15]
