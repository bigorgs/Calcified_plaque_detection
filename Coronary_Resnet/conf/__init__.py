
from service.Coronary_Resnet.conf import global_settings as settings
# import global_settings as settings

class Settings:
    def __init__(self, settings):

        for attr in dir(settings):
            if attr.isupper():
                setattr(self, attr, getattr(settings, attr))

settings = Settings(settings)