
from anyio import Path


class Exporter():

    def __init__(self, config=None):
        self.config = config if config else {}

    def getPath(self, extension, prefix=""):
        
        return Path(f"{prefix}{self.config["model_name"]}.{extension}")

    def run(self):
        return self
    
    def setup(self):
        return self