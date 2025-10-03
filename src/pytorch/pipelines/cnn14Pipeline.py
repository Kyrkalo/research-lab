class Cnn14Pipeline:
    def __init__(self, config):
        self.config = config

    def setup(self):
        print(f"Skipping setup the model. Model name: {self.config['model_name']}")
        return self

    def run(self):
        print(f"Skipping creating and training the model. Model name: {self.config['model_name']}")
        return self