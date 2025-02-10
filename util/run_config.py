from faker import Faker
import time
import wandb

class RunConfig():

    def __init__(self, cnf):
        self.config = cnf
        self.fake = Faker()
        self.run_id = self._generate_run_id()

    def _generate_run_id(self):
        word1 = self.fake.word()
        word2 = self.fake.word()
        num = str(int(time.time()))
        return f"{word1}-{word2}-{num}"

    def __getattr__(self, name):
        if name in self.config:
            return self.config[name]

        raise AttributeError(f'Arrtribute {name} not in RunConfig')

class WandBRunConfig():

    def __init__(self):
        pass

    def __getattr__(self, name):
        if name == 'run_id':
            return wandb.run.id

        if getattr(wandb.config, name, None):
            return getattr(wandb.config, name)
        else:
            raise AttributeError(f'Attribute {name} not in WandBRunConfig')
