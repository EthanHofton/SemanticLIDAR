from faker import Faker
import time

class RunConfig():

    def __init__(self, epochs=5, lr=1e-3):
        self.fake = Faker()
        self.epochs = epochs
        self.lr = lr
        self.run_id = self._generate_run_id()

    def _generate_run_id(self):
        word1 = self.fake.word()
        word2 = self.fake.word()
        num = str(int(time.time()))
        return f"{word1}-{word2}-{num}"
