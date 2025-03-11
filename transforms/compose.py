class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points, labels):
        for t in self.transforms:
            points, labels = t(points, labels)

        return points, labels

