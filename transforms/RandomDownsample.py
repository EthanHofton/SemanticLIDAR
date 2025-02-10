class RandomDownsample:
    def __init__(self, max_points=1024):
        """
        Args:
            max_points (int): Maximum number of points after downsampling.
        """
        self.max_points = max_points

    def __call__(self, points, labels):
        if points.shape[0] > self.max_points:
            # Randomly sample `max_points` from the point cloud
            idxs = np.random.choice(points.shape[0], self.max_points, replace=False)
            points = points[idxs]
            labels = labels[idxs]
        elif points.shape[0] < self.max_points:
            # Pad the point cloud with zeros (or some other padding strategy)
            padding = np.zeros((self.max_points - points.shape[0], 3))  # Padding for 3D points
            points = np.vstack([points, padding])
            labels = np.hstack([labels, np.zeros(self.max_points - labels.shape[0])])  # Pad labels accordingly

        return points, labels
