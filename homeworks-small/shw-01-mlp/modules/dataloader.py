import numpy as np

class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        return self.X.shape[0] // self.batch_size if self.X.shape[0] % self.batch_size == 0 else self.X.shape[0] // self.batch_size + 1

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        self.batch_id = 0
        if self.shuffle:
            # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
            p = np.random.permutation(self.X.shape[0])
            self.X = self.X[p]
            self.y = self.y[p]
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        if self.batch_id < len(self):
            x_batch = self.X[self.batch_id * self.batch_size : (self.batch_id + 1) * self.batch_size]
            y_batch = self.y[self.batch_id * self.batch_size : (self.batch_id + 1) * self.batch_size]
            self.batch_id += 1
            return (x_batch, y_batch)
        else:
            raise StopIteration
