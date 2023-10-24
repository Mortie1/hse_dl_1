import numpy as np
from scipy.special import softmax, log_softmax
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        self.output = np.where(input < 0, 0, input)
        return self.output
    
    
    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        df_dx = np.where(input < 0, 0, 1)
        return grad_output * df_dx


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        self.output = np.reciprocal(1 + np.exp(-input))
        return self.output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        df_dx = np.exp(-input) * np.reciprocal(np.square(np.exp(-input) + 1))
        return grad_output * df_dx


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        x = input - np.max(input)
        self.output = softmax(x, axis=1)
        return self.output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        grad_input = self.output * (grad_output - np.sum(self.output * grad_output, axis=1, keepdims=True))
        return grad_input
        


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        x = input - np.max(input)
        self.output = log_softmax(x, axis=1)
        return self.output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        softmax = np.exp(self.output)
        grad_log_softmax = grad_output - softmax * np.sum(grad_output, axis=1, keepdims=True)
        return grad_log_softmax
