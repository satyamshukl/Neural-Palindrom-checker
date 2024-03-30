import numpy as np
import pickle  # Importing necessary libraries/modules

SEED = 18  # Setting a seed for reproducibility
np.random.seed(SEED)  # Seeding numpy's random number generator


class Palindrome():
    """
    Class for the Palindrome neural network model.

    Attributes:
        learning_rate (float): The learning rate for training.
        momentum (float): The momentum for training.
        input_size (int): The size of the input layer.
        hidden_size (int): The size of the hidden layer.
        threshold (float): The threshold for classification.
        weights (dict): Dictionary containing the weights of the model.
        velocity (dict): Dictionary containing the velocity for momentum.
    """

    def __init__(self, learning_rate=0.1, momentum=0.9, input_size=10, hidden_size=2) -> None:
        """
        Initializes the Palindrome model with default or given parameters.

        Parameters:
            learning_rate (float): The learning rate for training.
            momentum (float): The momentum for training.
            input_size (int): The size of the input layer.
            hidden_size (int): The size of the hidden layer.
            threshold (float): The threshold for classification.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weights = {
            1: np.random.uniform(-1, 1, (hidden_size, input_size)),
            2: np.random.uniform(-1, 1, (1, hidden_size))
        }  
        self.velocity = { 
            1: np.zeros((hidden_size, input_size)),
            2: np.zeros((1, hidden_size))
        }

    def ReLU(self, Z):
        """
        Applies the Rectified Linear Unit (ReLU) activation function element-wise to the input.

        Parameters:
            Z (numpy.ndarray): Input to the ReLU function.

        Returns:
            numpy.ndarray: Output of the ReLU function.
        """
        return np.maximum(0, Z)

    def ReLU_grad(self, Z):
        """
        Computes the gradient of the ReLU activation function.

        Parameters:
            Z (numpy.ndarray): Input to the ReLU function.

        Returns:
            None
        """
        self.relu_grad = np.where(Z <= 0, 0, 1)

    def sigmoid(self, logits):
        """
        Applies the sigmoid activation function element-wise to the input.

        Parameters:
            logits (numpy.ndarray): Input to the sigmoid function.

        Returns:
            numpy.ndarray: Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-logits))

    def forward(self, input):
        """
        Performs forward pass through the neural network.

        Parameters:
            input (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output prediction.
        """
        self.hidden = np.matmul(self.weights[1], input)  
        self.Z = self.ReLU(self.hidden)  
        self.ReLU_grad(self.Z) 
        self.logits = np.matmul(self.weights[2], self.Z) 
        self.output = self.sigmoid(self.logits)
        return self.output

    def backward(self, inputs, targets, predictions):
        """
        Performs backward pass through the neural network and updates weights.

        Parameters:
            inputs (numpy.ndarray): Input data.
            targets (numpy.ndarray): Target labels.
            predictions (numpy.ndarray): Predicted labels.

        Returns:
            None
        """
        batch_size = inputs.shape[1]
        grad1 = np.zeros_like(self.weights[1])
        grad2 = np.zeros_like(self.weights[2])

        for i in range(batch_size):
            input = inputs[:, i].reshape(-1, 1)
            target = targets[:, i].reshape(1, -1)
            prediction = predictions[:, i].reshape(1, -1)

            grad1 += (self.relu_grad[:, i].reshape(self.weights[2].T.shape) * self.weights[2].T) @ (prediction - target).T @ input.T
            grad2 += (prediction - target).T @ self.Z[:, i].reshape(self.hidden_size, 1).T  # @ .reshape(1, self.hidden_size)

        grad1 /= batch_size
        grad2 /= batch_size

        self.velocity[1] = self.momentum * self.velocity[1] + (1 - self.momentum) * grad1  
        self.velocity[2] = self.momentum * self.velocity[2] + (1 - self.momentum) * grad2

        self.weights[1] -= self.learning_rate * self.velocity[1]  
        self.weights[2] -= self.learning_rate * self.velocity[2]

    def loss(self, target, prediction):
        """
        Computes the binary cross-entropy loss.

        Parameters:
            target (numpy.ndarray): Target labels.
            prediction (numpy.ndarray): Predicted labels.

        Returns:
            float: Binary cross-entropy loss.
        """
        prediction = np.clip(prediction, 1e-15, 1 - 1e-15)
        return -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))

    def load_weights(self, filepath):
        """
        Loads weights from a file.

        Parameters:
            filepath (str): Path to the file containing weights.

        Returns:
            None
        """
        with open(filepath, 'rb') as f:
            self.weights = pickle.load(f)  
        pass

    def predict(self, input, threshold=0.4):
        """
        Predicts the class label for given input data.

        Parameters:
            input (numpy.ndarray): Input data.

        Returns:
            int: Predicted class label (0 or 1).
        """
        output = self.forward(input)
        if float(output) > threshold:
            return 1
        else:
            return 0
