import numpy as np


"""
   Input Layer       Hidden Layer       Output Layer
  (Board Positions)   (Processing)      (Decision)
        |                |                 |
   1 -->|--> H1 -------->|---------------> x
   2 -->|--> H2 -------->|---------------> y
   3 -->|--> H3 -------->|
   4 -->|--> H4 -------->|
   5 -->|--> H5 -------->|
   6 -->|--> H6 -------->|
   7 -->|                |
   8 -->|                |
   9 -->|                |

   Legend:
   --|--> represents connections between nodes
   H1-H6 are hidden nodes, actual number can vary
   x and y are output nodes
   This code:

   * Defines a simple neural network with an input layer representing the Tic-Tac-Toe board, a hidden layer for processing, and an output layer for choosing a move.
   * Uses sigmoid activation for simplicity, though ReLU or other activations might be more common in modern networks.
   * Implements a basic training mechanism using backpropagation.
   * Includes a simple function to simulate playing a game, where the neural network decides the next move based on the current board state.


    This is a basic example and would need significant expansion to handle all game scenarios, proper evaluation of win/loss/draw states, and training with a diverse dataset. For better performance, you'd integrate game theory elements or use more sophisticated learning algorithms like reinforcement learning or Monte Carlo tree search for Tic-Tac-Toe, which is a solved game.
"""

# Define the neural network
class TicTacToeNeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weights between input and hidden layers
        self.w_ih = np.random.randn(self.input_size, self.hidden_size) / np.sqrt(self.input_size)
        # Weights between hidden and output layers
        self.w_ho = np.random.randn(self.hidden_size, self.output_size) / np.sqrt(self.hidden_size)
        
        # Bias for hidden and output layers
        self.b_h = np.zeros((1, hidden_size))
        self.b_o = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Hidden layer
        self.h = self.sigmoid(np.dot(X, self.w_ih) + self.b_h)
        # Output layer
        self.o = self.sigmoid(np.dot(self.h, self.w_ho) + self.b_o)
        return self.o

    def backward(self, X, y, o):
        # Calculate error
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoid_derivative(o)

        # Hidden layer error
        self.h_error = self.o_delta.dot(self.w_ho.T)
        self.h_delta = self.h_error * self.sigmoid_derivative(self.h)

        # Update weights and biases
        self.w_ho += self.h.T.dot(self.o_delta)
        self.w_ih += X.T.dot(self.h_delta)
        self.b_h += np.sum(self.h_delta, axis=0, keepdims=True)
        self.b_o += np.sum(self.o_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for _ in range(epochs):
            # Forward pass
            o = self.forward(X)
            # Backward pass
            self.backward(X, y, o)

# Example game play and training
def play_game(net, board):
    # Convert board state to input for the network
    X = np.array(board).reshape(1, 9)
    # Get network's decision
    decision = net.forward(X)
    # Convert to move (0-8)
    move = np.argmax(decision)
    # Update board if move is valid
    if board[move] == 0:
        board[move] = 1  # Player's move (1 for simplicity)
    return board

# Example training data (simplified for demonstration)
X_train = np.array([[1,-1,0,0,0,0,0,0,0]])  # Example input: player's X in top-left, opponent's O in top-middle
y_train = np.array([[0,1,0,0,0,0,0,0,1]])  # Example output: best move might be in bottom-right

net = TicTacToeNeuralNet(9, 6, 9)  # 9 inputs for board positions, 6 hidden neurons, 9 outputs for possible moves
net.train(X_train, y_train)

# Play a game
board = [0] * 9  # Initialize empty board
board = play_game(net, board)
print("After AI's move:", board)


 