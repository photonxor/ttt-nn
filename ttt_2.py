import numpy as np

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
        self.h = self.sigmoid(np.dot(X, self.w_ih) + self.b_h)
        self.o = self.sigmoid(np.dot(self.h, self.w_ho) + self.b_o)
        return self.o

    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoid_derivative(o)
        self.h_error = self.o_delta.dot(self.w_ho.T)
        self.h_delta = self.h_error * self.sigmoid_derivative(self.h)

        self.w_ho += self.h.T.dot(self.o_delta)
        self.w_ih += X.T.dot(self.h_delta)
        self.b_h += np.sum(self.h_delta, axis=0, keepdims=True)
        self.b_o += np.sum(self.o_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for _ in range(epochs):
            o = self.forward(X)
            self.backward(X, y, o)

def print_board(board):
    for row in [board[i:i+3] for i in range(0, 9, 3)]:
        print('|' + '|'.join([' X ' if x == 1 else ' O ' if x == -1 else '   ' for x in row]) + '|')
    print('-' * 11)

def play_game(net, board):
    print("\nBoard Before AI's Move:")
    print_board(board)
    
    # Convert board state to input for the network
    X = np.array(board).reshape(1, 9)
    # Get network's decision
    decision = net.forward(X)
    # Convert to move (0-8)
    move = np.argmax(decision)
    
    # Update board if move is valid
    if board[move] == 0:
        board[move] = 1  # Player's move (1 for simplicity)
    
    print("\nBoard After AI's Move:")
    print_board(board)
    
    return board

# Example training data (as provided earlier)
X_train = np.array([
    [1,-1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1],
    [-1,1,-1,0,0,0,0,0,0],
    [0,0,0,0,0,-1,0,1,0],
    [0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,-1,0,0,0],
    [0,0,0,0,0,0,0,1,-1],
    [-1,0,1,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0],
    [-1,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,1,0,0],
    [0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,-1,1,0,0],
    [-1,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0],
    [1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,-1,0,1],
    [0,0,0,0,0,1,0,0,-1],
])

y_train = np.array([
    [0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
])



# Initialize and train the neural network
net = TicTacToeNeuralNet(input_size=9, hidden_size=6, output_size=9)
net.train(X_train, y_train)

# Example usage to see the network's output for a given input
example_input = X_train[0].reshape(1, -1)
prediction = net.forward(example_input)
print("Prediction for the first example:", prediction)

# Example usage:
board = [0] * 9  # Initialize empty board

net = TicTacToeNeuralNet(input_size=9, hidden_size=6, output_size=9)
# Assuming net has been trained as per previous instructions
net.train(X_train, y_train)
board = play_game(net, board)

