This project an AI neural network written from scratch without any existing ai libraries. 
Behavior will be similar to TensorFlow or PyTorch.
Features will include neurons with weights and activation functions. Training via gradient descent back propagation.

This will be a feed forward network utilizing a perceptron architecture. 
The perceptron will function as a binary classifier that makes predictions based on a linear combination of input features. It takes one or more weighted inputs and returns a single binary output, either 1 or 0 

Adding these features:

Architecture Definability. 
Be able to specify the layers (input hidden output) and the activation function in each layer (ex relu)

Optimizer algo - to adjust the weights of the neural network to minimize the loss function. It determines how the model updates its weights based on the gradients computed during backpropagation.

Loss function -  measures how well the model's predictions match the actual target values. It quantifies the difference between the predicted outputs and the true outputs. The optimizer uses this loss to update the model's weights.

The gradient descent function will be similar to this: 
```
# Initialize weights and bias
w = np.random.randn(num_features)
b = np.random.randn()

# Set learning rate
alpha = 0.01

# Gradient Descent Loop
for epoch in range(num_epochs):
    # Compute predictions
    y_pred = np.dot(X, w) + b
    
    # Calculate loss (Mean Squared Error)
    loss = np.mean((y_pred - y) ** 2)
    
    # Compute gradients
    dw = (2 / len(y)) * np.dot(X.T, (y_pred - y))
    db = (2 / len(y)) * np.sum(y_pred - y)
    
    # Update weights and bias
    w -= alpha * dw
    b -= alpha * db
```
Notes on Gradient Descent: 
```
In gradient descent algorithms, determining the direction to adjust weights and biases to minimize loss involves
computing the gradient of the loss function with respect to these parameters. This gradient indicates the direction
 of the steepest ascent; thus, moving in the opposite direction reduces the loss.

Here's how this process is typically implemented in Python:

Compute Predictions:

Use the current weights (w) and bias (b) to compute the model's predictions for the input features (X).
Calculate Loss:

Evaluate the loss function (e.g., Mean Squared Error) to quantify the difference between the predictions and
the actual target values (y).
Compute Gradients:

Calculate the partial derivatives of the loss with respect to each weight and the bias. These derivatives form
the gradient vector, indicating how much the loss would change with a small change in each parameter.
Update Parameters:

Adjust the weights and bias by moving them in the direction opposite to the gradient. This step is controlled
by the learning rate (α), which determines the size of the update steps.
Mathematically, the updates for each weight (w_i) and the bias (b) can be expressed as:

Weight Update: w_i := w_i - α * ∂L/∂w_i
Bias Update: b := b - α * ∂L/∂b
Where:

∂L/∂w_i is the partial derivative of the loss function with respect to the weight w_i.
∂L/∂b is the partial derivative of the loss function with respect to the bias b.
```
