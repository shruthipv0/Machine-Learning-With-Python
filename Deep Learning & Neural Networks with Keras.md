# Neural Networks

## What is a Neural Network?

- It is a machine learning models that mimic the complex functions of the human brain.
- It consists of interconnected nodes or neurons that process data, learn patterns, and enable tasks such as pattern recognition and decision-making.

### Learning in neural networks follows a structured, three-stage process:

1. Input Computation: Data is fed in.  
2. Output Generation: Output based on given parameters.
3. Iterative Refinement: Output refinement by adjusting weights and biases.

![image](https://github.com/user-attachments/assets/e2f7cd68-0b1e-48b2-99ff-314743cfe4a5)

### Layers in Neural Network Architecture

1. Input
2. Hidden
3. Output

![image](https://github.com/user-attachments/assets/8f9f2857-a624-4341-8233-548bd8e3faf8)

## Working of Neural Networks

### Forward Propagation

When data is input into the network, it passes through the network in the __forward direction__, from the input layer through the hidden layers to the output layer.

__Linear Transformation__ : Each neuron in a layer receives inputs, which are multiplied by the weights associated with the connections. These products are summed together, and a bias is added to the sum. 

![image](https://github.com/user-attachments/assets/84513627-74c9-47d0-a4c6-8735a5929a2f)

__Activation__ : The result of the linear transformation is then passed through an activation function. It introduces non-linearity into the system because we linear functions cant capture the complexity of the data ,e.g, y = x^2 can't be done linearly. Popular activation functions include ReLU, sigmoid, and tanh.

![image](https://github.com/user-attachments/assets/dfd57b8d-d2dc-4df0-a63b-c734d12cab3f)

### Backpropagation

After forward propagation,model evaluates its performance using loss function(y_test, y_predict). Goal is to minimise this function. 

__Loss Calculation__ : Calculating loss function

__Gradient Calculation__ : Calculating the rate of change of the loss function with respect to each parameter (weight and bias). So how much the loss function varies if we vary the weight or bias.
![image](https://github.com/user-attachments/assets/87c95be9-fe22-484c-ac8c-5639c5c67871)

__Weight Update__ : Once the gradients are calculated, the weights and biases are updated using an optimization algorithm like stochastic gradient descent (SGD). 
![image](https://github.com/user-attachments/assets/247d6186-3ef4-45d9-96b5-5551bc000c24)
![image](https://github.com/user-attachments/assets/fe9deb9a-3fa7-44e0-8839-ce2f06de3e53)


