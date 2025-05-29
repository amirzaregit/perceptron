import random

# Dataset: points with bias term (1), x, y coordinates
input_values = [
    (1, 6.39, 7.89),
    (1, 0.25, 3.87),
    (1, 8.92, 9.88),
    (1, 4.21, 6.19),
    (1, 1.07, 7.32),
    (1, 9.31, 9.96),
    (1, 2.79, 8.67),
    (1, 0.38, 8.35),
    (1, 6.86, 8.11),
    (1, 5.19, 7.06),
    (1, 4.87, 7.52),
    (1, 3.66, 8.19),
    (1, 0.79, 9.03),
    (1, 5.68, 9.06),
    (1, 7.29, 8.81),
    (1, 4.45, 2.89),
    (1, 8.01, 6.93),
    (1, 7.83, 4.55),
    (1, 5.92, 4.23),
    (1, 3.14, 0.57),
    (1, 9.65, 7.22),
    (1, 6.18, 5.02),
    (1, 4.08, 3.62),
    (1, 7.79, 2.54),
    (1, 5.77, 1.49),
    (1, 3.95, 1.31),
    (1, 9.11, 8.84),
    (1, 8.44, 6.91),
    (1, 6.53, 3.99),
    (1, 7.02, 2.86)
]

# Corresponding labels:
# 1 means point is above the line y = x
# -1 means point is below the line y = x
output_values = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # Above line y=x
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1  # Below line y=x
]

# Initial weights for bias, x, and y inputs
weights = [1, 1, 1]

# Learning rate controls how much weights change each update
LEARNING_RATE = 0.1

# Create a list of indices for training samples to shuffle during training
index_list = list(range(len(input_values)))

# Function to print current weights nicely
def show_weights(weights):
    print(f"w0={weights[0]:.3f}, w1={weights[1]:.3f}, w2={weights[2]:.3f}")

show_weights(weights)

# Perceptron prediction function
# Calculates weighted sum z = w0*1 + w1*x + w2*y
# Returns 1 if z > 0 else -1
def perceptron(inputs, weights):
    z = 0.0
    for i in range(len(inputs)):
        z += inputs[i] * weights[i]
    
    return 1 if z > 0 else -1

# Training loop: keep iterating until all points are classified correctly
correct = False

while not correct:
    correct = True  # Assume all are correct initially
    random.shuffle(index_list)  # Shuffle training order each epoch
    
    # Iterate over each sample index in random order
    for i in index_list:
        input_set = input_values[i]
        expected_output = output_values[i]
        
        predicted_output = perceptron(input_set, weights)
        
        # If prediction is wrong, update weights
        if predicted_output != expected_output:
            for wi in range(len(weights)):
                # Update rule: w = w + learning_rate * (expected - predicted) * input
                weights[wi] += expected_output * LEARNING_RATE * input_set[wi]
            correct = False  # At least one error, need another epoch
            show_weights(weights)  # Show weights after update

# Test predictions after training on some points
print(perceptron(inputs=input_values[0], weights=weights))  # Should be 1 (above line)
print(perceptron(inputs=input_values[1], weights=weights))  # Should be 1
print(perceptron(inputs=input_values[2], weights=weights))  # Should be 1
print(perceptron(inputs=input_values[3], weights=weights))  # Should be 1

# Test on new points not in training set
print(perceptron(inputs=(1, 4.9, 1.1) , weights=weights))   # Expect -1 
