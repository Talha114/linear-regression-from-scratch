# Linear Regression from Scratch

Regression is a method used to model a target value based on independent predictors. Linear regression is a type of regression analysis where there is a linear relationship between the independent variable(s) and dependent variable. The goal of linear regression is to find the best values for the coefficients, which results in the line that best fits the data.

## Concepts

### Cost Function

The cost function helps us to figure out the best possible values for the coefficients, which would provide the best fit line for the data points. Since we want the best values for the coefficients, we convert this search problem into a minimization problem where we would like to minimize the error between the predicted value and the actual value. We choose the Mean Squared Error(MSE) function to minimize the error difference between the predicted values and the ground truth. The MSE is calculated by summing the squared difference between the predicted and actual values over all data points and dividing by the total number of data points.

### Gradient Descent

The idea of gradient descent is that we start with some values for the coefficients and then we change these values iteratively to reduce the cost. Gradient descent helps us to update the coefficients. To update the coefficients, we take gradients from the cost function. To find these gradients, we take partial derivatives with respect to each coefficient.

## Algorithm

1. Pick initial values of the coefficients randomly.
2. Calculate the predicted value using the hypothesis of the line equation.
3. Find the loss on your prediction.
4. Find the rate of change of the error (using the derivative).
5. Update the coefficients using the following equation: new coefficients = old coefficients - learning rate * derivative of the cost function.
6. Repeat steps 2-5 until the termination criteria are met.

### Termination Criteria

1. After a certain number of iterations.
2. When the value of the cost function between two iterations is less than a certain threshold.

## Implementation

To implement linear regression from scratch, we need to define the following functions:

### `hyp(theta, X)`

This function implements the hypothesis for the equation of the line.

### `cost_function(theta, X, Y)`

This function implements the cost function (Mean Squared Error) for linear regression training.

### `derivative_cost_function(theta, X, Y)`

This function implements the derivative of the cost function (error function) to find the rate of change of error in regression.

### `GradientDescent(X, Y, cost_function, derivative_cost_function, max_iter)`

This function implements the gradient descent algorithm to train the linear regression model.

```python
def GradientDescent(X, Y, cost_function, derivative_cost_function, max_iter):
    # Initialize coefficients randomly
    theta = np.random.rand(X.shape[1])
    alpha = 0.01
    for i in range(max_iter):
        # Calculate the predicted value using the hypothesis of the line equation
        hyp = np.dot(X, theta)
        # Find the loss on your prediction
        loss = hyp - Y
        # Find the cost function
        cost = cost_function(theta, X, Y)
        # Find the rate of change of the error (using the derivative)
        gradient = derivative_cost_function(theta, X, Y)
        # Update the coefficients
        theta = theta - alpha * gradient
    return theta
```

With the above functions, we can train a linear regression model on our dataset.

## Conclusion

In this article, we have learned about linear regression and its implementation from scratch. We have covered the concepts of cost function and gradient descent. We have also provided an implementation of linear regression in Python using NumPy.
