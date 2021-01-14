# -*- coding: utf-8 -*-
import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.


dtype = torch.float
device = torch.device('cpu')
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, dtype=dtype, device=device)
y = torch.randn(N, D_out, dtype=dtype, device=device)

# Randomly initialize weights
w1 = torch.randn(D_in, H, dtype=dtype, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, dtype=dtype, device=device, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    # if t % 100 == 99:
    print(t, loss.item())

    # Backprop to compute gradients of w1 and w2 with respect to loss
    loss.backward()

    # Update weights
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
