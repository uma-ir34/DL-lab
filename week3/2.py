#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


x = torch.tensor([2.0, 4.0]).view(-1, 1)  
y = torch.tensor([20.0, 40.0]).view(-1, 1)  


w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)


learning_rate = 0.001

epochs = 2

for epoch in range(epochs):
    
    y_pred = x * w + b
    
    diff = y_pred - y
    
    
    loss = torch.mean(diff ** 2)
    
   
    if epoch > 0:
        w.grad.zero_()
        b.grad.zero_()
    
    
    loss.backward()
    
   
    print(f"Epoch {epoch + 1}:")
    print(f"  w.grad: {w.grad.item()}")
    print(f"  b.grad: {b.grad.item()}")
    
    
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
   
    print(f"  Updated w: {w.item()}")
    print(f"  Updated b: {b.item()}")
    print(f"  Loss: {loss.item()}")
    print("="*40)


# In[ ]:




