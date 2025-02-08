#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        
        self.linear = nn.Linear(1, 1)  
    
    def forward(self, x):
        
        return self.linear(x)


x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
                  19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2]).view(-1, 1) 
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
                  16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6]).view(-1, 1)


model = LinearRegressionModel()


criterion = nn.MSELoss()


optimizer = optim.SGD(model.parameters(), lr=0.001)


epochs = 100

losses = []


for epoch in range(epochs):
    
    y_pred = model(x)
    
   
    loss = criterion(y_pred, y)
    
    
    optimizer.zero_grad() 
    loss.backward()
    
    
    optimizer.step()
    
   
    losses.append(loss.item())
    
   
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")


plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch vs Loss for Linear Regression using nn.Linear()')
plt.show()


final_w = model.linear.weight.item()  
final_b = model.linear.bias.item()
print(f"Final w (weight): {final_w}")
print(f"Final b (bias): {final_b}")


# In[ ]:




