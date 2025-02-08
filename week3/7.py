#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        
        self.linear = nn.Linear(1, 1)  
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))  

x = torch.tensor([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=torch.float32).view(-1, 1)  
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32).view(-1, 1)  


model = LogisticRegressionModel()


criterion = nn.BCELoss()


optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000

losses = []


for epoch in range(epochs):
   
    y_pred = model(x)
    
    
    loss = criterion(y_pred, y)
    
    
    optimizer.zero_grad() 
    loss.backward()  
    
    
    optimizer.step()
    
    
    losses.append(loss.item())
    
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")


plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch vs Loss for Logistic Regression')
plt.show()


final_weight = model.linear.weight.item()
final_bias = model.linear.bias.item()

print(f"Final Weight: {final_weight}")
print(f"Final Bias: {final_bias}")



X_test = torch.tensor([[30.0]], dtype=torch.float32)
y_test_pred = model(X_test)

print(f"Prediction for X = 30: {y_test_pred.item()}")


# In[ ]:




