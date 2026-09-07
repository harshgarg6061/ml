import torch
import numpy as np

x=torch.tensor([[1.0,1.0],[5.0,1.0],[10.0,1.0],[15.0,1.0],[20.0,1.0]],dtype=torch.float32)
y=torch.tensor([2.0,8.0,22.0,31.2,43.3],dtype=torch.float32)
w=torch.tensor([0.0,0.0],dtype=torch.float32,requires_grad=True)

def forward(x,w):
    return x@w

def loss(x,w,y,n):
    y_pred=forward(x,w)
    y_=y-y_pred
    ans=0
    for i in range(n):
        ans+=y_[i]**2
    return ans/n    

def nesterov(x,w,y,n,epochs,alpha,lr):
    v=torch.tensor([0.0,0.0],dtype=torch.float32)
    for i in range(epochs):
        w_=w+alpha*v
        l=loss(x,w_,y,n)
        if(i%100==0):
            print(f"loss on {i}th epoch is {l.item():.4f}\n")
        l.backward()
        with torch.no_grad():
            v=alpha*v-lr*w.grad
            w+=v
            w.grad.zero_()
    return w

lr=0.001
alpha=0.9
epochs=2000
w_final=nesterov(x,w,y,5,epochs,alpha,lr)
print(f"w after {epochs} is ")
for i in range(2):
    print(f"{w_final[i].item():.4f} ")
