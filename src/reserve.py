import torch 
import time 


a = torch.ones((500,500)).to("cuda:0") 
time.sleep(1000)
