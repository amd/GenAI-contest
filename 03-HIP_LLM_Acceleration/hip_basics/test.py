import torch
import demo.backend as db

mat = torch.Tensor(torch.randn(128,4)).half().to("cuda:0")
vec = torch.Tensor(torch.randn(4,1)).half().to("cuda:0")
res = torch.Tensor(torch.randn(128,1)).half().to("cuda:0")

db.gemv(mat, vec, res)

print("mat:", mat)
print("vec:", vec)
print("res:", res)
