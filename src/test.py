import torch
import torch.nn.functional as F
label1 = torch.tensor([1, 2, 3])
label2 = torch.tensor([1, 3, 3])
label3 = torch.tensor([2, 3, 4])
ll = torch.stack([label1, label2, label3], dim=1)
lll = []
for i in ll:
    l, c = torch.unique_consecutive(i, return_counts=True)
    print(i[c.argmax()])
    lll.append(i[c.argmax()])
print(torch.tensor(lll))

