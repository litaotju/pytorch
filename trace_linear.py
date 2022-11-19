import torch 
from torch import nn

class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(3, 4, bias=True),
            nn.ELU(),
            nn.Linear(4, 5, bias=True),
            nn.Softsign(),
            nn.Linear(5, 6, bias=True),
            nn.ReLU6(),
            nn.Linear(6, 7, bias=True),
        )

    def forward(self,x):
        x.requires_grad_(True)
        y = self.linears(x)
        y = y.sum()
        dx = torch.autograd.grad(y, x)[0]
        return x-dx

def freeze_module(module):
    for p in module.parameters():
        p.requires_grad_(False)

model = MyModule()
freeze_module(model)

torch.onnx.export(model, torch.ones([1, 3]) , "linears.onnx", 
    verbose=False, input_names=["in"], output_names=["out"],
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN
)