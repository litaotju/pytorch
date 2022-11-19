import torch

import torch.nn.functional as F

# F.gelu has issues in the exporter, no the tracer, skip it for now.

# From tools/autograd/derivatives.yaml
# expect to work, since the backward only use "self"
working_activations = (
    F.relu,
    F.relu6,
    F.elu,
    F.selu,
    F.celu,
    F.logsigmoid,
    F.softsign,
    F.softplus,
    torch.nn.LeakyReLU(inplace=False),
)

# expect to fail, since the backward use "result"
failing_activations = (
    F.sigmoid, 
    F.tanh,
    torch.nn.LeakyReLU(inplace=True),
)


class MyModule(torch.nn.Module):
    def __init__(self, f):
        super(MyModule, self).__init__()
        self.f = f

    def forward(self,x):
        x.requires_grad_(True)
        y = self.f(x)
        y = y.sum()
        dx = torch.autograd.grad(y, x)[0]
        return x-dx

def test_activation(f):
    try:
        ### torch.onnx.OperatorExportTypes.ONNX_ATEN is needed, since the threshold_backward is standard onnx op set.
        ### Error is: "Exporting the operator 'aten::threshold_backward' to ONNX opset version 14 is not supported"
        torch.onnx.export(MyModule(f), torch.ones([1, 4]) , "act.onnx", 
           verbose=False, input_names=["in"], output_names=["out"], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN)
    except Exception as e:
        print("Exception: ", e)
        print(f)
        raise e

for f in working_activations:
    test_activation(f)

for f in failing_activations:
    try:
        test_activation(f)
    except Exception as e:
        continue
    raise Exception("Expect to fail, but it works: ", f)