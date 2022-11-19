import torch 
from torch import nn

class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self,x):
        '''
        Error: "Cannot insert a Tensor that requires grad as a constant. " when
            torch.jit.trace trying to trace a autograd.grad function all.
        From the message, this was due to TracingState::getValue can not find a Value which it should be there.
        But why it can not find the value??

        Lets work step by step.

        1. conceptually, relu's backward function is threshold_backward(grad, result, 0), see tools/autograd/derivatives.yaml

        2. When the forward function is called, one "grad_fn" object will be attached to the tensor, and the 
           necessary input, output tensors of the forward pass will be **saved to be reused later** by attaching
           "SavedVariable" objects to the "grad_fn" object.

          See torch/csrc/autograd/generated/VariableType_4.cpp relu function
        ```cpp
            at::Tensor relu(c10::DispatchKeySet ks, const at::Tensor & self) {
                  if (_any_requires_grad) {
                  grad_fn = std::shared_ptr<ReluBackward0>(new ReluBackward0(), deleteNode);
                  grad_fn->set_next_edges(collect_next_edges( self ));
                }
                // forward to get the result
                result = at::relu(self);

                grad_fn->result_ = SavedVariable(result, true/*is_output*/);
            }
        ```
        and, in trace mode, at::relu will stores both the "self" and the "result" to the TracingState.

        When the grad_fn is called by user triggering "torch.autograd.grad" function, the grad_fn::apply will
        **reuse the saved tensor of the forward pass**, by calling SavedVariable::unpack
        See torch/csrc/autograd/generated/Functions.cpp
        ```
        variable_list ReluBackward0::apply(variable_list&& grads) {
            ...
            auto result = result_.unpack(shared_from_this());
            ..
            auto grad_result = any_grad_defined ? (threshold_backward(grad, result, 0)) : Tensor();
        }
        ```

        3. In trace mode, the threshold_backward tries to get saved "result" from the "TracingState" and reuse it.
        Conceptually, this should work whether the saved tensors are input or output tensors of the forward pass.
        And tensors should be the exact object stored in the forward pass, such that TracingState can find it,
        and for the saved tensor, the TracingState::getValue should be able find the tensor from saved env_stack, and early return.
        See  torch/csrc/jit/frontend/tracer.cpp

        ```
        Value* TracingState::getValue(const IValue& var) {

            for (const auto i : c10::irange(env_stack.size())) {
              auto& value_map = env_stack.at(env_stack.size() - 1 - i);
              auto it = value_map.find(var);
              if (it == value_map.end()) {
                continue;
              }
              .. .
              return it->second;
            }

            if (ten.requires_grad()) {
              pauseTracing();
              std::ostringstream oss;
              oss << "Cannot insert a Tensor that requires grad as a constant. "
            }
        }
        ```

        But when one saved tensor is the output of a forward layer (relu here), the TracingState::getValue can not find it.
        So it's suspected that "result = result_.unpack(...)" returns a different Value object than the one stored in the forward pass.
        By inspecting the object pointer, it's confirmed that the two Value objects are different.

        See the code for further investigation, the "saved_original_" is False, and when it's False, a new Variable is created.
        ```cpp
            Variable SavedVariable::unpack(std::shared_ptr<Node> saved_for) const {
            {
                ...
                  if (!hooks_ && saved_original_) {
                    return data_;
                  }
                ...
                  Variable var;
                  if (grad_fn) {
                    var = make_variable(data, Edge(std::move(grad_fn), output_nr_));
                  } else {
                    var = make_variable(data, requires_grad_);
                  }
                ...
                return var;
            }
        ```
        The "saved_original_" is set to False in the SavedVariable constructor, see torch/csrc/autograd/saved_variable.cpp
        But why? needs to back the ctor code. When "is_output" is False, the "saved_original_" is set to True, otherwise False.

        ```
        SavedVariable::SavedVariable(
            const Variable& variable,
            bool is_output,
            bool is_inplace_on_view) {

            if (!is_output || is_leaf_) {
              saved_original_ = true;
              data_ = variable;
              return;
            }
        }

        ```

        Let's back to the relu function, the "is_output" is True, so the "saved_original_" is False.
        If we change the "is_output" to False, the "saved_original_" is True, and the TracingState::getValue can find the Value object.
        How to change the "is_output" to False?  We can hack the relu's grad function to use relu's input (self) other than the output (result).
            Use self: when relu's input is >0, the grad is 1, other wise, grad is 0.
            Use output: when relu's output is >0, the grad is 1, other wise, grad is 0.
        So it's the same.  so we changed the tools/autograd/derivatives.yaml, 
        ```yaml
           - name: relu(Tensor self) -> Tensor
             self: threshold_backward(grad, self, 0)
            
        **And turns out this hack works for the torch.onnx.export, but not for the torch.jit.trace.**
        Need to figure out why torch.jit.trace does ntwork with this hack.
        ``` 

        This also explains why the "clamp" and "linear" works. Both clamp and linear backward function
        use the input tensor as the saved tensor, and the TracingState::getValue can find the Value object.
        See tools/autograd/derivatives.yaml
        ```yaml
         - name: clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
           self: clamp_backward(grad, self, min, max)
           result: auto_element_wise

        - name: linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
          input, weight, bias: linear_backward(input, grad, weight, grad_input_mask)
        ```

        To confirm the above hypothesis, 
            chose some other activation, with some use the self while others use result, see what happens.
        
        TODO: why the two linear does not work?
        '''

        x.requires_grad_(True)
        y = torch.relu(x)
        y = y.sum()
        dx = torch.autograd.grad(y, x)[0]
        return x-dx

## TODO:
## Even after the WAR to replace relu backward with the threshold_backward, which only ref the relu's input tensor.
## and does not requires relu's output, the torch.jiut.trace_module still fails, there is a new error other than
## The new error is: "element 0 of tensors does not require grad and does not have a grad_fn"

# torch.jit.trace_module(
#         MyModule(), 
#         {'forward': torch.ones([1,4])}
#         )

## torch.onnx.OperatorExportTypes.ONNX_ATEN is needed, since the threshold_backward is standard onnx op set.
## Error is: "Exporting the operator 'aten::threshold_backward' to ONNX opset version 14 is not supported"
torch.onnx.export(MyModule(), torch.ones([1, 4]) , "relu.onnx", 
    verbose=True, input_names=["in"], output_names=["out"], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN)