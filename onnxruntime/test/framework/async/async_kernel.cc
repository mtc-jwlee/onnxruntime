// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "async_kernel.h"
#include "async_execution_provider.h"

#include "core/providers/cpu/math/element_wise_ops.h"

namespace onnxruntime {

AsyncKernelState::AsyncKernelState(
    const Node& fused_node,
    const ComputeContext& ctx,
    const AsyncExecutionProvider& provider) : provider_(provider),
                                              ctx_(ctx) {
  const auto* func_body = fused_node.GetFunctionBody();
  ORT_ENFORCE(func_body != nullptr);
  const Graph& subgraph = func_body->Body();
  ORT_ENFORCE(subgraph.NumberOfNodes() == 1);
  op_type_ = subgraph.GetNode(0)->OpType();
}

AsyncKernelState::~AsyncKernelState() {
}

Status AsyncKernelState::Compute(OpKernelContext* op_kernel_context) const {
  if (op_type_ == "Add") {
    Add<float>::ComputeStatic(op_kernel_context);
  } else if (op_type_ == "Sub") {
    Sub<float>::ComputeStatic(op_kernel_context);
  } else if (op_type_ == "Mul") {
    Mul<float>::ComputeStatic(op_kernel_context);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "unsupported op_type_", op_type_);
  }
  return Status::OK();
};

}  // namespace onnxruntime