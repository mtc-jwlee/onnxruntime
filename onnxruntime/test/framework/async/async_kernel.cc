// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "async_kernel.h"
#include "async_execution_provider.h"

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
  // TODO: add actual compute code
}

AsyncKernelState::~AsyncKernelState() {
}

Status AsyncKernelState::Compute(OpKernelContext* /*op_kernel_context*/) const {
  return Status::OK();
};

}  // namespace onnxruntime