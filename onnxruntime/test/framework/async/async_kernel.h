// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"
#include "core/framework/func_kernel.h"

namespace onnxruntime {

class AsyncExecutionProvider;
class OpKernel;

class AsyncKernelState {
 public:
  explicit AsyncKernelState(
      const Node& fused_node,
      const ComputeContext& ctx,
      const AsyncExecutionProvider& provider);

  ~AsyncKernelState();

  Status Compute(OpKernelContext* op_kernel_context) const;

 private:
  const AsyncExecutionProvider& provider_;

  // Here ComputeContext of Ort is used for allocator
  ComputeContext ctx_;   // the compute context from IExecutionProvider::Compile interface
  std::string op_type_;  // op type of the compute, since each subgraph only contains one node for now
};

}  // namespace onnxruntime