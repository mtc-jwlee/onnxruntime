// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/session/inference_session.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "async_execution_provider.h"
#include "async_kernel.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {

AsyncExecutionProvider::AsyncExecutionProvider() : IExecutionProvider{kAsyncExecutionProvider} {
  AllocatorCreationInfo device_info{
      [](int) {
        return make_unique<CPUAllocator>(OrtMemoryInfo("Async", OrtAllocatorType::OrtDeviceAllocator));
      }};
  InsertAllocator(device_info.device_alloc_factory(0));
}

std::vector<std::unique_ptr<ComputeCapability>>
AsyncExecutionProvider::GetCapability(const GraphViewer& graph,
                                      const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node : graph.Nodes()) {
    // each node is a subgraph
    std::unique_ptr<IndexedSubGraph> sub_graph = make_unique<IndexedSubGraph>();
    sub_graph->nodes.push_back(node.Index());
    auto meta_def = make_unique<IndexedSubGraph::MetaDef>();
    meta_def->name = node.Name();
    meta_def->domain = "Async";
    node.ForEachDef([&](const NodeArg& def, bool is_input) {
      if (is_input) {
        meta_def->inputs.push_back(def.Name());
      }
    });
    node.ForEachDef([&](const NodeArg& def, bool is_input) {
      if (!is_input) {
        meta_def->outputs.push_back(def.Name());
      }
    });
    meta_def->since_version = 1;
    meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
    sub_graph->SetMetaDef(std::move(meta_def));
    result.push_back(make_unique<ComputeCapability>(std::move(sub_graph)));
  }
  return result;
}

common::Status
AsyncExecutionProvider::Compile(const std::vector<Node*>& fused_nodes,
                                std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto* node : fused_nodes) {
    NodeComputeInfo info;

    // Create state function
    // This is similar to the original OpKernel constructor
    // TODO move compilation part out of create_state_func to above
    info.create_state_func =
        [&, node](ComputeContext* ctx, FunctionState* state) {
          std::unique_ptr<AsyncKernelState> s =
              onnxruntime::make_unique<AsyncKernelState>(
                  *node,
                  *ctx,
                  *this);

          *state = s.release();
          return 0;
        };

    // Release state function
    // This is similar to the original OpKernel destructor
    info.release_state_func =
        [](FunctionState state) {
          if (state)
            delete static_cast<AsyncKernelState*>(state);
        };

    // Compute function
    // This is similar to the original OpKernel's Compute()
    info.compute_func =
        [](FunctionState state, const OrtCustomOpApi*, OrtKernelContext* op_kernel_context) {
          AsyncKernelState* s = reinterpret_cast<AsyncKernelState*>(state);
          return s->Compute(reinterpret_cast<OpKernelContext*>(op_kernel_context));
        };

    node_compute_funcs.push_back(info);
  }
  return Status::OK();
}

};  // namespace onnxruntime
