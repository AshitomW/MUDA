/**
 * @file cusparse_descriptors.hpp
 * @brief Internal cuSPARSE context and CSR/dense descriptor structures backed by HandleTable.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>

#include "cusparse.h"
#include "handle_table.hpp"

namespace muda::libraries::sparse {

struct CusparseContext {
  mutable std::mutex mutex;
  cudaStream_t stream = nullptr;
};

struct SpMatCsrDescriptor {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::int64_t nnz = 0;
  void* row_offsets = nullptr;
  void* col_indices = nullptr;
  void* values = nullptr;
  cusparseIndexType_t row_offsets_type = CUSPARSE_INDEX_32I;
  cusparseIndexType_t col_ind_type = CUSPARSE_INDEX_32I;
  cusparseIndexBase_t idx_base = CUSPARSE_INDEX_BASE_ZERO;
  cudaDataType value_type = CUDA_R_32F;
};

struct DnVecDescriptor {
  std::int64_t size = 0;
  void* values = nullptr;
  cudaDataType value_type = CUDA_R_32F;
};

struct DnMatDescriptor {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::int64_t ld = 0;
  void* values = nullptr;
  cudaDataType value_type = CUDA_R_32F;
  cusparseOrder_t order = CUSPARSE_ORDER_COL;
};

using CusparseHandleTable =
    runtime::cudart::HandleTable<CusparseContext, runtime::cudart::HandleKind::kCusparseHandle>;
using SpMatDescTable =
    runtime::cudart::HandleTable<SpMatCsrDescriptor,
                                 runtime::cudart::HandleKind::kCusparseSpMatDesc>;
using DnVecDescTable =
    runtime::cudart::HandleTable<DnVecDescriptor, runtime::cudart::HandleKind::kCusparseDnVecDesc>;
using DnMatDescTable =
    runtime::cudart::HandleTable<DnMatDescriptor, runtime::cudart::HandleKind::kCusparseDnMatDesc>;

[[nodiscard]] CusparseHandleTable& GetCusparseHandleTable() noexcept;
[[nodiscard]] SpMatDescTable& GetSpMatDescTable() noexcept;
[[nodiscard]] DnVecDescTable& GetDnVecDescTable() noexcept;
[[nodiscard]] DnMatDescTable& GetDnMatDescTable() noexcept;

}  // namespace muda::libraries::sparse

