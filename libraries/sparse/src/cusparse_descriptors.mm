/**
 * @file cusparse_descriptors.mm
 * @brief Implementation of cuSPARSE context and generic CSR/dense descriptor lifecycle.
 */

#include "cusparse_descriptors.hpp"

#include "muda/libraries/metal_executor.hpp"

namespace muda::libraries::sparse {
namespace {

constexpr int kCusparseVersion = 12000;

[[nodiscard]] constexpr bool IsSupportedValueType(cudaDataType dt) noexcept {
  return dt == CUDA_R_32F || dt == CUDA_R_16F;
}

}  // namespace

CusparseHandleTable& GetCusparseHandleTable() noexcept {
  static CusparseHandleTable table;
  return table;
}

SpMatDescTable& GetSpMatDescTable() noexcept {
  static SpMatDescTable table;
  return table;
}

DnVecDescTable& GetDnVecDescTable() noexcept {
  static DnVecDescTable table;
  return table;
}

DnMatDescTable& GetDnMatDescTable() noexcept {
  static DnMatDescTable table;
  return table;
}

}  // namespace muda::libraries::sparse

using namespace muda::libraries;
using namespace muda::libraries::sparse;

extern "C" {

cusparseStatus_t cusparseCreate(cusparseHandle_t* handle) {
  if (handle == nullptr) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  if (LibraryExecutor::Instance().EnsureInitialized() != cudaSuccess) {
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  }
  auto ctx = std::make_shared<CusparseContext>();
  *handle = GetCusparseHandleTable().Insert<cusparseHandle_t>(std::move(ctx));
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) {
  return (GetCusparseHandleTable().Remove(handle) != nullptr) ? CUSPARSE_STATUS_SUCCESS
                                                              : CUSPARSE_STATUS_NOT_INITIALIZED;
}

cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int* version) {
  if (version == nullptr) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  if (handle != nullptr && GetCusparseHandleTable().Lookup(handle) == nullptr) {
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  }
  *version = kCusparseVersion;
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId) {
  auto ctx = GetCusparseHandleTable().Lookup(handle);
  if (ctx == nullptr) {
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  }
  if (!LibraryExecutor::Instance().IsValidStream(streamId)) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->stream = streamId;
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId) {
  if (streamId == nullptr) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  auto ctx = GetCusparseHandleTable().Lookup(handle);
  if (ctx == nullptr) {
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  *streamId = ctx->stream;
  return CUSPARSE_STATUS_SUCCESS;
}

const char* cusparseGetErrorName(cusparseStatus_t status) {
  switch (status) {
    case CUSPARSE_STATUS_SUCCESS:
      return "CUSPARSE_STATUS_SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "CUSPARSE_STATUS_ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE:
      return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "CUSPARSE_STATUS_MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSPARSE_STATUS_ZERO_PIVOT:
      return "CUSPARSE_STATUS_ZERO_PIVOT";
    case CUSPARSE_STATUS_NOT_SUPPORTED:
      return "CUSPARSE_STATUS_NOT_SUPPORTED";
    case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
      return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
  }
  return "CUSPARSE_STATUS_UNKNOWN";
}

const char* cusparseGetErrorString(cusparseStatus_t status) { return cusparseGetErrorName(status); }

cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols,
                                   int64_t nnz, void* csrRowOffsets, void* csrColInd,
                                   void* csrValues, cusparseIndexType_t csrRowOffsetsType,
                                   cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase,
                                   cudaDataType valueType) {
  if (spMatDescr == nullptr || rows < 0 || cols < 0 || nnz < 0 || csrRowOffsets == nullptr ||
      (nnz > 0 && (csrColInd == nullptr || csrValues == nullptr))) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  if (csrRowOffsetsType != CUSPARSE_INDEX_32I || csrColIndType != CUSPARSE_INDEX_32I) {
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  }
  if (idxBase != CUSPARSE_INDEX_BASE_ZERO && idxBase != CUSPARSE_INDEX_BASE_ONE) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  if (!IsSupportedValueType(valueType)) {
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  }

  auto desc = std::make_shared<SpMatCsrDescriptor>(SpMatCsrDescriptor{
      .rows = rows,
      .cols = cols,
      .nnz = nnz,
      .row_offsets = csrRowOffsets,
      .col_indices = csrColInd,
      .values = csrValues,
      .row_offsets_type = csrRowOffsetsType,
      .col_ind_type = csrColIndType,
      .idx_base = idxBase,
      .value_type = valueType,
  });
  *spMatDescr = GetSpMatDescTable().Insert<cusparseSpMatDescr_t>(std::move(desc));
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr) {
  return (GetSpMatDescTable().Remove(spMatDescr) != nullptr) ? CUSPARSE_STATUS_SUCCESS
                                                             : CUSPARSE_STATUS_INVALID_VALUE;
}

cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, void* values,
                                     cudaDataType valueType) {
  if (dnVecDescr == nullptr || size < 0 || (size > 0 && values == nullptr)) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  if (!IsSupportedValueType(valueType)) {
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  }
  auto desc = std::make_shared<DnVecDescriptor>(DnVecDescriptor{
      .size = size,
      .values = values,
      .value_type = valueType,
  });
  *dnVecDescr = GetDnVecDescTable().Insert<cusparseDnVecDescr_t>(std::move(desc));
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr) {
  return (GetDnVecDescTable().Remove(dnVecDescr) != nullptr) ? CUSPARSE_STATUS_SUCCESS
                                                             : CUSPARSE_STATUS_INVALID_VALUE;
}

cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols,
                                     int64_t ld, void* values, cudaDataType valueType,
                                     cusparseOrder_t order) {
  if (dnMatDescr == nullptr || rows < 0 || cols < 0 || (rows > 0 && cols > 0 && values == nullptr)) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  if (order != CUSPARSE_ORDER_COL && order != CUSPARSE_ORDER_ROW) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  const int64_t min_ld = (order == CUSPARSE_ORDER_COL) ? (rows > 1 ? rows : 1)
                                                       : (cols > 1 ? cols : 1);
  if (ld < min_ld) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  if (!IsSupportedValueType(valueType)) {
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  }
  auto desc = std::make_shared<DnMatDescriptor>(DnMatDescriptor{
      .rows = rows,
      .cols = cols,
      .ld = ld,
      .values = values,
      .value_type = valueType,
      .order = order,
  });
  *dnMatDescr = GetDnMatDescTable().Insert<cusparseDnMatDescr_t>(std::move(desc));
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr) {
  return (GetDnMatDescTable().Remove(dnMatDescr) != nullptr) ? CUSPARSE_STATUS_SUCCESS
                                                             : CUSPARSE_STATUS_INVALID_VALUE;
}

}  // extern "C"

