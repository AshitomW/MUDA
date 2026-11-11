/**
 * @file curand.h
 * @brief Clean-room CUDA Random Number Generation (cuRAND) compatibility API.
 */

#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------- */
/* Types & Enums                                                              */
/* -------------------------------------------------------------------------- */

struct curandGenerator_st;
typedef struct curandGenerator_st* curandGenerator_t;

#ifndef __LIBRARY_TYPES_H__
#define __LIBRARY_TYPES_H__
typedef enum libraryPropertyType_enum {
  MAJOR_VERSION = 0,
  MINOR_VERSION = 1,
  PATCH_LEVEL = 2
} libraryPropertyType;
#endif

typedef enum curandStatus {
  CURAND_STATUS_SUCCESS = 0,
  CURAND_STATUS_VERSION_MISMATCH = 100,
  CURAND_STATUS_NOT_INITIALIZED = 101,
  CURAND_STATUS_ALLOCATION_FAILED = 102,
  CURAND_STATUS_TYPE_ERROR = 103,
  CURAND_STATUS_OUT_OF_RANGE = 104,
  CURAND_STATUS_INVALID_VALUE = 104,
  CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
  CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
  CURAND_STATUS_LAUNCH_FAILURE = 201,
  CURAND_STATUS_PREEXISTING_FAILURE = 202,
  CURAND_STATUS_INITIALIZATION_FAILED = 203,
  CURAND_STATUS_ARCH_MISMATCH = 204,
  CURAND_STATUS_INTERNAL_ERROR = 999
} curandStatus_t;

typedef enum curandRngType {
  CURAND_RNG_TEST = 0,
  CURAND_RNG_PSEUDO_DEFAULT = 100,
  CURAND_RNG_PSEUDO_XORWOW = 101,
  CURAND_RNG_PSEUDO_MRG32K3A = 121,
  CURAND_RNG_PSEUDO_MTGP32 = 141,
  CURAND_RNG_PSEUDO_MT19937 = 142,
  CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161,
  CURAND_RNG_QUASI_DEFAULT = 200,
  CURAND_RNG_QUASI_SOBOL32 = 201,
  CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,
  CURAND_RNG_QUASI_SOBOL64 = 203,
  CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204
} curandRngType_t;

typedef enum curandOrdering {
  CURAND_ORDERING_PSEUDO_BEST = 100,
  CURAND_ORDERING_PSEUDO_DEFAULT = 101,
  CURAND_ORDERING_PSEUDO_SEEDED = 102,
  CURAND_ORDERING_QUASI_DEFAULT = 201
} curandOrdering_t;

typedef enum curandDirectionVectorSet {
  CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101,
  CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102,
  CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103,
  CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104
} curandDirectionVectorSet_t;

typedef unsigned int curandDirectionVectors32_t[32];

/* -------------------------------------------------------------------------- */
/* Generator Lifecycle and Configuration                                      */
/* -------------------------------------------------------------------------- */

curandStatus_t curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type);
curandStatus_t curandCreateGeneratorHost(curandGenerator_t* generator, curandRngType_t rng_type);
curandStatus_t curandDestroyGenerator(curandGenerator_t generator);

curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream);
curandStatus_t curandGetStream(curandGenerator_t generator, cudaStream_t* stream);

curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator,
                                                  unsigned long long seed);
curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset);
curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order);

curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator,
                                                       unsigned int num_dimensions);

curandStatus_t curandGetVersion(int* version);
curandStatus_t curandGetProperty(libraryPropertyType type, int* value);

/* -------------------------------------------------------------------------- */
/* Generation Functions                                                       */
/* -------------------------------------------------------------------------- */

curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int* outputPtr, size_t num);

curandStatus_t curandGenerateUniform(curandGenerator_t generator, float* outputPtr, size_t num);
curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, double* outputPtr,
                                           size_t num);

curandStatus_t curandGenerateNormal(curandGenerator_t generator, float* outputPtr, size_t n,
                                    float mean, float stddev);
curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n,
                                          double mean, double stddev);

curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float* outputPtr, size_t n,
                                       float mean, float stddev);
curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, double* outputPtr,
                                             size_t n, double mean, double stddev);

curandStatus_t curandGeneratePoisson(curandGenerator_t generator, unsigned int* outputPtr, size_t n,
                                     double lambda);

/* Direction vector queries */
curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t* vectors[],
                                           curandDirectionVectorSet_t set);
curandStatus_t curandGetScrambleConstants32(const unsigned int** constants);

#ifdef __cplusplus
}
#endif
