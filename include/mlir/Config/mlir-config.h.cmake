//===- mlir-config.h - MLIR configuration ------------------------*- C -*-===*//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* This file enumerates variables from the MLIR configuration so that they
   can be in exported headers and won't override package specific directives.
   Defining the variables here is preferable over specifying them in CMake files
   via `target_compile_definitions` because it is easier to ensure that they are
   defined consistently across all targets: They are guaranteed to be 0/1
   variables thanks to #cmakedefine01, so we can test with `#if` and find
   missing definitions or includes with `-Wundef`. With `#ifdef`, these mistakes
   can go unnoticed.

   This is a C header that can be included in the mlir-c headers. */

#ifndef MLIR_CONFIG_H
#define MLIR_CONFIG_H

/* If set, enable deprecated serialization passes. */
#cmakedefine01 MLIR_DEPRECATED_GPU_SERIALIZATION_ENABLE

/* Enable expensive checks to detect invalid pattern API usage. Failed checks
   manifest as fatal errors or invalid memory accesses (e.g., accessing
   deallocated memory) that cause a crash. Running with ASAN is recommended for
   easier debugging. */
#cmakedefine01 MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS

/* If set, greedy pattern application is randomized: ops on the worklist are
   chosen at random. For testing/debugging purposes only. This feature can be
   used to ensure that lowering pipelines work correctly regardless of the order
   in which ops are processed by the GreedyPatternRewriteDriver. This flag is
   numeric seed that is passed to the random number generator. */
#cmakedefine MLIR_GREEDY_REWRITE_RANDOMIZER_SEED ${MLIR_GREEDY_REWRITE_RANDOMIZER_SEED}

/* If set, enables PDL usage. */
#cmakedefine01 MLIR_ENABLE_PDL_IN_PATTERNMATCH

/* If set, enables CUDA-related features in CUDA-related transforms, pipelines,
   and targets. */
#cmakedefine01 MLIR_ENABLE_CUDA_CONVERSIONS

/* If set, enables features that depend on the NVIDIA's PTX compiler. */
#cmakedefine01 MLIR_ENABLE_NVPTXCOMPILER

/* If set, enables ROCm-related features in ROCM-related transforms, pipelines,
   and targets. */
#cmakedefine01 MLIR_ENABLE_ROCM_CONVERSIONS

#endif
