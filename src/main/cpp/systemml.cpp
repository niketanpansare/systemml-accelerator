/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

//#include "config.h"

// *****************************************************************
// We support Intel MKL (recommended) or OpenBLAS.
#ifndef USE_OPEN_BLAS
#define USE_INTEL_MKL
#else
#undef USE_INTEL_MKL
#endif
// *****************************************************************
#include "systemml.h"

#include <vector>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <cstring>
#include "omp.h"
#include <ctime>
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#ifdef __cplusplus
extern "C" {
#endif

#ifdef USE_OPEN_BLAS
#include <cblas.h>
extern int openblas_get_num_threads(void);
extern void openblas_set_num_threads(int MAX_NUM_THREADS);
#endif

#ifdef USE_INTEL_MKL
#include <mkl.h>
#include <mkl_service.h>
#endif

#ifdef __cplusplus
}
#endif

#define GET_DOUBLE_ARRAY(env, input) \
  env->GetDoubleArrayElements(input,NULL)
// ((double*)env->GetPrimitiveArrayCritical(input, NULL))

#define RELEASE_DOUBLE_ARRAY(env, input, inputPtr) \
  env->ReleaseDoubleArrayElements(input, inputPtr, 0)
// env->ReleasePrimitiveArrayCritical(input, inputPtr, 0)

int MAX_NUM_THREADS = -1;
int CURRENT_NUM_THREADS = -1;
void setSequentialBLAS() {
#ifdef USE_INTEL_MKL
	if(MAX_NUM_THREADS == -1) {
		MAX_NUM_THREADS = mkl_get_max_threads();
	}
	if(CURRENT_NUM_THREADS != 1) {
		CURRENT_NUM_THREADS = 1;
		mkl_set_num_threads(1);
	}
#endif

#ifdef USE_OPEN_BLAS
	if(MAX_NUM_THREADS == -1) {
		MAX_NUM_THREADS = openblas_get_num_threads();
	}
	if(CURRENT_NUM_THREADS != 1) {
		CURRENT_NUM_THREADS = 1;
		openblas_set_num_threads(1);
	}
#endif
}

void setMultiThreadedBLAS() {
#ifdef USE_INTEL_MKL
	if(MAX_NUM_THREADS == -1) {
		MAX_NUM_THREADS = mkl_get_max_threads();
	}
	if(CURRENT_NUM_THREADS != MAX_NUM_THREADS) {
		CURRENT_NUM_THREADS = MAX_NUM_THREADS;
		mkl_set_num_threads(MAX_NUM_THREADS);
	}
#endif
#ifdef USE_OPEN_BLAS
	if(MAX_NUM_THREADS == -1) {
		MAX_NUM_THREADS = openblas_get_num_threads();
	}
	if(CURRENT_NUM_THREADS != MAX_NUM_THREADS) {
		CURRENT_NUM_THREADS = MAX_NUM_THREADS;
		openblas_set_num_threads(MAX_NUM_THREADS);
	}
#endif
}

void setNumberOfThreadsBLAS(int numThreads) {
#ifdef USE_INTEL_MKL
	if(CURRENT_NUM_THREADS != numThreads) {
		CURRENT_NUM_THREADS = numThreads;
		mkl_set_num_threads(numThreads);
	}
#endif
#ifdef USE_OPEN_BLAS
	if(CURRENT_NUM_THREADS != numThreads) {
		CURRENT_NUM_THREADS = numThreads;
		openblas_set_num_threads(numThreads);
	}
#endif
}

// Multiplies two matrices m1Ptr and m2Ptr and transposes the output in row-major format of shape
// (m1rlen, m1clen) and (m1clen, m2clen)
void matmult_transpose_output(double* m1Ptr, double* m2Ptr, double* retPtr, int m1rlen,
             int m1clen, int m2clen, int numThreads) {
  setNumberOfThreadsBLAS(numThreads);
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, m2clen, m1rlen, m1clen, 1.0, m2Ptr, m2clen,
              m1Ptr, m1clen, 0.0, retPtr, m1rlen);
}

// Multiplies two matrices m1Ptr and m2Ptr in row-major format of shape
// (m1rlen, m1clen) and (m1clen, m2clen)
void matmult(double* m1Ptr, double* m2Ptr, double* retPtr, int m1rlen,
             int m1clen, int m2clen) {
  int m = m1rlen;
  int n = m2clen;
  int k = m1clen;
  setMultiThreadedBLAS();
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, m1Ptr, k,
              m2Ptr, n, 0.0, retPtr, n);
}

// Multiplies two matrices m1Ptr and m2Ptr in row-major format of shape
// (m1rlen, m1clen) and (m1clen, m2clen)
void matmult(double* m1Ptr, double* m2Ptr, double* retPtr, int m1rlen,
             int m1clen, int m2clen, int numThreads) {
  int m = m1rlen;
  int n = m2clen;
  int k = m1clen;
  setNumberOfThreadsBLAS(numThreads);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, m1Ptr, k,
              m2Ptr, n, 0.0, retPtr, n);
}

void rotate180(double* inputArray, double* outputArray, int N, int C, int H, int W,
            int K, int R, int S, int stride_h, int stride_w, int pad_h,
            int pad_w, int P, int Q) {
    int PQ = P*Q;
    int KQ = K*Q;
	for (int k = 0; k < K; k++) {
		for (int p = 0; p < P; p++) {
			for (int q = 0; q < Q; q++) {
				outputArray[p*KQ + q*K + k] = inputArray[k*PQ + p*Q + q];
			}
		}
	}
}

void col2im(double* inputArray, double* outputArray, int N, int C, int H, int W,
            int K, int R, int S, int stride_h, int stride_w, int pad_h,
            int pad_w, int P, int Q) {
	for (int p = 0; p < P; p++) {
		// h = p*stride_h + r - pad_h
		//   = r + hOffset
		// Based on restrictions: h >= 0 and r >= 0 and h < H and r < R, we get
		// max(0, - hOffset) <= r < min(R, H - hOffset)
		int hOffset = p*stride_h - pad_h;
		int rStart = MAX(0, - hOffset);
		int rEnd = MIN(R, H - hOffset);
		for (int q = 0; q < Q; q++) {
			// Using the same logic as above on following:
			// w = q*stride_w + s - pad_w
			int wOffset = q*stride_w - pad_w;
			int sStart = MAX(0, - wOffset);
			int sEnd = MIN(S, W - wOffset);
			int tempOffset = (p*Q + q)*C*R*S;
			for (int c = 0; c < C; c++) {
				int outOffset = c*H*W;
				int inputOffset = tempOffset + c*R*S;
				for (int r = rStart; r < rEnd; r++) {
					for (int s = sStart; s < sEnd; s++) {
						int inputIndex = inputOffset + r*S + s;
						int outIndex = outOffset + (hOffset + r)*W + wOffset + s;
						outputArray[outIndex] += inputArray[inputIndex];
					}
				}
			}
		}
	}
}

void im2col(double* inputArray, double* outputArray, int N, int C, int H, int W,
            int K, int R, int S, int stride_h, int stride_w, int pad_h,
            int pad_w, int P, int Q) {
  int CRS = C * R * S;
  std::size_t size = Q * sizeof(double);
  if (stride_h == 1 && stride_w == 1 && pad_h == 0 && pad_w == 0) {
    for (int c = 0; c < CRS; ++c) {
      int wOffset = c % S;
      int hOffset = (c / S) % R;
      int cInput = c / R / S;
      for (int h = 0; h < P; ++h) {
        int hPadded = h + hOffset;
        int outOffset = (c * P + h) * Q;
        int inputOffset = (cInput * H + hPadded) * W;
        std::memcpy(outputArray + outOffset, inputArray + inputOffset + wOffset,
                    size);
        int w = Q - 1;
        int wPadded = w + wOffset;
        if (hPadded < H && wPadded < W)
          outputArray[outOffset + w] = inputArray[inputOffset + wPadded];
        else
          outputArray[outOffset + w] = 0;
      }
    }
  } else {
    for (int c = 0; c < CRS; ++c) {
      int wOffset = c % S;
      int hOffset = (c / S) % R;
      int cInput = c / R / S;
      for (int h = 0; h < P; ++h) {
        int outOffset = (c * P + h) * Q;
        int hPadded = h * stride_h - pad_h + hOffset;
        int inputOffset = (cInput * H + hPadded) * W;
        if (hPadded < 0 || hPadded >= H) {
          std::memset(outputArray + outOffset, 0, size);
        } else {
          for (int w = 0; w < Q; ++w) {
            int wPadded = w * stride_w - pad_w + wOffset;
            if (wPadded >= 0 && wPadded < W)
              outputArray[outOffset + w] = inputArray[inputOffset + wPadded];
            else
              outputArray[outOffset + w] = 0;
          }
        }
      }
    }
  }
}

JNIEXPORT void JNICALL
Java_org_apache_sysml_runtime_controlprogram_CPPUtil_matrixMultDenseDense(
    JNIEnv* env, jclass cls, jdoubleArray m1, jdoubleArray m2, jdoubleArray ret,
    jint m1rlen, jint m1clen, jint m2clen, jint numThreads) {
  double* m1Ptr = GET_DOUBLE_ARRAY(env, m1);
  double* m2Ptr = GET_DOUBLE_ARRAY(env, m2);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret);

  matmult(m1Ptr, m2Ptr, retPtr, (int)m1rlen, (int)m1clen, (int)m2clen, (int)numThreads);

  RELEASE_DOUBLE_ARRAY(env, m1, m1Ptr);
  RELEASE_DOUBLE_ARRAY(env, m2, m2Ptr);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr);
}

JNIEXPORT void JNICALL Java_org_apache_sysml_runtime_controlprogram_CPPUtil_conv2dBackwardDataDense(
  JNIEnv* env, jclass, jdoubleArray filter, jdoubleArray dout,
    jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q) {
  double* filterPtr = GET_DOUBLE_ARRAY(env, filter);
  double* doutPtr = GET_DOUBLE_ARRAY(env, dout);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret);
  int CHW = (int)C * (int)H * (int)W;
  int KPQ = (int)K * (int)P * (int)Q;
  int numRotatedElem = (int)P * (int)Q * (int)K;
  int numCol2ImElem = (int)P * (int)Q * (int)C * (int)R * (int)S;

  setSequentialBLAS();
#pragma omp parallel for
  for (int n = 0; n < (int)N; n++) {
    // Step 1: Rotate dout
    double* rotatedDoutPtr = new double[numRotatedElem];
    rotate180(doutPtr + n * KPQ, rotatedDoutPtr, 1, (int)C, (int)H, (int)W, (int)K,
           (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w,
           (int)P, (int)Q);

    // Step 2: t(rotatedDout (PQ X K) %*% filter (K X CRS))
    double* col2imInput = new double[numCol2ImElem];
    matmult(rotatedDoutPtr, filterPtr, col2imInput,
            (int)P * (int)Q, (int)K, (int)C * (int)R * (int)S, 1);

    // Step 3: Perform col2im
    col2im(col2imInput, retPtr + n * CHW, 1, (int)C, (int)H, (int)W, (int)K,
           (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w,
           (int)P, (int)Q);

    delete[] rotatedDoutPtr;
    delete[] col2imInput;
  }

  RELEASE_DOUBLE_ARRAY(env, filter, filterPtr);
  RELEASE_DOUBLE_ARRAY(env, dout, doutPtr);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr);
}

JNIEXPORT void JNICALL
Java_org_apache_sysml_runtime_controlprogram_CPPUtil_conv2dDense(
    JNIEnv* env, jclass, jdoubleArray input, jdoubleArray filter,
    jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q) {
  double* inputPtr = GET_DOUBLE_ARRAY(env, input);
  double* filterPtr = GET_DOUBLE_ARRAY(env, filter);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret);
  int CHW = (int)C * (int)H * (int)W;
  int KPQ = (int)K * (int)P * (int)Q;
  int numIm2ColElem = (int)C * (int)R * (int)S * (int)P * (int)Q;

  setSequentialBLAS();
#pragma omp parallel for
  for (int n = 0; n < (int)N; n++) {
    double* loweredMat = new double[numIm2ColElem];

    // Step 1: Perform im2col
    im2col(inputPtr + n * CHW, loweredMat, 1, (int)C, (int)H, (int)W, (int)K,
           (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w,
           (int)P, (int)Q);

    // Step 2: filter (K X CRS) %*% loweredMat (CRS X PQ)
    matmult(filterPtr, loweredMat, retPtr + n * KPQ, (int)K,
            (int)C * (int)R * (int)S, (int)P * (int)Q, 1);

    delete[] loweredMat;
  }

  RELEASE_DOUBLE_ARRAY(env, input, inputPtr);
  RELEASE_DOUBLE_ARRAY(env, filter, filterPtr);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr);
}