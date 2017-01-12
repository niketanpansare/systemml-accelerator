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
  ((double*)env->GetPrimitiveArrayCritical(input, NULL))
// env->GetDoubleArrayElements(input,NULL)

#define RELEASE_DOUBLE_ARRAY(env, input, inputPtr) \
  env->ReleasePrimitiveArrayCritical(input, inputPtr, 0)
// env->ReleaseDoubleArrayElements(input, inputPtr, 0)

int CURRENT_NUM_THREADS = -1;
void ensureSequentialBLAS() {
/*
#ifdef USE_INTEL_MKL
	if(CURRENT_NUM_THREADS != 1) {
	    #ifdef USE_MKL_THREADING_GNU
			mkl_set_threading_layer(MKL_THREADING_GNU)
		#else
			mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL)
		#endif
		CURRENT_NUM_THREADS = 1;
		mkl_set_num_threads(1);
	}
#endif
*/
#ifdef USE_OPEN_BLAS
	if(CURRENT_NUM_THREADS != 1) {
		CURRENT_NUM_THREADS = 1;
		openblas_set_num_threads(1);
	}
#endif
}

// -----------------------------------------------------------------------------------------

// Multiplies two matrices m1Ptr and m2Ptr in row-major format of shape
// (m1rlen, m1clen) and (m1clen, m2clen)
void matmult(double* m1Ptr, double* m2Ptr, double* retPtr, int m1rlen,
             int m1clen, int m2clen, int numThreads) {
  int m = m1rlen;
  int n = m2clen;
  int k = m1clen;
  if(numThreads == 1) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, m1Ptr, k,
              m2Ptr, n, 0.0, retPtr, n);
  }
  else {
#ifdef USE_OPEN_BLAS
  openblas_set_num_threads(numThreads);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, m1Ptr, k,
          m2Ptr, n, 0.0, retPtr, n);
  openblas_set_num_threads(1);
#else
	#ifdef USE_MKL_THREADING_GNU
    	mkl_set_num_threads(numThreads);
  		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, m1Ptr, k,
          m2Ptr, n, 0.0, retPtr, n);
  		mkl_set_num_threads(1);
	#else
		// Row-wise parallelism - suboptimal but avoids internal BLAS threading issues with Java/OpenMP threading
    	// See https://software.intel.com/en-us/node/528707
#pragma omp parallel for num_threads(numThreads)
	    for(int i = 0; i < m1rlen; i++) {
	      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, n, k, 1.0, m1Ptr + i*m1clen, k,
	              m2Ptr, n, 0.0, retPtr + i*m2clen, n);
	    }
	#endif
#endif
  }
}
// -----------------------------------------------------------------------------------------

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
  // First step:  Avoids oversubscription and other openmp/internal blas threading issues
  ensureSequentialBLAS();
  
  double* m1Ptr = GET_DOUBLE_ARRAY(env, m1);
  double* m2Ptr = GET_DOUBLE_ARRAY(env, m2);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret);

  matmult(m1Ptr, m2Ptr, retPtr, (int)m1rlen, (int)m1clen, (int)m2clen, (int)numThreads);

  RELEASE_DOUBLE_ARRAY(env, m1, m1Ptr);
  RELEASE_DOUBLE_ARRAY(env, m2, m2Ptr);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr);
}

JNIEXPORT void JNICALL Java_org_apache_sysml_runtime_controlprogram_CPPUtil_conv2dBackwardFilterDense(
  JNIEnv* env, jclass, jdoubleArray input, jdoubleArray dout,
    jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q) {
  // First step: Avoids oversubscription and other openmp/internal blas threading issues
  ensureSequentialBLAS();
  
  double* inputPtr = GET_DOUBLE_ARRAY(env, input);
  double* doutPtr = GET_DOUBLE_ARRAY(env, dout);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret);
  
  int CHW = (int)C * (int)H * (int)W;
  int KPQ = (int)K * (int)P * (int)Q;
  int numRotatedElem = (int)P * (int)Q * (int)K;
  int numIm2ColElem = (int)C * (int)R * (int)S * (int)P * (int)Q;
  int numTempElem = (int)C * (int)R * (int)S * (int)K;
  
  double* loweredMatArrays;
  double* rotatedDoutPtrArrays;
  double* temp; 
  
  int m1 = (int)C * (int)R * (int)S;
  int n1 = (int)K;
  int k1 = (int)P * (int)Q;
  
  int numOpenMPThreads = -1;
  
  #pragma omp parallel  
{
  numOpenMPThreads = omp_get_num_threads();

#pragma omp master  
{  
  temp = new double[numTempElem*numOpenMPThreads];
  std::memset(temp, 0, numTempElem*numOpenMPThreads*sizeof(double));
  rotatedDoutPtrArrays = new double[numRotatedElem*numOpenMPThreads];
  loweredMatArrays = new double[numIm2ColElem*numOpenMPThreads];
} 
    
#pragma omp barrier

#pragma omp for
  for (int n = 0; n < (int)N; n++) {
  	double* loweredMat = loweredMatArrays + numIm2ColElem*omp_get_thread_num();

    // Step 1: Perform im2col
    im2col(inputPtr + n * CHW, loweredMat, 1, (int)C, (int)H, (int)W, (int)K,
           (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w,
           (int)P, (int)Q);
           
    // Step 2: Rotate dout
    double* rotatedDoutPtr = rotatedDoutPtrArrays + numRotatedElem*omp_get_thread_num();
    rotate180(doutPtr + n * KPQ, rotatedDoutPtr, 1, (int)C, (int)H, (int)W, (int)K,
           (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w,
           (int)P, (int)Q);
    
    // Multiply to get CRS X K
    double* temp1 = temp + numTempElem*omp_get_thread_num();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1, n1, k1, 1.0, loweredMat, k1,
              rotatedDoutPtr, n1, 1.0, temp1, n1);
  }
  
}
  
  // Inplace transpose addition
  int numRow = (int)C * (int)R * (int)S;
  for(int t = 0; t < numOpenMPThreads; t++) {
  	int iter = 0;
  	double* temp1 = temp + numTempElem*t;
	for(int i = 0; i < (int)C * (int)R * (int)S; i++) {
		for(int j = 0; j < (int)K; j++, iter++) {
			int index = j*numRow+i;
			retPtr[index] += temp1[iter];
		}
	}
  }
  
  delete [] temp;
  delete [] loweredMatArrays;
  delete [] rotatedDoutPtrArrays;
  
  RELEASE_DOUBLE_ARRAY(env, input, inputPtr);
  RELEASE_DOUBLE_ARRAY(env, dout, doutPtr);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr);
}

JNIEXPORT void JNICALL Java_org_apache_sysml_runtime_controlprogram_CPPUtil_conv2dBackwardDataDense(
  JNIEnv* env, jclass, jdoubleArray filter, jdoubleArray dout,
    jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q) {
  // First step: Avoids oversubscription and other openmp/internal blas threading issues
  ensureSequentialBLAS();
  
  double* filterPtr = GET_DOUBLE_ARRAY(env, filter);
  double* doutPtr = GET_DOUBLE_ARRAY(env, dout);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret);
  int CHW = (int)C * (int)H * (int)W;
  int KPQ = (int)K * (int)P * (int)Q;
  int numRotatedElem = (int)P * (int)Q * (int)K;
  int numCol2ImElem = (int)P * (int)Q * (int)C * (int)R * (int)S;

  double* rotatedDoutPtrArrays;
  double* col2imInputArrays;

#pragma omp parallel  
{
  int numOpenMPThreads = omp_get_num_threads();

#pragma omp master  
{  
  rotatedDoutPtrArrays = new double[numRotatedElem*numOpenMPThreads];
  col2imInputArrays = new double[numCol2ImElem*numOpenMPThreads];
} 
    
#pragma omp barrier

#pragma omp for
  for (int n = 0; n < (int)N; n++) {
    // Step 1: Rotate dout
    double* rotatedDoutPtr = rotatedDoutPtrArrays + numRotatedElem*omp_get_thread_num();
    rotate180(doutPtr + n * KPQ, rotatedDoutPtr, 1, (int)C, (int)H, (int)W, (int)K,
           (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w,
           (int)P, (int)Q);

    // Step 2: t(rotatedDout (PQ X K) %*% filter (K X CRS))
    double* col2imInput = col2imInputArrays + numCol2ImElem*omp_get_thread_num();
    matmult(rotatedDoutPtr, filterPtr, col2imInput,
            (int)P * (int)Q, (int)K, (int)C * (int)R * (int)S, 1);

    // Step 3: Perform col2im
    col2im(col2imInput, retPtr + n * CHW, 1, (int)C, (int)H, (int)W, (int)K,
           (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w,
           (int)P, (int)Q);

  } // end omp parallel for
  
#pragma omp barrier

} // end omp parallel

  delete [] rotatedDoutPtrArrays;
  delete [] col2imInputArrays;
  
  RELEASE_DOUBLE_ARRAY(env, filter, filterPtr);
  RELEASE_DOUBLE_ARRAY(env, dout, doutPtr);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr);
}

JNIEXPORT void JNICALL Java_org_apache_sysml_runtime_controlprogram_CPPUtil_conv2dBiasAddDense(
    JNIEnv* env, jclass, jdoubleArray input, jdoubleArray bias, jdoubleArray filter,
    jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q) {
  // First step:  Avoids oversubscription and other openmp/internal blas threading issues
  ensureSequentialBLAS();
  
  double* inputPtr = GET_DOUBLE_ARRAY(env, input);
  double* biasPtr = GET_DOUBLE_ARRAY(env, bias);
  double* filterPtr = GET_DOUBLE_ARRAY(env, filter);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret);
  int CHW = (int)C * (int)H * (int)W;
  int KPQ = (int)K * (int)P * (int)Q;
  int PQ = (int)P * (int)Q;
  int numIm2ColElem = (int)C * (int)R * (int)S * (int)P * (int)Q;
  
  double* loweredMatArrays;
  
#pragma omp parallel  
{
  int numOpenMPThreads = omp_get_num_threads();

#pragma omp master  
{  
  loweredMatArrays = new double[numIm2ColElem*numOpenMPThreads];
} 
    
#pragma omp barrier

#pragma omp for
  for (int n = 0; n < (int)N; n++) {
    double* loweredMat = loweredMatArrays + numIm2ColElem*omp_get_thread_num();

    // Step 1: Perform im2col
    im2col(inputPtr + n * CHW, loweredMat, 1, (int)C, (int)H, (int)W, (int)K,
           (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w,
           (int)P, (int)Q);

    // Step 2: filter (K X CRS) %*% loweredMat (CRS X PQ)
    matmult(filterPtr, loweredMat, retPtr + n * KPQ, (int)K,
            (int)C * (int)R * (int)S, (int)P * (int)Q, 1);
    
    // Step 3: Add bias
    double* outputArr = retPtr + n*KPQ;
    int index = 0;
	for(int k = 0; k < K; k++) {
		for(int pq = 0; pq < PQ; pq++, index++) {
			outputArr[index] += biasPtr[k];
		}
	}
    
  } // end omp parallel for

#pragma omp barrier

} // end omp parallel

  delete [] loweredMatArrays;
  
  RELEASE_DOUBLE_ARRAY(env, input, inputPtr);
  RELEASE_DOUBLE_ARRAY(env, bias, biasPtr);
  RELEASE_DOUBLE_ARRAY(env, filter, filterPtr);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr);
}

JNIEXPORT void JNICALL
Java_org_apache_sysml_runtime_controlprogram_CPPUtil_conv2dDense(
    JNIEnv* env, jclass, jdoubleArray input, jdoubleArray filter,
    jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q) {
  // First step:  Avoids oversubscription and other openmp/internal blas threading issues
  ensureSequentialBLAS();
  
  double* inputPtr = GET_DOUBLE_ARRAY(env, input);
  double* filterPtr = GET_DOUBLE_ARRAY(env, filter);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret);
  int CHW = (int)C * (int)H * (int)W;
  int KPQ = (int)K * (int)P * (int)Q;
  int numIm2ColElem = (int)C * (int)R * (int)S * (int)P * (int)Q;
  
  double* loweredMatArrays;
  
#pragma omp parallel  
{
  int numOpenMPThreads = omp_get_num_threads();

#pragma omp master  
{  
  loweredMatArrays = new double[numIm2ColElem*numOpenMPThreads];
} 
    
#pragma omp barrier

#pragma omp for
  for (int n = 0; n < (int)N; n++) {
    double* loweredMat = loweredMatArrays + numIm2ColElem*omp_get_thread_num();

    // Step 1: Perform im2col
    im2col(inputPtr + n * CHW, loweredMat, 1, (int)C, (int)H, (int)W, (int)K,
           (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w,
           (int)P, (int)Q);

    // Step 2: filter (K X CRS) %*% loweredMat (CRS X PQ)
    matmult(filterPtr, loweredMat, retPtr + n * KPQ, (int)K,
            (int)C * (int)R * (int)S, (int)P * (int)Q, 1);
    
  } // end omp parallel for

#pragma omp barrier

} // end omp parallel

  delete [] loweredMatArrays;
  
  RELEASE_DOUBLE_ARRAY(env, input, inputPtr);
  RELEASE_DOUBLE_ARRAY(env, filter, filterPtr);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr);
}