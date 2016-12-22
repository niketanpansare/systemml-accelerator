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
package org.apache.sysml.utils.accelerator;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import jcuda.Pointer;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcudnn.JCudnn;
import jcuda.jcusparse.JCusparse;
import jcuda.runtime.JCuda;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcublas.cublasHandle;
import jcuda.jcusparse.cusparseHandle;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcudnn.JCudnn.cudnnDestroy;
import static jcuda.jcusparse.JCusparse.cusparseDestroy;
import static jcuda.jcudnn.JCudnn.cudnnCreate;
import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcusparse.JCusparse.cusparseCreate;

public class JCudaHelper {
	private static final Log LOG = LogFactory.getLog(JCudaHelper.class.getName());
	private static boolean isJCudaLoaded = false;
	private static long jcudaInitTime = 0;
	private static int deviceCount = 0;
	private static cudnnHandle initializedCudnnHandle;
	private static cublasHandle initializedCublasHandle;
	private static cusparseHandle initializedCusparseHandle;
	public static cudnnHandle getCudnnHandle() {
		return initializedCudnnHandle;
	}
	public static cublasHandle getCublasHandle() {
		return initializedCublasHandle;
	}
	public static cusparseHandle getCusparseHandle() {
		return initializedCusparseHandle;
	}
	static {
		String specifiedGPU = System.getenv("SYSTEMML_GPU");
		if(specifiedGPU == null || specifiedGPU.trim().toLowerCase().equals("cuda")) {
			if(LibraryLoader.isCUDAAvailable()) {
				long start = System.nanoTime();
				
				try {
					JCuda.setExceptionsEnabled(true);
					JCudnn.setExceptionsEnabled(true);
					JCublas2.setExceptionsEnabled(true);
					JCusparse.setExceptionsEnabled(true);
					JCudaDriver.setExceptionsEnabled(true);
				}
				catch(java.lang.UnsatisfiedLinkError e) {
					LOG.debug(e.getMessage());
					throw new java.lang.UnsatisfiedLinkError("Couldnot load native JCuda libraries");
				}
				
				cuInit(0); // Initialize the driver
				// Obtain the number of devices
		        int deviceCountArray[] = { 0 };
		        cuDeviceGetCount(deviceCountArray);
		        deviceCount = deviceCountArray[0];
		        LOG.info("Total number of GPUs on the machine: " + deviceCount);
		        jcudaInitTime = System.nanoTime() - start;
		        if(testGPU()) {
		        	isJCudaLoaded = true;
		        	LOG.info("GPU is enabled");
		        	initializedCudnnHandle = new cudnnHandle();
		        	cudnnCreate(initializedCudnnHandle);
		        	initializedCublasHandle = new cublasHandle();
		        	cublasCreate(initializedCublasHandle);
		        	initializedCusparseHandle = new cusparseHandle();
		    		cusparseCreate(initializedCusparseHandle);
		        	Runtime.getRuntime().addShutdownHook(new Thread() {
	        	      public void run() {
	        	    	cudnnDestroy(initializedCudnnHandle);
	        			cublasDestroy(initializedCublasHandle);
	        			cusparseDestroy(initializedCusparseHandle);
	        	      }
	        	    });
		        }
		        else {
		        	LOG.info("GPU is not enabled (memcpy test not successful)");
		        }
			}
		}
		else {
			LOG.info("Not loading JCUDA as SYSTEMML_GPU="+specifiedGPU);
		}
	}
	
	public static long getJCudaInitTime() {
		return jcudaInitTime;
	}
	public static int getDeviceCount() {
		return deviceCount;
	}
	public static boolean isGPUAvailable() {
		return isJCudaLoaded;
	}
	
	private static boolean testGPU() {
		Pointer deviceData = new Pointer();
		int numElems = 3;
		double hostData[] = new double[numElems];
		long memorySize = numElems*((long)jcuda.Sizeof.DOUBLE);
		cudaMalloc(deviceData, memorySize);
		cudaMemcpy(deviceData, Pointer.to(hostData), memorySize, cudaMemcpyKind.cudaMemcpyHostToDevice);
		double hostData1[] = new double[numElems];
		cudaMemcpy(Pointer.to(hostData1), deviceData, memorySize, cudaMemcpyKind.cudaMemcpyDeviceToHost);
		for(int i = 0; i < numElems; i++) {
			if(hostData[i] != hostData1[i])
				return false;
		}
		return true;
	}
}
