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

import java.io.IOException;

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

/**
 * Supported version:
 * CUDA 8.0 
 * cuDNN v5.1 (August 10, 2016), for CUDA 8.0
 */
public class JCudaHelper {
	private static final Log LOG = LogFactory.getLog(JCudaHelper.class.getName());
	private static boolean isJCudaLoaded = false;
	private static String jcudaVersion = "0.8.0";
	private static long jcudaInitTime = 0;
	private static int deviceCount = 0;
	static {
		try {
			if(LibraryLoader.isCUDAAvailable()) {
				long start = System.nanoTime();
				LibraryLoader.loadLibrary("JCublas-" + jcudaVersion, "");
				LibraryLoader.loadLibrary("JCublas2-" + jcudaVersion, "");
				LibraryLoader.loadLibrary("JCudaDriver-" + jcudaVersion, "");
				LibraryLoader.loadLibrary("JCudaRuntime-" + jcudaVersion, "");
				LibraryLoader.loadLibrary("JCusparse-" + jcudaVersion, "");
				LibraryLoader.loadLibrary("JNvrtc-" + jcudaVersion, "");
				LibraryLoader.loadLibrary("JCudnn-" + jcudaVersion, "");
				
				JCuda.setExceptionsEnabled(true);
				JCudnn.setExceptionsEnabled(true);
				JCublas2.setExceptionsEnabled(true);
				JCusparse.setExceptionsEnabled(true);
				JCudaDriver.setExceptionsEnabled(true);
				cuInit(0); // Initialize the driver
				// Obtain the number of devices
		        int deviceCountArray[] = { 0 };
		        cuDeviceGetCount(deviceCountArray);
		        deviceCount = deviceCountArray[0];
		        LOG.info("Total number of GPUs on the machine: " + deviceCount);
		        jcudaInitTime = System.nanoTime() - start;
		        if(testGPU()) {
		        	isJCudaLoaded = true;
		        	LOG.info("Successfully loaded jcuda libraries");
		        }
			}
		} catch (IOException e) { }
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
