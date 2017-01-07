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

public class BLASHelper {
	private static boolean isSystemMLLoaded = false;
	private static final Log LOG = LogFactory.getLog(BLASHelper.class.getName());
	static {
		String blasType = getBLASType();
		if(blasType != null) {
			LOG.info("Found BLAS: " + blasType);
			try {
				LibraryLoader.loadLibrary("systemml", "_" + blasType);
				isSystemMLLoaded = true;
				LOG.info("Successfully loaded systemml library with " + blasType);
			} catch (IOException e) { }
		}
	}
	
	public static boolean isNativeBLASAvailable() {
		return isSystemMLLoaded;
	}
	
	private static String getBLASType() {
		String specifiedBLAS = System.getenv("SYSTEMML_BLAS");
		if(specifiedBLAS != null) {
			if(specifiedBLAS.trim().toLowerCase().equals("mkl")) {
				return LibraryLoader.isMKLAvailable() ? "mkl" : null;
			}
			else if(specifiedBLAS.trim().toLowerCase().equals("openblas")) {
				return LibraryLoader.isOpenBLASAvailable() ? "openblas" : null;
			}
			else if(specifiedBLAS.trim().toLowerCase().equals("none")) {
				LOG.info("Not loading native BLAS as SYSTEMML_BLAS=" + specifiedBLAS);
				return null;
			}
			else {
				LOG.info("Unknown BLAS:" + specifiedBLAS);
				return null;
			}
		}
		else {
			// No BLAS specified ... try loading Intel MKL first
			return LibraryLoader.isMKLAvailable() ? "mkl" : (LibraryLoader.isOpenBLASAvailable() ? "openblas" : null);
		}
	}
	
	public static native void initializePreMKLLoad();
}
