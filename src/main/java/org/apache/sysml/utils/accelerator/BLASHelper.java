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

public class BLASHelper {
	private static boolean isSystemMLLoaded = false;
	static {
		String blasType = getBLASType();
		if(blasType != null) {
			try {
				LibraryLoader.loadLibrary("systemml", "_" + blasType);
				isSystemMLLoaded = true;
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
				try {
					 System.loadLibrary("mkl_rt");
					 return "mkl";
				}
				catch (UnsatisfiedLinkError e) {
					// If you cannot load mkl, don't try loading openblas as SYSTEMML_BLAS is specified
					return null;
				}
			}
			else if(specifiedBLAS.trim().toLowerCase().equals("openblas")) {
				try {
					 System.loadLibrary("openblas");
					 return "openblas";
				}
				catch (UnsatisfiedLinkError e) {
					// If you cannot load openblas, don't try loading mkl as SYSTEMML_BLAS is specified
					return null;
				}
			}
			else {
				return null;
			}
		}
		else {
			// No BLAS specified ... try loading Intel MKL first
			try {
				 System.loadLibrary("mkl_rt");
				 return "mkl";
			}
			catch (UnsatisfiedLinkError e) { }
			try {
				 System.loadLibrary("openblas");
				 return "openblas";
			}
			catch (UnsatisfiedLinkError e) { }
			return null;
		}
	}
	
}
