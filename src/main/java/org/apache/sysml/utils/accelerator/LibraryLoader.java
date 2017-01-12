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
import org.apache.commons.lang.SystemUtils;

import java.util.HashMap;

// --------------------------------------
// Required for loadLibrary

//import jcuda.LibUtils;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.File;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;

//--------------------------------------

public class LibraryLoader {
	
	private static final Log LOG = LogFactory.getLog(LibraryLoader.class.getName());
	
	private static HashMap<String, String> archMap = new HashMap<String, String>();;
	static {
        archMap.put("x86", "x86_32");
        archMap.put("i386", "x86_32");
        archMap.put("i486", "x86_32");
        archMap.put("i586", "x86_32");
        archMap.put("i686", "x86_32");
        archMap.put("x86_64", "x86_64");
        archMap.put("amd64", "x86_64");
        archMap.put("powerpc", "ppc_64");
	}
	
	public static boolean isMKLAvailable() {
		try {
			if(SystemUtils.IS_OS_LINUX) {
				try {
					// This sets the environment variable MKL_THREADING_LAYER to GNU (which has to be done before loading MKL)
					// which is helpful in avoid performance issues with Intel Multi-threading and GNU OpenMP
					// See https://software.intel.com/en-us/node/528707 and https://software.intel.com/en-us/node/528522
					LibraryLoader.loadLibrary("preload_systemml", "");
					BLASHelper.initializePreMKLLoad();
				} catch (IOException e) {}
				// GNU OpenMP is needed by Intel MKL
				System.loadLibrary("gomp");
			}
			System.loadLibrary("mkl_rt");
			return true;
		}
		catch (UnsatisfiedLinkError e) {
			LOG.info("Unable to load MKL:" + e.getMessage());
			return false;
		}
	}
	
	public static boolean isOpenBLASAvailable() {
		if(SystemUtils.IS_OS_WINDOWS) {
			String message = "";
			try {
				 System.loadLibrary("openblas");
				 return true;
			}
			catch (UnsatisfiedLinkError e) {
				message += e.getMessage() + " ";
			}
			try {
				 System.loadLibrary("libopenblas");
				 return true;
			}
			catch (UnsatisfiedLinkError e) {
				message += e.getMessage() + " ";
			}
			LOG.info("Unable to load OpenBLAS:" + message);
			return false;
		}
		else {
			try {
				 System.loadLibrary("openblas");
				 return true;
			}
			catch (UnsatisfiedLinkError e) {
				LOG.info("Unable to load OpenBLAS:" + e.getMessage());
				return false;
			}
		}
	}
	
	/*
	 * Note: CUDA 8.0 64 bit should be accessible via path on windows
	 */
	public static boolean isCUDAAvailable() {
		try {
			if (SystemUtils.IS_OS_WINDOWS) {
				if(System.getenv("CUDA_PATH") != null) 
					return true;
				else
					return false;
			}
			else {
				System.loadLibrary("cuda");
			}
			return true;
		}
		catch (UnsatisfiedLinkError e) {
			LOG.info("Unable to load CUDA:" + e.getMessage());
			return false;
		}
	}
	
//	// Loading systemml through LibUtils is not working
//	public static void loadLibrary(String libName, String suffix1) throws IOException {
//		try {
//			LibUtils.loadLibrary(libName+suffix1);
//		} 
//		catch (UnsatisfiedLinkError e) {
//			throw new IOException(e.getMessage());
//		}
//	}
	
	public static void loadLibrary(String libName, String suffix1) throws IOException {
		String prefix = "";
		String suffix2 = "";
		String os = "";
		if (SystemUtils.IS_OS_MAC_OSX) {
			prefix = "lib";
			suffix2 = "dylib";
			os = "apple";
		} else if (SystemUtils.IS_OS_LINUX) {
			prefix = "lib";
			suffix2 = "so";
			os = "linux";
		} else if (SystemUtils.IS_OS_WINDOWS) {
			prefix = "";
			suffix2 = "dll";
			os = "windows";
		} else {
			LOG.info("Unsupported OS:" + SystemUtils.OS_NAME);
			throw new IOException("Unsupported OS");
		}
		
		String arch = archMap.get(SystemUtils.OS_ARCH);
		if(arch == null) {
			LOG.info("Unsupported architecture:" + SystemUtils.OS_ARCH);
			throw new IOException("Unsupported architecture:" + SystemUtils.OS_ARCH);
		}
		loadLibraryHelper(prefix + libName + suffix1 + "-" + os + "-" + arch + "." + suffix2);
	}

	public static void loadLibraryHelper(String path) throws IOException {
		InputStream in = null; OutputStream out = null;
		try {
			in = LibraryLoader.class.getResourceAsStream("/lib/"+path);
			if(in != null) {
				File temp = File.createTempFile(path, "");
				temp.deleteOnExit();
				out = FileUtils.openOutputStream(temp);
		        IOUtils.copy(in, out);
		        in.close(); in = null;
		        out.close(); out = null;
				System.load(temp.getAbsolutePath());
			}
			else
				throw new IOException("No lib available in the jar:" + path);
			
		} catch(IOException e) {
			LOG.info("Unable to load library " + path + " from resource:" + e.getMessage());
			throw e;
		} finally {
			if(out != null)
				out.close();
			if(in != null)
				in.close();
		}
		
	}
}
