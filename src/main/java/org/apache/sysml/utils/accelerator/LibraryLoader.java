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

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;

public class LibraryLoader {
	
	public static boolean isMKLAvailable() {
		try {
			 System.loadLibrary("mkl_rt");
			 return true;
		}
		catch (UnsatisfiedLinkError e) {
			return false;
		}
	}
	
	public static boolean isOpenBLASAvailable() {
		try {
			 System.loadLibrary("openblas");
			 return true;
		}
		catch (UnsatisfiedLinkError e) {
			return false;
		}
	}
	
	public static boolean isCUDAAvailable() {
		try {
			 System.loadLibrary("cuda");
			 return true;
		}
		catch (UnsatisfiedLinkError e) {
			return false;
		}
	}
	
	public static void loadLibrary(String libName, String suffix1) throws IOException {
		String OS = System.getProperty("os.name", "generic").toLowerCase();
		boolean is64bit = System.getProperty("sun.arch.data.model").contains("64");
		String prefix = "";
		String suffix2 = "_x86";
		String suffix3 = "";
		String suffix4 = "";
		
		if ((OS.indexOf("mac") >= 0) || (OS.indexOf("darwin") >= 0)) {
			prefix = "lib";
			suffix4 = "dylib";
		} else if (OS.indexOf("nux") >= 0) {
			prefix = "lib";
			suffix4 = "so";
		} else if (OS.indexOf("win") >= 0) {
			prefix = "";
			suffix4 = "dll";
		} else {
			throw new IOException("Unsupported OS");
		}
		if(is64bit)
			suffix3 = "_64";
		else
			suffix3 = "_32";
		
		String resourceFolder = ""; // "/src/main/resources/";
		loadLibraryHelper(resourceFolder + prefix + libName + suffix1 + suffix2 + suffix3 + "." + suffix4);
	}

	public static void loadLibraryHelper(String path) throws IOException {
		File temp = File.createTempFile(path, "");
		temp.deleteOnExit();
		InputStream in = LibraryLoader.class.getResourceAsStream("/"+path);
		OutputStream out = FileUtils.openOutputStream(temp);
        IOUtils.copy(in, out);
        in.close();
        out.close();
		System.load(temp.getAbsolutePath());
	}
}
