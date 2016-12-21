<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->


By default, SystemML implements all its matrix operations in Java.
This simplifies deployment especially in a distributed environment.

In some cases (such as deep learning), the user might want to use native BLAS
rather than SystemML's internal Java library for performing single-node
operations such matrix multiplication, convolution, etc.
To enable that, the user has to build `systemml.cpp` (available in this folder)
as a shared library and make it available through `LD_LIBRARY_PATH` (on linux)
or `PATH` (on Windows). If the shared library is not accessible, SystemML
falls back to its internal Java library.

In the below section, we describe the steps to build `systemml.cpp` using native BLAS.

# Step 1: Install BLAS

## Option 1: Install Intel MKL (recommended)

1. Download and install the [community version of Intel MKL](https://software.intel.com/sites/campaigns/nest/).
Intel requires you to first register your email address and then sends the download link to your email address
with license key.

	* Linux users will have to extract the downloaded `.tgz` file and execute `install.sh`.
	* Windows users will have to execute the downloaded `.exe` file and follow the guided setup.

2. Set `MKLROOT` enviroment variable to point to the installed location.

	* Linux: By default, Intel MKL will be installed in `/opt/intel/mkl/`.
	 
		```bash
		export MKLROOT=/opt/intel/mkl/
		```
	
	* Windows: By default, Intel MKL will be installed in `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017\windows\mkl`.
	
To add a new enviroment variable on Windows, the user has to right-click on `Computer` and then click `Properties > Advanced system settings > Environment Variables > New`.
	
## Option 2: Install OpenBLAS  

1. RedHat / CentOS:
	```bash
	sudo yum install openblas
	``` 

2. Ubuntu:
	```bash
	sudo apt-get install openblas
	``` 

3. Windows:
Download [pre-built binaries](https://sourceforge.net/projects/openblas/) or install from [the source](https://github.com/xianyi/OpenBLAS).

4. Set `OPENBLASROOT` enviroment variable to point to the installed location.

# Compiling systemml.cpp

## 64-bit Linux 

1. Intel MKL:
	
	```bash
	g++ -o libsystemml_mkl_x86_64.so systemml.cpp  -I$JAVA_HOME/include -I$MKLROOT/include -I$JAVA_HOME/include/linux -lmkl_rt -lpthread -lm -ldl -L$MKLROOT/lib/intel64 -m64 -Wl,--no-as-needed -fopenmp -O3 -shared -fPIC
	```

2. OpenBLAS:

	```bash
	# -I$OPENBLASROOT/include -L$OPENBLASROOT/lib 
	g++ -o libsystemml_openblas_x86_64.so systemml.cpp  -I$JAVA_HOME/include  -I$JAVA_HOME/include/linux -lopenblas -lpthread -lm -ldl -DUSE_OPEN_BLAS -fopenmp -O3 -shared -fPIC
	```

## 62-bit Windows

1. Intel MKL:
	
	```bash
	"C:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\VC\\vcvarsall.bat" amd64
	"%MKLROOT%"\bin\mklvars.bat intel64
	set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_25
	cl systemml.cpp -I. -I"%MKLROOT%"\include -I"%JAVA_HOME%"\include -I"%JAVA_HOME%"\include\win32 -Fesystemml_mkl_x86_64.dll -MD -LD  "%MKLROOT%"\lib\intel64_win\mkl_intel_thread_dll.lib "%MKLROOT%"\lib\intel64_win\mkl_core_dll.lib "%MKLROOT%"\lib\intel64_win\mkl_intel_lp64_dll.lib
	```


2. OpenBLAS:
	
Download and extract [64-bit OpenBLAS](https://sourceforge.net/projects/openblas/files/v0.2.19/OpenBLAS-v0.2.19-Win64-int32.zip/download)

	```bash
	"C:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\VC\\vcvarsall.bat" amd64
	set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_25
	cl systemml.cpp -I. -I"%OPENBLASROOT%"\include -I"%JAVA_HOME%"\include -I"%JAVA_HOME%"\include\win32 -DUSE_OPEN_BLAS -Fesystemml_openblas_x86_64.dll -MD -LD "%OPENBLASROOT%"\lib\libopenblas.dll.a 
	```
	