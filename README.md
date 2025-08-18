# SYCL Learning Project

This repository contains SYCL foundation lessons and examples for learning parallel programming concepts that are essential for CUTLASS-SYCL development.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Compilation](#compilation)
- [Execution](#execution)
- [Project Structure](#project-structure)
- [Lessons Overview](#lessons-overview)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **C++ Compiler**: GCC 9+ or Clang 10+
- **CMake**: Version 3.20+
- **Intel oneAPI DPC++ Compiler** or **hipSYCL/Open SYCL**

## Installation

### Option 1: Intel oneAPI DPC++ (Recommended)

1. **Download Intel oneAPI Base Toolkit:**
   ```bash
   wget https://registrationcenter.intel.com/en/download-links/
   # Or use the installer script
   curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo apt-key add -
   echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
   sudo apt update
   sudo apt install intel-oneapi-dpcpp-cpp-compiler
   ```

2. **Set up environment:**
   ```bash
   source /opt/intel/oneapi/setvars.sh
   # Add to ~/.bashrc for permanent setup
   echo "source /opt/intel/oneapi/setvars.sh" >> ~/.bashrc
   ```

## Compilation

### Method 1: Direct Compilation (Intel DPC++)

```bash
# Compile individual lessons
dpcpp -fsycl L5_LocMem_Barr.cpp -o L5_LocMem_Barr

# With optimization
dpcpp -fsycl -O3 L5_LocMem_Barr.cpp -o L5_LocMem_Barr

# For debugging
dpcpp -fsycl -g -O0 L5_LocMem_Barr.cpp -o L5_LocMem_Barr
```

### Method 2: Using CMake (Recommended)

Create a `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(SYCLLearning)

find_package(IntelSYCL REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executables for each lesson
add_executable(L5_LocMem_Barr SYCL/L5_LocMem_Barr.cpp)
add_sycl_to_target(TARGET L5_LocMem_Barr SOURCES SYCL/L5_LocMem_Barr.cpp)

# Add other lessons here...
```

Build with CMake:
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Execution

### Run CPU Backend (Default)
```bash
./L5_LocMem_Barr
```

### Run on GPU (Intel)
```bash
SYCL_DEVICE_FILTER=gpu ./L5_LocMem_Barr
```

### Run on CPU with specific device
```bash
SYCL_DEVICE_FILTER=cpu ./L5_LocMem_Barr
```

### Check Available Devices
```bash
sycl-ls
```

## Project Structure

```
/home/user_name/
├── README.md
├── SYCL/
│   ├── sycl.tex                 # Comprehensive SYCL roadmap document
│   ├── L1_HelloSYCL.cpp         # Basic SYCL introduction
│   ├── L2_FirstKernel.cpp       # Parallel work-items
│   ├── L3_BuffersAccessors.cpp  # Memory management
│   ├── L4_NDRange_WorkGroups.cpp # 2D indexing and work-groups
│   ├── L5_LocMem_Barr.cpp       # Local memory and barriers
│   ├── L6_SubGroups_SIMD.cpp    # Sub-groups and vectorization
│   └── L7_USM.cpp               # Unified Shared Memory
└── cutlass-sycl/                # CUTLASS-SYCL repository (future)
```

## Lessons Overview

| Lesson | Concept | Key Learning |
|--------|---------|--------------|
| L1 | Hello SYCL | Queue, device selection |
| L2 | First Kernel | Parallel work-items, device code |
| L3 | Buffers & Accessors | Host ↔ Device memory management |
| L4 | ND-Range & Work-Groups | 2D indexing, local/global sizes |
| L5 | Local Memory & Barriers | Shared memory, synchronization |
| L6 | Sub-Groups & SIMD | Vectorized operations |
| L7 | USM | Pointer-style memory management |

## Testing

### Run All Examples
```bash
# Create a test script
cat > run_tests.sh << 'EOF'
#!/bin/bash
cd SYCL
for file in L*.cpp; do
    echo "Compiling and running $file..."
    dpcpp -fsycl "$file" -o "${file%.cpp}"
    ./"${file%.cpp}"
    echo "---"
done
EOF

chmod +x run_tests.sh
./run_tests.sh
```

### Expected Output for L5_LocMem_Barr
```
0 + 0 = 0
1 + 2 = 3
2 + 4 = 6
3 + 6 = 9
4 + 8 = 12
5 + 10 = 15
6 + 12 = 18
7 + 14 = 21
```

## Troubleshooting

### Common Issues

1. **SYCL runtime not found:**
   ```bash
   # Ensure oneAPI environment is sourced
   source /opt/intel/oneapi/setvars.sh
   ```

2. **No GPU device found:**
   ```bash
   # Check available devices
   sycl-ls
   # Install GPU drivers
   sudo apt install intel-opencl-icd
   ```

3. **Compilation errors:**
   ```bash
   # Check SYCL compiler version
   dpcpp --version
   # Use correct C++ standard
   dpcpp -fsycl -std=c++17 file.cpp
   ```

4. **Runtime errors:**
   ```bash
   # Enable SYCL debugging
   export SYCL_PI_TRACE=1
   export SYCL_RT_WARNING_LEVEL=1
   ./your_program
   ```

## Next Steps: CUTLASS-SYCL

After mastering these lessons:

1. **Clone CUTLASS-SYCL:**
   ```bash
   git clone https://github.com/intel/cutlass-sycl.git
   cd cutlass-sycl
   ```

2. **Build GEMM examples:**
   ```bash
   mkdir build && cd build
   cmake .. -DCUTLASS_ENABLE_SYCL=ON
   make -j$(nproc)
   ```

3. **Run GEMM tests:**
   ```bash
   ./examples/basic_gemm
   ```

## Contributing

Feel free to contribute by:
- Adding more SYCL examples
- Improving documentation
- Reporting issues
- Optimizing kernels

## Resources

- [SYCL Specification](https://www.khronos.org/sycl/)
- [Intel oneAPI Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
- [CUTLASS-SYCL Repository](https://github.com/intel/cutlass-sycl)
- [SYCL Academy](https://github.com/codeplaysoftware/syclacademy)
