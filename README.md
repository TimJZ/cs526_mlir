# CS526 Final Project

## Installation

### 1. Install LLVM

To use this project, you'll need to clone the LLVM repository and build it from source. Follow these steps to install LLVM:

#### Linux / macOS / Windows (Git)

1. Clone the LLVM repository:
   ```bash
   git clone https://github.com/llvm/llvm-project.git

2. Install LLVM project
Notice that the DCMAKE_INSTALL_PREFIX should be the path to the llvm folder you just created. Also, this build assumes you use x86. 
    ```bash
    cd llvm-project
    mkdir build
    mkdir -p ./llvm-install/llvm 
    cd llvm-project/build 

    cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS='mlir;lld;clang' \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_BUILD_TOOLS=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DCMAKE_INSTALL_PREFIX=/home/jz23/cs526_proj/llvm-install/llvm \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \


    ```bash
    cmake --build . --target install 

3. Clone the current project repo and replace the MLIR folder




