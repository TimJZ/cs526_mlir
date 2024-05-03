# CS526 Final Project

## Installation

### 1. Install LLVM

To use this project, you'll need to clone the LLVM repository and build it from source. Follow these steps to install LLVM:

#### Linux / macOS / Windows (Git)

1. Clone the LLVM repository:
   ```bash
   git clone https://github.com/llvm/llvm-project.git

2. Configure the make file 
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


   

3. Clone the current project repo and replace the original MLIR folder
    ```bash 
    git clone git@github.com:TimJZ/cs526_mlir.git
    
    # Replace the original MLIR folder with the cloned repository
    rm -rf llvm-project/mlir  # Remove the existing MLIR folder
    mv cs526_mlir llvm-project/mlir  # Move the cloned repository to the correct location

    #Install 
    cmake --build . --target install 


4. Test the pass 
    ```bash 
    cd /mlir/test/cs526_test 
    /path/to/llvm-project/build/bin/mlir-opt --cs526 real_lenet.mlir



