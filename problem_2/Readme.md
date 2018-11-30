# Dynamic Matrix Multiplication

This project compares different methods of matrix multiplication: classical CPU multiplication, loop unrolling, GPU accelerated (CUDA), tiled multiplication, etc...

### Prerequisites
* [CMake](https://cmake.org/download/) - Cross Platform Build System  
* C++ 11 Compiler (g++, MSVC, clang++, etc)  
* [CUDA](https://developer.nvidia.com/cuda-zone) - GPGPU Interface Used

### Installing

With the prerequisites satisfied, run cmake in relative to the problem_2 root directory. It is recommended that a build directory is made, and the cmake build files are built in there. For example:
```
mkdir ./build #Make a build repository at ./problem_2/build  
cd build  
cmake ..
```

Then run the platform specific build tool that is available in the build folder, such as make, ninja, or msvc.  
In the traditional linux system, run the following command:  
```
make
```

## Running the tests

TODO: Explain how to run the automated tests for this system


## Authors

* **Khalid Akash**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

