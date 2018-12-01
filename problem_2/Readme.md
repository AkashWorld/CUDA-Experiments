# Dynamic Matrix Multiplication

This project compares different methods of matrix multiplication: classical CPU multiplication, loop unrolling, GPU accelerated (CUDA), tiled multiplication, etc...

### Prerequisites
* [CMake](https://cmake.org/download/) - Cross Platform Build System  
* C++ 11 Compiler (g++, MSVC(cl), clang++, etc)  
* [CUDA Compiler (nvcc)](https://developer.nvidia.com/cuda-zone) - GPGPU Interface Used

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

Build the project as specified in the **Installing** section.  
Run the following command at the root **build** directory:
```
make test
```
To run the tests directly, run the following in the root **build** directory:
```
./catch_test
```

## Authors

* **Khalid Akash**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

