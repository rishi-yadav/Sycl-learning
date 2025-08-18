/*
Buffer -> Represent memory shared between host and device
Accessor -> Provide a way to access buffer data from kernels
In other way a bridge b/w cpu and gpu memory
*/

#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
using namespace sycl;

int main(){
    queue q;

    const int N = 16;
    std::vector<int> A(N), B(N), C(N);

    // Initialize host data

    for (auto i=0; i<N; i++){
        A[i] = i;
        B[i] = i * 2;
    }

    {
        /*
        buffer<int, 1> bufA(A.data(), range<1>(N));
                    ^                    ^
                    |                    |
            1D buffer            1D range with N elements
        */
        // Create buffers (size N)
        buffer<int, 1> bufA(A.data(), range<1>(N));
        buffer<int, 1> bufB(B.data(), range<1>(N));
        buffer<int, 1> bufC(C.data(), range<1>(N));

        // Submit kernel to device
        q.submit([&](handler& h){
            auto accA = bufA.get_access<access::mode::read>(h);
            auto accB = bufB.get_access<access::mode::read>(h);
            auto accC = bufC.get_access<access::mode::write>(h);
            
            // [=] captures all variables by value (accessors accA, accB, accC)
            // id<1> i is the unique identifier for each work-item
            h.parallel_for(range<1>(N), [=](id<1> i){
                accC[i] = accA[i] + accB[i];
            });
        }).wait();
    } // Buffers go out of scope, data is copied back to host

    for (auto i=0;i<N;i++){
        std::cout<< A[i]<<" + "<< B[i]<<" = "<< C[i] << std::endl;
    }

    return 0;
}
