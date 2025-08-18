/*
ND-Range: Defines global and local work sizes for kernel execution
Work-Group: A group of work-items that can share local memory
Work-Item: Single thread that executes kernel code
In GEMM, each work-group computes a tile of the output matrix
*/

// 2D Parallel Matrix Add example

#include<sycl/sycl.hpp>
#include<iostream>
#include<vector>

using namespace sycl;

int main(){
    queue q;
    const int N = 4; // 4x4 matrix
    std::vector<int> A(N*N), B(N*N), C(N*N);

    // Initialize matrices
    for (int i=0;i<N*N;i++){
        A[i] = i;
        B[i] = 2*i;
    }

    {
        buffer<int, 2> bufA(A.data(), range<2>(N, N));
        buffer<int, 2> bufB(B.data(), range<2>(N, N));
        buffer<int, 2> bufC(C.data(), range<2>(N, N));

        q.submit([&](handler& h){
            auto accA = bufA.get_access<access::mode::read>(h);
            auto accB = bufB.get_access<access::mode::read>(h);
            auto accC = bufC.get_access<access::mode::write>(h);

            // ND range: global size = 4x4, local size = 2x2
            h.parallel_for(nd_range<2>(range<2>(N,N), range<2>(2,2)),[=](nd_item<2> item){
                    int row = item.get_global_id(0);
                    int col = item.get_global_id(1);
                    accC[row][col] = accA[row][col] + accB[row][col];
            });
        }).wait();
    }

    // Print results
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            std::cout<< C[i*N + j] <<" ";
        }
        std::cout<<"\n";
    }
}