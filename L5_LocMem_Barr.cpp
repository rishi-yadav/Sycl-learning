/*
Local Memory -> shared fast memory by all work-items in a work-groups
Barrier -> Syc work-items in a work-group, ensuring all r/w to local memory are complete
In GEMM kernels, each work-group loads tiles of matrices into local memory to avoid slow global memory accesses
*/

#include<sycl/sycl.hpp>
#include<iostream>
#include<vector>

using namespace sycl;

int main(){
    queue q;
    const int N = 8;
    std::vector<int> A(N), B(N), C(N);

    for (int i=0;i<N;i++){
        A[i] = i;
        B[i] = 2*i;
    }

    {
        buffer<int, 1> bufA(A.data(), range<1>(N));
        buffer<int, 1> bufB(B.data(), range<1>(N));
        buffer<int, 1> bufC(C.data(), range<1>(N));

        q.submit([&](handler& h){
            auto accA = bufA.get_access<access::mode::read>(h);
            auto accB = bufB.get_access<access::mode::read>(h);
            auto accC = bufC.get_access<access::mode::write>(h);

            // Local memory allocation (shared per work-group)
            local_accessor<int, 1> localA(range<1>(N/2), h); // Assumes 2 work-groups

            h.parallel_for(nd_range<1>(range<1>(N), range<1>(N/2)), [=](nd_item<1> item){
                int l_id = item.get_local_id(0);
                int g_id = item.get_global_id(0);

                // Load into local memory
                localA[l_id] = accA[g_id];
                item.barrier(access::fence_space::local_space);

                // Each work-item adds local_value + global B
                accC[g_id] = localA[l_id] + accB[g_id];
            });
        }).wait();
    }

    for (int i=0;i<N;i++)
        std::cout<<A[i] << " + "<<B[i]<<" = "<<C[i]<<"\n";

    return 0;
}