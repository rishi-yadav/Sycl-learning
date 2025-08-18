#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
    // Create a queue for default device (CPU/GPU)
    queue q;

    // Run a kernel with 16 parallel work-items
    q.submit([&](handler& h) {
        stream out(1024, 256, h);
        h.parallel_for(range<1>(16), [=](id<1> i) {
            out << "Hello from work-item " << (int)i[0] << "\n";
        });
    }).wait();

    return 0;
}