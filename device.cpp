#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
    queue q;
    std::cout << "Running on: " 
              << q.get_device().get_info<info::device::name>() << "\n";
    return 0;
}
