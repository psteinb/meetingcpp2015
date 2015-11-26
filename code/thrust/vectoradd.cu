#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <iostream>
#include <cstdint>

// allows us to use "_1" instead of "thrust::placeholders::_1"
using namespace thrust::placeholders;

int main(int argc, char *argv[])
{
  std::size_t N = 1<<20;

  thrust::host_vector<float> host_a(N,1.f);
  thrust::host_vector<float> host_b(N,2.f);
  const float scale = 42.f;

  thrust::device_vector<float> dev_a = host_a;
  thrust::device_vector<float> dev_b = host_b;

  thrust::transform(dev_a.begin(), dev_a.end(),  // input range #1
		    dev_b.begin(),           // input range #2
		    dev_a.begin(),           // output range
		    scale * _1 + _2);        // placeholder expression
  return 0;
}
