#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
//#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <iostream>
#include <cstdint>

struct saxpy_functor : public thrust::binary_function<float,float,float>
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

	__host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};

int main(int argc, char *argv[])
{
  std::size_t N = 1<<20;

  thrust::host_vector<float> host_a(N,1.f);
  thrust::host_vector<float> host_b(N,2.f);
  const float scale = 42.f;

  thrust::device_vector<float> dev_a = host_a;
  thrust::device_vector<float> dev_b = host_b;

  thrust::transform(dev_a.begin(), dev_a.end(), // input range #1
  		    dev_b.begin(),              // input range #2
  		    dev_a.begin(),              // output range
  		    saxpy_functor(scale));      // placeholder expression
  
  // thrust::transform(thrust::system::cuda::par,
  // 		    dev_a.begin(), dev_a.end(), // input range #1
  //    		    dev_b.begin(),              // input range #2
  //    		    dev_a.begin(),              // output range
  //    		    saxpy_functor(scale));      // placeholder expression
		    
  host_a = dev_a;
  
  float max_error = 0.0f;
  for (const float& item : host_a )
    max_error = std::max(max_error, std::abs(item-44.0f));

  std::cout << "Max error: " << max_error << std::endl;
  return 0;
}
