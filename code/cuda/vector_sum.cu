#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath>

__global__ void vector_sum(std::size_t _size,
			   float _scale,
			   float* _a,
			   float* _b){
  const std::size_t index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index < _size)
    _a[index] = _scale*_a[index] + _b[index];
}

int main(int argc, char *argv[])
{
  std::size_t vector_size = (1<<20);

  if(argc>1)
    vector_size*=std::stoi(argv[1]);

  std::cout << "vector sum: " << vector_size << " elements" << std::endl;
  
  std::vector<float> host_a(vector_size,1.f);
  std::vector<float> host_b(vector_size,2.f);
  const float host_d = 42.f;

  //gpu relevant code
  float * device_a=nullptr, *device_b=nullptr;

  const std::size_t vector_size_byte=vector_size*sizeof(float);
  cudaMalloc(&device_a, vector_size_byte); 
  cudaMalloc(&device_b, vector_size_byte);
  cudaMemcpy(device_a, &host_a[0], vector_size_byte,
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, &host_b[0], vector_size_byte,
             cudaMemcpyHostToDevice);

  vector_sum<<<(vector_size+255)/256, 256>>>(vector_size,
					     host_d,
					     device_a,
					     device_b);

  cudaMemcpy(&host_a[0], device_a, vector_size_byte,
             cudaMemcpyDeviceToHost);

  float max_error = 0.0f;
  for (const float& item : host_a )
    max_error = std::max(max_error, std::abs(item-44.0f));
  
  std::cout << "Max error: " << max_error << std::endl;
   
  cudaFree(device_a);
  cudaFree(device_b);
  return 0;
}
