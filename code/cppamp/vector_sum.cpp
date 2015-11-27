// RUN: clang++ `clamp-config --cxxflags --ldflags` vector_sum.cpp -o vector_sum
#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath>
#include "amp.h"

void amp_sum(std::vector<float>& _va,
	     const std::vector<float>& _vb,
	     float _scale)
{

  concurrency::extent<1> ext_a(_va.size()), ext_b(_vb.size());

  concurrency::array_view<float, 1> 		view_a(ext_a, _va); 
  concurrency::array_view<const float, 1>	view_b(ext_b, _vb); 
  
  parallel_for_each(view_a.get_extent(),
		    [=](concurrency::index<1> idx) restrict(amp)
		    {
		      view_a[idx] = view_a[idx]*_scale + view_b[idx]  ;
		    }
		    );

  view_a.synchronize();
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

  concurrency::accelerator default_device;
  amp_sum(host_a,host_b,host_d);
    
  float max_error = 0.0f;
  for (const float& item : host_a )
    max_error = std::max(max_error, std::abs(item-44.0f));
  
  std::cout << "Max error: " << max_error << std::endl;
  
  return 0;
}
