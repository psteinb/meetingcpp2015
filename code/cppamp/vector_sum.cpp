// RUN: clang++ `clamp-config --cxxflags --ldflags` vector_sum.cpp -o vector_sum
#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath>
#include "amp.h"

template<typename _type>
void amp_sum(std::vector<_type>& _va,
	     const std::vector<_type>& _vb,
	     _type _scale)
{

  concurrency::extent<1> e_a(_va.size()), e_b(_vb.size());

  concurrency::array_view<_type, 1> av_a(e_a, _va); 
  concurrency::array_view<const _type, 1> av_b(e_b, _vb); 
  
  parallel_for_each(av_a.get_extent(), [=](concurrency::index<1> idx) restrict(amp)
		    {
		      av_a[idx] = av_a[idx]*_scale + av_b[idx]  ;
		    }
		    );

  av_a.synchronize();
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

  amp_sum(host_a,host_b,host_d);
    
  float max_error = 0.0f;
  for (const float& item : host_a )
    max_error = std::max(max_error, std::abs(item-44.0f));
  
  std::cout << "Max error: " << max_error << std::endl;
  
  return 0;
}
