---
title: C++ on GPUs done right?
author: Peter Steinbach
origin: Max Planck Institute of Molecular Cell Biology and Genetics, Dresden
email: steinbac@mpi-cbg.de
date: Meeting C++, December 05, 2015
---


# Disclaimer(s)



## No OpenGL/Vulkan here!

[columns,class="row vertical-align"]

[column,class="col-xs-6"]

![](img/OpenGL_Nov14/OpenGL_White_500px_Nov14.png)

[/column]

[column,class="col-xs-6"]

![](img/Vulkan_Mar15/Vulkan_White_500px_Mar15.png)

[/column]

[/columns]

<center>
&copy; [Khronos Group](https://www.khronos.org/news/logos/)
</center>

## This is Open-Source!

<center>
Feel free to reply, discuss, inform, correct, ...   

[github.com/psteinb/meetingcpp2015](https://github.com/psteinb/meetingcpp2015)

</center>

<center>
If not stated otherwise, the slides and all it's code is licensed under

__Creative Commons Attribution 4.0 International License__ ([CC-BY 4.0](http://creativecommons.org/licenses/by/4.0/))
</center>


## Outline

<center>
1. Massively Parallel Programming

2. Look and Feel

3. GPGPU Landscape

4. Outlook
</center>

# Massively Parallel Programming


## Food Hunt

[columns,class="row vertical-align"]

[column,class="col-xs-6"]

<center>
Tuna  
![](img/tuna_x600.jpg)  
(fast, single, versatile)
</center>

[/column]

[column,class="col-xs-6"]

<center>
Forage Fish  
![](img/forage_fish_x600.jpg)
(small, many, parallel)
</center>

[/column]

[/columns]



## The same principle on die

[columns,class="row vertical-align"]

[column,class="col-xs-6"]

<center>
CPU  
![](img/Central-Processing-Unit_x600.jpeg)
</center>

[/column]

[column,class="col-xs-6"]

<center>
GPU  
![](img/titan_x_small_x600.png)
</center>

[/column]

[/columns]


## Vendor Options

[columns,class="row vertical-align"]

[column,class="col-xs-4"]

<center>
Nvidia Tesla  
![](img/Nvidia-Tesla-K80_x400.jpg)
</center>

[/column]

[column,class="col-xs-4"]

<center>
AMD FirePro  
![](img/amd-firepro-s9150-server-graphics_x400.png)
</center>

[/column]

[column,class="col-xs-4"]

<center>
Intel MIC  
![](img/xeon_phi_x400.jpg)
</center>

[/column]

[/columns]

