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

**[github.com/psteinb/meetingcpp2015](https://github.com/psteinb/meetingcpp2015)**
</center>

. . .  

<center>
If not stated otherwise, the slides and all it's code is licensed under

__Creative Commons Attribution 4.0 International License__ ([CC-BY 4.0](http://creativecommons.org/licenses/by/4.0/))
</center>


## Outline

<center>
1. Massively Parallel Programming

2. Architecture

3. What can you use today

4. What can you use tomorrow
</center>


# Massively Parallel Programming

## Why all the fuzz??

<center>
![](data/201x_acc_fraction_1200x.png)  
Data obtained from [Top500.org](www.Top500.org)
</center>

## Food Hunt

[columns,class="row vertical-align"]

[column,class="col-xs-6"]

<!-- https://commons.wikimedia.org/wiki/File:Thunnus_orientalis_(Osaka_Kaiyukan_Aquarium).jpg -->
<center>
Tuna  
![](img/1024px-Thunnus_orientalis_Osaka_Kaiyukan_Aquarium.jpg)  
(fast, single, versatile)
</center>

[/column]

. . .

[column,class="col-xs-6"]

<!-- TODO -->
<center>
Forage Fish  
![](img/forage_fish.jpg)
(small, many, use wakefield of neighbor)
</center>

[/column]

[/columns]



## The same principle on die

[columns,class="row vertical-align"]

[column,class="col-xs-6"]

<!-- TODO -->
<center>
CPU  
![](img/Central-Processing-Unit_x600.jpeg)
</center>

[/column]

[column,class="col-xs-6"]

<!-- TODO -->
<center>
GPU  
![](img/titan_x_small_x600.png)
</center>

[/column]

[/columns]


## Vendor Options

<!-- TODO: image origins -->
[columns,class="row vertical-align"]

[column,class="col-xs-4"]

<center>
Nvidia Tesla  
![](img/Nvidia-Tesla-K80_x400.jpg)

GPU without Graphics
</center>

[/column]

[column,class="col-xs-4"]

<center>
AMD FirePro  
![](img/amd-firepro-s9150-server-graphics_x400.png)

GPU without Graphics
</center>

[/column]

[column,class="col-xs-4"]

<center>
Intel MIC  
![](img/xeon_phi_x400.jpg)

Not Covered Today!
</center>

[/column]

[/columns]


## Vendor flag ships

<!-- TODO: image origins -->
[columns,class="row vertical-align"]

[column,class="col-xs-4"]

<center>
_Nvidia Tesla K80_ 
![](img/Nvidia-Tesla-K80_x200.jpg)
</center>

[/column]

[column,class="col-xs-4"]

<center>
_AMD FirePro S9170_
![](img/amd-firepro-s9150-server-graphics_x200.png)
</center>

[/column]

[column,class="col-xs-4"]

<center>
Intel Xeon Phi 5110P
![](img/xeon_phi_x200.jpg)
</center>

[/column]

[/columns]


[columns,class="row vertical-align"]

[column,class="col-xs-4"]

* 2x GK210 chipsets
* 2x 12 GB GDDR5 RAM
* 2x 288 GB/s to RAM
* 8.7 TFlops SP
* 2.9 TFlops DP

[/column]

[column,class="col-xs-4"]

* 1x Grenada XT
* 32 GB GDDR5 RAM
* 320 GB/s to RAM
* 5.2 TFlops SP
* 2.6 TFlops DP

[/column]

[column,class="col-xs-4"]

* 62x x86 CPUs
* 8 GB GDDR5 RAM
* 320 GB/s to RAM
* 2.1 TFlops SP
* 1.1 TFlops DP

[/column]

[/columns]

<!-- http://www.theregister.co.uk/2012/05/18/inside_nvidia_kepler2_gk110_gpu_tesla/ -->
# Architecture { data-background="img/nvidia_kepler_die_shot.jpg"} 


## { data-background="img/1200x_islay_overbright.png" data-background-size="1200px" }


## { data-background="img/1200x_islay_overbright_annotated.png" data-background-size="1200px" }

## For simplicity ... 

<!-- http://www.techpowerup.com/img/14-11-17/58a.jpg -->
<center>
![](img/1200x_K80_tech_powerup.jpg)

Nvidia Kepler based
(dominant GPU architecture in HPC installations)
</center>


## A more in-depth look

<div style="text-align: center;margin-top: 4%;">
<object type="image/svg+xml" data="figures/K40.svg"
width="1400" border="0" style="background-color: #FFFFFF;">
</object>
</div>

<center>
Nvidia K40: 15 Streaming Multiprocessors (SMX), 12 GB of GDDR5 DRAM
</center>

## Kepler SMX Close-up

<div style="text-align: center;margin-top: 4%;">
<object type="image/svg+xml" data="figures/GK210_sm.svg"
width="1600" border="0" style="background-color: #FFFFFF;">
</object>
</div>

<center>
CUDA core: 1 fp32 ops / clock <!-- (1/3 fp64 ops / clock) -->
</center>


## SIMT Execution

[columns,class="row vertical-align"]

[column,class="col-xs-2"]

**Warp**

[/column]


[column,class="col-xs-4"]


<object type="image/svg+xml" data="figures/thread.svg"
height="200" border="0">
</object>


[/column]

[column,class="col-xs-8"]

* smallest unit of concurrency: *32 threads*
* thread = single CUDA core
* all threads execute same program

[/column]

[/columns]

. . .  

[columns,class="row vertical-align"]

[column,class="col-xs-2"]

**Block**

[/column]


[column,class="col-xs-4"]


<object type="image/svg+xml" data="figures/thread_block.svg"
height="200" border="0" >
</object>


[/column]

[column,class="col-xs-8"]

* can synchronize (barriers)
* can exchange data (common "shared" memory, etc.)

[/column]

[/columns]


. . .  

[columns,class="row vertical-align"]

[column,class="col-xs-2"]

**Grid**

[/column]


[column,class="col-xs-4"]


<object type="image/svg+xml" data="figures/grid_block.svg"
height="200" border="0" >
</object>


[/column]

[column,class="col-xs-8"]

* grids/blocks serve as work distribution/sharing mechanism on device (occupancy)

[/column]

[/columns]


# What can you use today

## A Word of Warning!

## CUDA

## OpenCL

## thrust

## C++AMP and HC

## Pragma Mafia

# What can you use tomorrow

## Boost.Compute

## Sycle and Spear

## C++17


# Summary

## Image References



# Backup
