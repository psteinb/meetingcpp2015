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



# Summary

## Image References






# Backup
