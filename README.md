# opencl
A place to store my work in Open Computing Language. I'll be updating this with simple programs as I explore the language.

## add_numbers:
 
*note: this example is property of Matthew Scarpino - code was downloaded from his article "A Gentle Introduction to OpenCL" on www.drdobbs.com.*
This simple kernel gives an example of how to use separate work items to complete data-independent tasks in parallel using OpenCL library calls. 

## matrix_mult:

Multiply two square matrices with randomly generated data. Each output matrix element is computed by one work-group. Each work-group uses one work-item for each element-wise multiplicaiton. A local memory fence is used to synchronize work-items before summing their individual products. A global memory fence is used to synchronize each work group before returning the result matrix to host memory.

## perlin:
*in progress*
Compute perlin noise on a 2 dimensional grid.

