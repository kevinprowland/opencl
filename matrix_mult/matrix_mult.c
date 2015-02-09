/*  OpenCL implementation of matrix multiplication on two 64 x 64 matrices
	
	The point of this exercise is to develop rudimentary OpenCL programming 
	skills and a familiarity with OCL structures and concepts.
	
	I'll be racing the GPU against the CPU by performing the multiplication 
	on both devices at the same time

*/

#define PROGRAM_NAME "matrix_mult.cl"
#define KERNEL_FUNC "matrix_mult"
#define MATRIX_SIZE 3

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

cl_program build_program(cl_context context, cl_device_id device, const char* filename) {
	FILE* program_file;
	char* program_buffer, *program_log;
	size_t program_size, log_size;
	int err;

	program_file = fopen(filename, "r");
	if(program_file == NULL) {
		printf("Couldn't find the program file\n");
		exit(1);
	}
	fseek(program_file, 0, SEEK_END);
	program_size = ftell(program_file);
	rewind(program_file);

	program_buffer = malloc(program_size + 1);
	program_buffer[program_size] = '\0';

	fread(program_buffer, sizeof(char), program_size, program_file);
	fclose(program_file);

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&program_buffer,
		&program_size, &err);
	if(err < 0) {
		printf("Couldn't load program from source\n");
	}

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
		if(err < 0) {

      	/* Find size of log and print to std output */
      	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
      	program_log = (char*) malloc(log_size + 1);
      	program_log[log_size] = '\0';
      	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
      	printf("%s\n", program_log);
      	free(program_log);
      	exit(1);
   }
	return program;
}

cl_device_id create_device(cl_device_type type) {
	cl_platform_id platform;
	cl_device_id device;
	int err;

	err = clGetPlatformIDs(1, &platform, NULL);
	if(err < 0) {
		printf("Couldn't find platform(s)\n");
	}

	err = clGetDeviceIDs(platform, type, 1, &device, NULL);
	if(err < 0) {
		printf("Couldn't access any devices of type %llu\n", type);
	}
	return device;

}

float* make_matrix(size_t nn) {
	float* a = (float*)malloc(nn*sizeof(float));
	for(int i = 0; i < nn; i++) {
		a[i] = rand() / (float)RAND_MAX;
	}
	return a;
}

int main() {
	/* OpenCL structures */
	cl_device_id devices[2];
	cl_context context;
	cl_program program;
	cl_kernel kernel;
	cl_command_queue cpu_queue, gpu_queue;
	cl_int err;
	cl_mem a_buffer, b_buffer, c_buffer;
	size_t global_size, local_size;

	/* initialize data */
	float* a = make_matrix(MATRIX_SIZE);
	float* b = make_matrix(MATRIX_SIZE);
	float* c = (float*)malloc(MATRIX_SIZE * sizeof(float));

	/* and result buffer */
	float* result = (float*)malloc(MATRIX_SIZE * sizeof(float));

	/* create contexts and devices */
	devices[0] = create_device(CL_DEVICE_TYPE_CPU);
	devices[1] = create_device(CL_DEVICE_TYPE_GPU);

	context = clCreateContext(NULL, 2, (const cl_device_id*)&devices, NULL, NULL, &err);

	if(err < 0) {
		printf("Couldn't create one or more contexts");
	}

	/* compile program */
	program = build_program(context, devices[0], PROGRAM_NAME);

	/* create command queues */
	cpu_queue = clCreateCommandQueue(context, devices[0], 0, &err);
	if(err < 0) {
		printf("Couldn't create command queue for cpu\n");
	}

	gpu_queue = clCreateCommandQueue(context, devices[1], 0, &err);
	if(err < 0) {
		printf("Couldn't create command queue for gpu\n");
	}

	/* create a kernel */
   	kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   	if(err < 0) {
      	perror("Couldn't create a kernel");
      	printf("%d\n", err);
      	exit(1);
   	}

   	/* create kernel arguments */
   	a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
        CL_MEM_COPY_HOST_PTR, MATRIX_SIZE * sizeof(float), a, &err);
   	b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
   		CL_MEM_COPY_HOST_PTR, MATRIX_SIZE * sizeof(float), b, &err);
   	c_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
        CL_MEM_COPY_HOST_PTR, MATRIX_SIZE * sizeof(float), c, &err);

   	int i = MATRIX_SIZE;
   	int* p = &i;
   	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buffer);
   	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buffer);
   	err |= clSetKernelArg(kernel, 2, sizeof(int), &p);
   	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_buffer);
   	if(err < 0) {
      	perror("Couldn't create a kernel argument");
      	exit(1);
   	}

   	/* 
	enqueue kernel

   	global size should be MATRIX_SIZE x MATRIX_SIZE
   		because that is the size of the output
   	
   	local size should be MATRIX_SIZE because that is
   		the number of operations per element of the output 
   	*/
   	global_size = MATRIX_SIZE * MATRIX_SIZE;
   	local_size = MATRIX_SIZE;
   	err = clEnqueueNDRangeKernel(cpu_queue, kernel, 2, NULL, &global_size, 
        &local_size, 0, NULL, NULL); 
   	if(err < 0) {
      	perror("Couldn't enqueue the kernel");
      	exit(1);
   	}

   	/* Read the kernel's output */
   	err = clEnqueueReadBuffer(cpu_queue, c_buffer, CL_TRUE, 0, 
        MATRIX_SIZE*sizeof(float), result, 0, NULL, NULL);
   	if(err < 0) {
      	perror("Couldn't read the buffer");
      	exit(1);
   	}
}