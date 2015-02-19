/*  OpenCL implementation of matrix multiplication on two 64 x 64 matrices
	
	The point of this exercise is to develop rudimentary OpenCL programming 
	skills and a familiarity with OCL structures and concepts.
	
	I'll be racing the GPU against the CPU by performing the multiplication 
	on both devices at the same time

*/

#define PROGRAM_NAME "matrix_mult.cl"
#define KERNEL_FUNC "matrix_mult"
#define MATRIX_SIZE 3ul

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

	cl_program program = clCreateProgramWithSource(context, 1, 
		(const char**)&program_buffer, &program_size, &err);
	if(err < 0) {
		printf("Couldn't load program from source\n");
	}

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
		if(err < 0) {
		printf("Error building program\n");
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
	/*
	clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(int), NULL, &param_value_size);
   	char *device_name = (char *)malloc(param_value_size + 1);
   	device_name[param_value_size] = '\0';
   	clGetDeviceInfo(dev, CL_PLATFORM_VERSION, param_value_size, device_name, NULL);
   	printf("Device name: %s\n", device_name);
	*/
	if(err < 0) {
		printf("Couldn't access any devices of type %llu\n", type);
	}
	return device;

}

void make_matrix(float* ra, size_t n) {
	for(int i = 0; i < n*n; i++) {
		ra[i] = 1.0f; //rand() / (float)RAND_MAX;
	}
}

void get_info(cl_device_id dev) {
 	
 	size_t param_value_size;
   
   	clGetPlatformInfo(0, CL_PLATFORM_NAME, sizeof(int), NULL, &param_value_size);
   	char *platform_name = (char *)malloc(param_value_size + 1);
   	platform_name[param_value_size] = '\0';
   	clGetPlatformInfo(0, CL_PLATFORM_NAME, param_value_size, platform_name, NULL);
   	printf("Platform name: %s\n", platform_name);

   	clGetPlatformInfo(0, CL_PLATFORM_PROFILE, sizeof(int), NULL, &param_value_size);
   	char *platform_profile = (char *)malloc(param_value_size + 1);
   	platform_profile[param_value_size] = '\0';
   	clGetPlatformInfo(0, CL_PLATFORM_PROFILE, param_value_size, platform_profile, NULL);
   	printf("Platform version: %s\n", platform_profile);

   	clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(int), NULL, &param_value_size);
   	char *device_name = (char *)malloc(param_value_size + 1);
   	device_name[param_value_size] = '\0';
   	clGetDeviceInfo(dev, CL_DEVICE_NAME, param_value_size, device_name, NULL);
   	printf("Device name: %s\n", device_name);

   	clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(int), NULL, &param_value_size);
   	char *device_vendor = (char *)malloc(param_value_size + 1);
   	device_vendor[param_value_size] = '\0';
   	clGetDeviceInfo(dev, CL_DEVICE_VENDOR, param_value_size, device_vendor, NULL);
   	printf("Device vendor: %s\n", device_vendor);

   	unsigned long max_work_item_sizes[3];
	clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), 
   		&max_work_item_sizes, NULL);
	printf("Max work item size: %lu / %lu / %lu\n", 
		max_work_item_sizes[0], max_work_item_sizes[1], max_work_item_sizes[2]);
	printf("\n");
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

	/* initialize data */
	float a[MATRIX_SIZE];
	make_matrix(a, MATRIX_SIZE);
	float b[MATRIX_SIZE];
	make_matrix(b, MATRIX_SIZE);
	float c[MATRIX_SIZE*MATRIX_SIZE] = {0};

	/* and result buffer */
	float* result = (float*)malloc(MATRIX_SIZE*MATRIX_SIZE*sizeof(float));

	/* create contexts and devices */
	devices[0] = create_device(CL_DEVICE_TYPE_CPU);
	devices[1] = create_device(CL_DEVICE_TYPE_GPU);
   	
   	get_info(devices[1]);

	context = clCreateContext(NULL, 2, devices, NULL, NULL, &err);

	if(err < 0) {
		printf("Couldn't create one or more contexts");
	}

	/* compile program */
	program = build_program(context, devices[1], PROGRAM_NAME);

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
      	exit(1);
   	}

   	/* create kernel arguments */
   	a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
        CL_MEM_COPY_HOST_PTR, MATRIX_SIZE * sizeof(float), a, &err);
   	if(err < 0) {
    	perror("Couldn't create a buffer");
      	exit(1);   
   	}

   	b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
   		CL_MEM_COPY_HOST_PTR, MATRIX_SIZE * sizeof(float), b, &err);
   	if(err < 0) {
    	perror("Couldn't create b buffer");
      	exit(1);   
   	}

   	c_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
        CL_MEM_COPY_HOST_PTR, MATRIX_SIZE * sizeof(float), c, &err);
   	if(err < 0) {
    	perror("Couldn't create c buffer");
      	exit(1);   
   	}
   	
   	for(int i = 0; i < MATRIX_SIZE; i++) {
   		for(int j = 0; j < MATRIX_SIZE; j++)
   			printf("%f   ", a[i*MATRIX_SIZE+j]);
   		printf("\n");
   	}
   	printf("\n");
   	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buffer);
   	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buffer);
   	err |= clSetKernelArg(kernel, 2, sizeof(float), NULL);
   	err |= clSetKernelArg(kernel, 3, sizeof(float)*MATRIX_SIZE, NULL);
   	err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &c_buffer);
   	if(err < 0) {
      	perror("Couldn't create a kernel argument");
      	exit(1);
   	}

   	/* 
	enqueue kernel

   	group size should be MATRIX_SIZE x MATRIX_SIZE
   		because that is the size of the output
   	
   	local size should be MATRIX_SIZE because that is
   		the number of operations per element of the output 
   	*/
   	size_t global_size[2] = {MATRIX_SIZE*MATRIX_SIZE, MATRIX_SIZE};
   	size_t local_size[2] = {MATRIX_SIZE, 1};
   	err = clEnqueueNDRangeKernel(gpu_queue, kernel, 2, NULL, global_size, 
        local_size, 0, NULL, NULL);

   	if(err < 0) {
      	perror("Couldn't enqueue the kernel");
      	printf("%d\n", err);
      	exit(1);
   	}

   	/* Read the kernel's output */
   	err = clEnqueueReadBuffer(gpu_queue, c_buffer, CL_TRUE, 0, 
        MATRIX_SIZE*sizeof(float), result, 0, NULL, NULL);
   	if(err < 0) {
      	perror("Couldn't read the buffer");
      	exit(1);
   	}

	printf("Printing host representation of a:\n");
   	for(int i = 0; i < MATRIX_SIZE; i++) {
   		for(int j = 0; j < MATRIX_SIZE; j++)
   			printf("%f   ", a[i*MATRIX_SIZE+j]);
   		printf("\n");
   	}
   	printf("\n");

   	printf("Printing host representation of b:\n");
   	for(int i = 0; i < MATRIX_SIZE; i++) {
   		for(int j = 0; j < MATRIX_SIZE; j++)
   			printf("%f   ", b[i*MATRIX_SIZE+j]);
   		printf("\n");
   	}
   	printf("\n");

	printf("Printing host representation of c:\n");
   	for(int i = 0; i < MATRIX_SIZE; i++) {
   		for(int j = 0; j < MATRIX_SIZE; j++)
   			printf("%f   ", result[i*MATRIX_SIZE+j]);
   		printf("\n");
   	}
   	printf("\n");

   	printf("*** KERNEL WORKS, USING BUFFERS TO PASS VALUES BACK TO HOST DOES NOT! ***\n\n");

   	/* Deallocate resources */
   clReleaseKernel(kernel);
   clReleaseMemObject(a_buffer);
   clReleaseMemObject(b_buffer);
   clReleaseMemObject(c_buffer);
   clReleaseCommandQueue(cpu_queue);
   clReleaseCommandQueue(gpu_queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}