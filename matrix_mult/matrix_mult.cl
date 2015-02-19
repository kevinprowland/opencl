__kernel void matrix_mult(__global float* a,
			   			  __global float* b,
			   			  __local  float* local_result,
			   			  __local  float* local_result_vector,
			   			  __global float* c) {

	int gid_0 = get_group_id(0);
	int gid_1 = get_group_id(1);

	int lid = get_local_id(0);
	//lid_1? No! Work items are 1 dimensional

	//global_id(0) is just gid(0)*lid 
	//global_id(1) likewise

	int matrix_dimen = get_num_groups(0); //TODO use matrix_dimen_1
	int matrix_dimen_1 = get_num_groups(1);

	if((get_global_id(0) == 0) && (get_global_id(1) == 0)) {
		printf("Printing kernel representation of a:\n");
		for(int i = 0; i < matrix_dimen; i++) {
   			for(int j = 0; j < matrix_dimen; j++)
   				printf("%f   ", a[i*matrix_dimen+j]);
   			printf("\n");
   		}
   		printf("\n");
	}
	if((get_global_id(0) == 0) && (get_global_id(1) == 0)) {
		printf("Printing kernel representation of b:\n");
		for(int i = 0; i < matrix_dimen; i++) {
   			for(int j = 0; j < matrix_dimen; j++)
   				printf("%f   ", a[i*matrix_dimen+j]);
   			printf("\n");
   		}
   		printf("\n");
	}


	local_result_vector[lid] = a[gid_0*matrix_dimen + lid]*b[lid*matrix_dimen + gid_1];
	//printf("num groups: %d\n", matrix_dimen);
	//printf("group id: (%d,%d)\t local_id: %d\n", gid_0, gid_1, lid);
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(lid == 0) {					//if master thread
		(*local_result) = 0.0f;
		for(int i = 0; i < matrix_dimen; i++) {
			(*local_result) += local_result_vector[i];
		}
	}
	c[gid_1*matrix_dimen+gid_0] = (*local_result);

	barrier(CLK_GLOBAL_MEM_FENCE);
	if((get_global_id(0) == 0) && (get_global_id(1) == 0)) {
		printf("Printing kernel representation of c:\n");
		for(int i = 0; i < matrix_dimen; i++) {
   			for(int j = 0; j < matrix_dimen; j++)
   				printf("%f   ", c[i*matrix_dimen+j]);
   			printf("\n");
   		}
   		printf("\n");
	}

	
}

/*
0,0		0,1 	0,2 	0,3 	... 	0,63

1,0		1,1		1,2		1,3		...		1,63

2,0		2,1		2,2		2,3		...		2,63

3,0		3,1		3,2		3,3		...		3,63

.										.
.										.

63,0	63,1	63,2	63,3	...		63,63
*/