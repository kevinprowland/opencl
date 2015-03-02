/*
	matrix_mult.cl
	@author Kevin Rowland
	Feb 25, 2015

	| Matrix multiplication on an n x n square matrix |

	+ The output matrix size (n^2) is chosen as the workgroup size
	   - Each workgroup therefore computes one element of the output array
	+ Each workgroup contains n work items
	   - Each workitem therefore computes one elementwise multiplication
	+ The local barrier allows us to wait for each elementwise multiplication to 
	finish before adding them together to compute the output element
	+ The global barrier allows us to wait until each output element is computed
	before filling up the output array

*/

__kernel void matrix_mult(__global float* a,
			   			  __global float* b,
			   			  __local  float* local_result,
			   			  __local  float* local_result_vector,
			   			  __global float* c) {

	int matrix_dimen = get_num_groups(0);
	int matrix_dimen_1 = get_num_groups(1);



	int gid_0 = get_group_id(0);
	int gid_1 = get_group_id(1);

	int lid = get_local_id(0);
	//lid_1? No! Work items are 1 dimensional

	//global_id(0) is just gid(0)*lid 
	//global_id(1) likewise

	if(matrix_dimen_1 == 1) { //cpu
		local_result_vector[lid] = a[gid_0]; //TODO 1D range kernel math
	}

	local_result_vector[lid] = a[gid_0*matrix_dimen + lid]*b[lid*matrix_dimen + gid_1];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(lid == 0) {					//if master thread
		(*local_result) = 0.0f;
		for(int i = 0; i < matrix_dimen; i++) {
			(*local_result) += local_result_vector[i];
		}
	}
	c[gid_0*matrix_dimen+gid_1] = (*local_result);

	barrier(CLK_GLOBAL_MEM_FENCE);
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