__kernel void matrix_mult(__global float* a
			   			  __global float* b
			   			  __global int size
			   			  __global float* c) {
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	float local_result;

	local_result += a[gid*size + lid]*b[lid*size + gid];
	printf("global id: %d\t local_id: %d\n", gid, lid);
	c[gid] = local_result;

	barrier(CLK_LOCAL_MEM_FENCE);
	//find determinant?
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