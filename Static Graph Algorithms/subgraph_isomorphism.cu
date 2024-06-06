
#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include<cmath>
#include<algorithm>
#include<cuda.h>
#include"graph.hpp"


__device__ bool isFeasible(int u, int v, int * gpu_OA_pattern, int * gpu_edgeList_pattern, int V_pattern, int E_pattern, int * gpu_OA_target, int * gpu_edgeList_target, int V_target,  int E_target, int* mapping, int* reverseMapping) 
{
    for (int i = gpu_OA_pattern[u]; i < gpu_OA_pattern[u + 1]; ++i) {
        int patternNeighbor = gpu_edgeList_pattern[i];
        if (mapping[patternNeighbor] != -1) {
            bool found = false;
            for (int j = gpu_OA_target[v]; j < gpu_OA_target[v + 1]; ++j) {
                if (mapping[patternNeighbor] == gpu_edgeList_target[j]) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
    }
    return true;
}



__global__ void VF2PlusKernel(int * gpu_OA_pattern, int * gpu_edgeList_pattern, int V_pattern, int E_pattern, int * gpu_OA_target, int * gpu_edgeList_target, int V_target, int E_target, int* mapping, int* reverseMapping, bool* result, int depth) 

{
    int u = -1;
    for (int i = 0; i < V_pattern; ++i) {
        if (mapping[i] == -1) {
            u = i;
            break;
        }
    }

    if (u == -1) return;

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (v < V_target && !(*result)) {
        if (reverseMapping[v] == -1 && isFeasible(u, v, gpu_OA_pattern, gpu_edgeList_pattern, V_pattern, E_pattern, gpu_OA_target, gpu_edgeList_target, V_target,  E_target , mapping, reverseMapping)) {
            mapping[u] = v;
            reverseMapping[v] = u;

            if (depth + 1 == V_pattern ) {
                *result = true;
            } else {
            
                VF2PlusKernel<<<V_target / 256 + 1, 256>>>( gpu_OA_pattern, gpu_edgeList_pattern, V_pattern, E_pattern, gpu_OA_target, gpu_edgeList_target, V_target,  E_target, mapping, reverseMapping, result, depth + 1);
                cudaDeviceSynchronize();
            }

            mapping[u] = -1;
            reverseMapping[v] = -1;
        }
    }
}


void Compute_subgraph(int * OA_pattern, int * edgeList_pattern, int V_pattern, int E_pattern,int * OA_target, int * edgeList_target, int V_target, int E_target)
{
  
 // printf("hi from function\n");   
 
   int *gpu_edgeList_pattern;
   int *gpu_OA_pattern;
   
   int *gpu_edgeList_target;
   int *gpu_OA_target;
  // printf("V inside fun =%d",V);
  
  
    int* d_mapping;
    int* d_reverseMapping;
    bool* d_result;
  
  cudaMalloc( &gpu_OA_pattern, sizeof(int) * (1+V_pattern) );
  cudaMalloc( &gpu_edgeList_pattern, sizeof(int) * (E_pattern) );
  
  cudaMalloc( &gpu_OA_target, sizeof(int) * (1+V_target) );
  cudaMalloc( &gpu_edgeList_target, sizeof(int) * (E_target) );
  
  
   cudaMalloc(&d_mapping, V_pattern * sizeof(int));
    cudaMalloc(&d_reverseMapping, V_target * sizeof(int));
    cudaMalloc(&d_result, sizeof(bool));

    cudaMemset(d_mapping, -1, V_pattern * sizeof(int));
    cudaMemset(d_reverseMapping, -1, V_target * sizeof(int));
    cudaMemset(d_result, 0, sizeof(bool));
  
  
  unsigned int block_size;
	unsigned int num_blocks;
 
  
 
  cudaMemcpy(gpu_OA_pattern, OA_pattern, sizeof(int) * (1+V_pattern), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_edgeList_pattern, edgeList_pattern, sizeof(int) * (E_pattern), cudaMemcpyHostToDevice);
  
  cudaMemcpy(gpu_OA_target, OA_target, sizeof(int) * (1+V_target), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_edgeList_target, edgeList_target, sizeof(int) * (E_target), cudaMemcpyHostToDevice);
  
  printf("before kernel\n");
  
  VF2PlusKernel<<<(V_target + 255) / 256, 256>>>(gpu_OA_pattern, gpu_edgeList_pattern, V_pattern, E_pattern, gpu_OA_target, gpu_edgeList_target, V_target, E_target, d_mapping,    d_reverseMapping, d_result, 0);
  cudaDeviceSynchronize();
  
    bool result;
    cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    if (result) {
        std::cout << "Subgraph isomorphism found!" << std::endl;
    } else {
        std::cout << "No subgraph isomorphism." << std::endl;
    }

 }


int main()
{


//-----------------------------------------------------------------------------------------

  graph pattern("/home/ashwina/Subgraph_isomorphism/cuda/input2.txt");  //this will be changed
  pattern.parseGraph();
  
  int V_pattern = pattern.num_nodes();
  
 // printf("number pf nodes =%d",V);
  
  int E_pattern = pattern.num_edges();
  
//  printf("number pf edges =%d",E);
  

  int *OA_pattern;
  int *edgeList_pattern;
  
  
   OA_pattern = (int *)malloc( (V_pattern+1)*sizeof(int));
   edgeList_pattern = (int *)malloc( (E_pattern)*sizeof(int));
  
    
  for(int i=0; i<= V_pattern; i++) {
    int temp = pattern.indexofNodes[i];
    OA_pattern[i] = temp;
  }
  
  for(int i=0; i< E_pattern; i++) {
    int temp = pattern.edgeList[i];
    edgeList_pattern[i] = temp;
  }
  
 
 //-----------------------------------------------------------------------------------------------------------
 
  graph target("/home/ashwina/Subgraph_isomorphism/cuda/input1.txt");  //this will be changed
  target.parseGraph();
  
  int V_target = target.num_nodes();
  
 // printf("number pf nodes =%d",V);
  
  int E_target = target.num_edges();
  
//  printf("number pf edges =%d",E);
  

  int *OA_target;
  int *edgeList_target;
  
  
   OA_target = (int *)malloc( (V_target+1)*sizeof(int));
   edgeList_target = (int *)malloc( (E_target)*sizeof(int));
  
    
  for(int i=0; i<= V_target; i++) {
    int temp = target.indexofNodes[i];
    OA_target[i] = temp;
  }
  
  for(int i=0; i< E_target; i++) {
    int temp = target.edgeList[i];
    edgeList_target[i] = temp;
  } 
  
 //------------------------------------------------------------------------------------------------------------------------- 
  Compute_subgraph(OA_pattern, edgeList_pattern, V_pattern, E_pattern, OA_target, edgeList_target, V_target, E_target);
  return 0;

}
