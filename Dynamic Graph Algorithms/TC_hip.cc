#include <vector>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "graph.hpp"
#include "libcuda.cuh"
#include <hip/hip_runtime.h>

// kernel for computing TC
// edges are u->v, v->w & w->u
__global__ void tc_kernel(int V, int *d_meta, int *d_data, unsigned long long *d_Tcount)
{
  unsigned int u = blockDim.x * blockIdx.x + threadIdx.x;
  if (u < V)
  {
    for (int idx1 = d_meta[u]; idx1 < d_meta[u + 1]; idx1++)
    {
      int v = d_data[idx1];
      if (u < v)
      {
        for (int idx2 = d_meta[v]; idx2 < d_meta[v + 1]; idx2++)
          int w = d_data[idx2];
          if (v < w)
          {
            // check for w->u edge
            for (int idx3 = d_meta[w]; idx3 < d_meta[w + 1]; idx3++)
            {
              if (d_data[idx3] == u)
                atomicAdd(d_Tcount, 1);
            }
          }
        }
      }
    }
  }
}




// kernel for computing the number of triangles added/removed as edges get added/deleted
__global__ void count_triangles(int batchSize, int E, update *d_batch, bool *d_modified,
                                int *d_meta, int *d_data, int *d_diffMeta, int *d_diffData,
                                unsigned long long *d_count1, unsigned long long *d_count2, unsigned long long *d_count3)
{
  unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < batchSize)
  {
    int u = d_batch[id].source;
    int v = d_batch[id].destination;

    for (int idx1 = d_meta[v]; idx1 < d_meta[v + 1]; idx1++)
    {
      int w = d_data[idx1];
      if (w != INT_MAX && u != v && u != w && v != w)
      {
        int count = 1;
        bool isTriangle = false;

         if (d_modified[idx1])
          count += 1;

        if (__isEdge(w, u, d_meta, d_data, d_diffMeta, d_diffData))
        {
          isTriangle = true;
          int idx2 = __getEdge(w, u, E, d_meta, d_data, d_diffMeta, d_diffData);
          if (d_modified[idx2])
            count += 1;
        }

        if (isTriangle)
        {
          if (count == 1)
            atomicAdd(d_count1, 1);
          else if (count == 2)
            atomicAdd(d_count2, 1);
          else if (count == 3)
            atomicAdd(d_count3, 1);
        }
      }
    }
for (int idx1 = d_diffMeta[v]; idx1 < d_diffMeta[v + 1]; idx1++)
    {
      int w = d_diffData[idx1];
      if (w != INT_MAX && u != v && u != w && v != w)
      {
        int count = 1;
        bool isTriangle = false;
        if (d_modified[E + idx1])
          count += 1;

        if (__isEdge(w, u, d_meta, d_data, d_diffMeta, d_diffData))
        {
          isTriangle = true;
          int idx2 = __getEdge(w, u, E, d_meta, d_data, d_diffMeta, d_diffData);
          if (d_modified[idx2])
            count += 1;
        }

        if (isTriangle)
        {
          if (count == 1)
            atomicAdd(d_count1, 1);
          else if (count == 2)
             atomicAdd(d_count2, 1);
          else if (count == 3)
            atomicAdd(d_count3, 1);
        }
      }
    }
  }
}


unsigned long long Compute_TC(graph &g)
{
  // data on host
  int V = g.num_nodes();
  int E = g.num_edges();

  int *h_meta = g.indexofNodes;
  int *h_data = g.edgeList;
  unsigned long long h_Tcount = 0;
// data on device
  int *d_meta;
  int *d_data;
  unsigned long long *d_Tcount;


  // allocate memory on device
  hipMalloc(&d_meta, sizeof(int) * (V + 1));
  hipMalloc(&d_data, sizeof(int) * (E));
  hipMalloc(&d_Tcount, sizeof(unsigned long long));

  // copy to device
  hipMemcpy(d_meta, h_meta, sizeof(int) * (V + 1), hipMemcpyHostToDevice);
  hipMemcpy(d_data, h_data, sizeof(int) * (E), hipMemcpyHostToDevice);
  hipMemcpy(d_Tcount, &h_Tcount, sizeof(unsigned long long), hipMemcpyHostToDevice);

  // launch config
  const int numThreads = 1024;
  const int numBlocks = (V + numThreads - 1) / numThreads;

// call kernel to compute the triangle counts
  tc_kernel<<<numBlocks, numThreads>>>(V, d_meta, d_data, d_Tcount);
  hipDeviceSynchronize();

  // copy distances back to host
  hipMemcpy(&h_Tcount, d_Tcount, sizeof(unsigned long long), hipMemcpyDeviceToHost);

  // free up the memory
  hipFree(d_meta);
  hipFree(d_data);
  hipFree(d_Tcount);

  return h_Tcount;
}

unsigned long long Compute_dynamicTCDel(graph &g, unsigned long long triangleCount,
                                        bool *modified, std::vector<update> &deleteBatch)
{

// data on host
  int V = g.num_nodes();
  int E = g.num_edges_CSR();
  int diffE = g.num_edges_diffCSR();
  int batchSize = deleteBatch.size();

  int *h_meta = g.indexofNodes;
  int *h_data = g.edgeList;
  int *h_diffMeta = g.diff_indexofNodes;
  int *h_diffData = g.diff_edgeList;

  unsigned long long h_Tcount = triangleCount;
  unsigned long long h_count1, h_count2, h_count3;
  h_count1 = h_count2 = h_count3 = 0;

  // data on device
  update *d_deleteBatch;
  bool *d_modified;
  int *d_meta;
  int *d_data;
  int *d_diffMeta;
  int *d_diffData;

unsigned long long *d_count1, *d_count2, *d_count3;


  // allocate memory on device
  hipMalloc(&d_deleteBatch, sizeof(update) * (batchSize));
  hipMalloc(&d_modified, sizeof(bool) * (E + diffE));
  hipMalloc(&d_meta, sizeof(int) * (V + 1));
  hipMalloc(&d_data, sizeof(int) * (E));
  hipMalloc(&d_diffMeta, sizeof(int) * (V + 1));
  hipMalloc(&d_diffData, sizeof(int) * (diffE));
  hipMalloc(&d_count1, sizeof(unsigned long long));
  hipMalloc(&d_count2, sizeof(unsigned long long));
  hipMalloc(&d_count3, sizeof(unsigned long long));

  // copy to device
  hipMemcpy(d_deleteBatch, deleteBatch.data(), sizeof(update) * (batchSize), hipMemcpyHostToDevice);
  hipMemcpy(d_modified, modified, sizeof(bool) * (E + diffE), hipMemcpyHostToDevice);
  hipMemcpy(d_meta, h_meta, sizeof(int) * (V + 1), hipMemcpyHostToDevice);
  hipMemcpy(d_data, h_data, sizeof(int) * (E), hipMemcpyHostToDevice);

hipMemcpy(d_diffMeta, h_diffMeta, sizeof(int) * (V + 1), hipMemcpyHostToDevice);
  hipMemcpy(d_diffData, h_diffData, sizeof(int) * (diffE), hipMemcpyHostToDevice);
  hipMemcpy(d_count1, &h_count1, sizeof(unsigned long long), hipMemcpyHostToDevice);
  hipMemcpy(d_count2, &h_count2, sizeof(unsigned long long), hipMemcpyHostToDevice);
  hipMemcpy(d_count3, &h_count3, sizeof(unsigned long long), hipMemcpyHostToDevice);

  // launch config
  const int numThreads = 1024;
  const int numBlocks = (batchSize + numThreads - 1) / numThreads;

  // call kernel to compute the reduction in number of triangles
  count_triangles<<<numBlocks, numThreads>>>(batchSize, E, d_deleteBatch, d_modified,

                                                d_meta, d_data, d_diffMeta, d_diffData,
                                             d_count1, d_count2, d_count3);
  hipDeviceSynchronize();

  // copy to host
  hipMemcpy(&h_count1, d_count1, sizeof(unsigned long long), hipMemcpyDeviceToHost);
  hipMemcpy(&h_count2, d_count2, sizeof(unsigned long long), hipMemcpyDeviceToHost);
  hipMemcpy(&h_count3, d_count3, sizeof(unsigned long long), hipMemcpyDeviceToHost);

  hipFree(d_deleteBatch);
  hipFree(d_modified);
  hipFree(d_meta);
  hipFree(d_data);

 hipFree(d_diffMeta);
  hipFree(d_diffData);
  hipFree(d_count1);
  hipFree(d_count2);
  hipFree(d_count3);

  h_Tcount = h_Tcount - (h_count1 / 2 + h_count2 / 4 + h_count3 / 6);
  return h_Tcount;
}

unsigned long long Compute_dynamicTCAdd(graph &g, unsigned long long triangleCount,
                                        bool *modified, std::vector<update> &addBatch)
{
  // data on host
  int V = g.num_nodes();
  int E = g.num_edges_CSR();
  int diffE = g.num_edges_diffCSR();
int batchSize = addBatch.size();

  int *h_meta = g.indexofNodes;
  int *h_data = g.edgeList;
  int *h_diffMeta = g.diff_indexofNodes;
  int *h_diffData = g.diff_edgeList;

  unsigned long long h_Tcount = triangleCount;
  unsigned long long h_count1, h_count2, h_count3;
  h_count1 = h_count2 = h_count3 = 0;

  // data on device
  update *d_addBatch;
  bool *d_modified;
  int *d_meta;
  int *d_data;
  int *d_diffMeta;
  int *d_diffData;
  unsigned long long *d_count1, *d_count2, *d_count3;
// allocate memory on device
  hipMalloc(&d_addBatch, sizeof(update) * (batchSize));
  hipMalloc(&d_modified, sizeof(bool) * (E + diffE));
  hipMalloc(&d_meta, sizeof(int) * (V + 1));
  hipMalloc(&d_data, sizeof(int) * (E));
  hipMalloc(&d_diffMeta, sizeof(int) * (V + 1));
  hipMalloc(&d_diffData, sizeof(int) * (diffE));
  hipMalloc(&d_count1, sizeof(unsigned long long));
  hipMalloc(&d_count2, sizeof(unsigned long long));
  hipMalloc(&d_count3, sizeof(unsigned long long));

  // copy to device
  hipMemcpy(d_addBatch, addBatch.data(), sizeof(update) * (batchSize), hipMemcpyHostToDevice);
  hipMemcpy(d_modified, modified, sizeof(bool) * (E + diffE), hipMemcpyHostToDevice);
  hipMemcpy(d_meta, h_meta, sizeof(int) * (V + 1), hipMemcpyHostToDevice);
  hipMemcpy(d_data, h_data, sizeof(int) * (E), hipMemcpyHostToDevice);
  hipMemcpy(d_diffMeta, h_diffMeta, sizeof(int) * (V + 1), hipMemcpyHostToDevice);
  hipMemcpy(d_diffData, h_diffData, sizeof(int) * (diffE), hipMemcpyHostToDevice);

hipMemcpy(d_count1, &h_count1, sizeof(unsigned long long), hipMemcpyHostToDevice);
  hipMemcpy(d_count2, &h_count2, sizeof(unsigned long long), hipMemcpyHostToDevice);
  hipMemcpy(d_count3, &h_count3, sizeof(unsigned long long), hipMemcpyHostToDevice);

  // launch config
  const int numThreads = 1024;
  const int numBlocks = (batchSize + numThreads - 1) / numThreads;

  // call kernel to compute the number of triangles new triangles
  count_triangles<<<numBlocks, numThreads>>>(batchSize, E, d_addBatch, d_modified,

                                                 d_meta, d_data, d_diffMeta, d_diffData,
                                             d_count1, d_count2, d_count3);
  hipDeviceSynchronize();
  //checkCudaError();

  // copy to host
  hipMemcpy(&h_count1, d_count1, sizeof(unsigned long long), hipMemcpyDeviceToHost);
  hipMemcpy(&h_count2, d_count2, sizeof(unsigned long long), hipMemcpyDeviceToHost);


hipMemcpy(&h_count3, d_count3, sizeof(unsigned long long), hipMemcpyDeviceToHost);


  // free up the memory
  hipFree(d_addBatch);
  hipFree(d_modified);
  hipFree(d_meta);
  hipFree(d_data);
  hipFree(d_diffMeta);
  hipFree(d_diffData);
  hipFree(d_count1);
  hipFree(d_count2);
  hipFree(d_count3);

  h_Tcount = h_Tcount + (h_count1 / 2 + h_count2 / 4 + h_count3 / 6);
  return h_Tcount;
}

int main(int argc, char **argv)
{


  graph G("/home/ashwina/graphsuite/USAud.txt", "hip", false);


  G.parseGraph();
  printf("num of nodes =%llu\n",G.num_nodes());

  unsigned long long triangleCount = Compute_TC(G);
  printf("Initial TC = %llu\n", triangleCount);

  bool *modified_add;
  bool *modified_del;
  int elements = 0;

  std::vector<update> updateEdges = G.parseUpdates("/home/ashwina/graphsuite/dynamic_graphs/update_usaRoadNet.txt");
  std::vector<update> deleteBatch;

std::vector<update> addBatch;

  int percent = atoi(argv[1]);
  int updateSize = percent * 0.01 * G.num_edges();
  if (!G.isGraphDirected() && updateSize & 1)
    updateSize++;
  int batchSize = updateSize;
  // int batchSize = 1000000;

  std::chrono::time_point<std::chrono::high_resolution_clock> startTime, endTime;
  std::chrono::duration<double, std::milli> timeTaken;

  startTime = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < updateSize; k += batchSize)
  {
    if ((k + batchSize) > updateSize)
      elements = updateSize - k;
    else
      elements = batchSize;
 // batch-wise deletion and decremental call
    modified_del = new bool[G.num_edges()];
    for (int i = 0; i < G.num_edges(); i++)
      modified_del[i] = false;

    for (int i = k; i < (k + batchSize) && i < updateSize; i++)
    {
      if (updateEdges[i].type == 'd')
      {
        update u = updateEdges[i];
        int src = u.source;
        int dest = u.destination;
        for (edge e : G.getNeighbors(src))
        {
          if (e.destination == dest)
            modified_del[e.id] = true;
        }
        deleteBatch.push_back(u);
      }
    }


triangleCount = Compute_dynamicTCDel(G, triangleCount, modified_del, deleteBatch);
    G.updateCSR_Del(updateEdges, k, elements);
    deleteBatch.clear();

    // batch-wise addition and incremental call
    G.updateCSR_Add(updateEdges, k, elements);

    modified_add = new bool[G.num_edges()];
    for (int i = 0; i < G.num_edges(); i++)
      modified_add[i] = false;

    for (int i = k; i < (k + batchSize) && i < updateSize; i++)
    {
      if (updateEdges[i].type == 'a')
      {
        update u = updateEdges[i];
        int src = u.source;
        int dest = u.destination;
        for (edge e : G.getNeighbors(src))
        {

if (e.destination == dest)
          {
            modified_add[e.id] = true;
          }
        }
        addBatch.push_back(u);
      }
    }

    triangleCount = Compute_dynamicTCAdd(G, triangleCount, modified_add, addBatch);
    addBatch.clear();

free(modified_add);
    free(modified_del);
  }
  endTime = std::chrono::high_resolution_clock::now();
  timeTaken = endTime - startTime;
  printf("Total time taken for %d percent updates: %.3f ms\n", percent, timeTaken.count());
  printf("Final TC = %llu\n\n", triangleCount);

  return 0;
