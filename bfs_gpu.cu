#include <iostream>
#include <fstream>
#include <stack>
#include <queue>
#include <sys/time.h>
#include <vector>
using namespace std;

#include <cuda_runtime.h>


#define INF 99999
#define GOAL 5000
#define DEBUG 0

double GetTime(void)
{
   struct  timeval time;
   double  Time;
   
   gettimeofday(&time, (struct timezone *) NULL);
   Time = ((double)time.tv_sec*1000000.0 + (double)time.tv_usec);
   return(Time);
}


// Referencia: https://m-sp.org/downloads/titech_bfs_cuda.pdf
__global__ void bfs_kernel(int nNodes, int *v_adj_length, int *v_adj_begin, int *v_adj_list, int *dist, bool *running) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int i = 0; i < nNodes; i += num_threads) {
        int v = i + tid;
        if (v < nNodes) {
            for (int j = 0; j < v_adj_length[v]; j++) {
                int neighbor = v_adj_list[v_adj_begin[v] + j];
                if (dist[neighbor] > dist[v] + 1) {
                    dist[neighbor] = dist[v] + 1;
                    *running = true;
                }
            }
        }
    }
}

int main(int argc, char **argv){
	double timeElapsed, clockBegin;
	int **graph;
    int a, b, w, nNodes;

    if (argc < 2) {
        cout << "Usage: ./" << argv[0] << " <graph>" << endl;
        exit(-1);
    }

    /* Initialization */
    ifstream inputfile(argv[1]);
    inputfile >> nNodes;
    graph = new int*[nNodes];
    for (int i = 0; i < nNodes; ++i)
    {
        graph[i] = new int[nNodes]; 
        for (int j = 0; j < nNodes; ++j)
            graph[i][j] = INF;
    }
    while (inputfile >> a >> b >> w)
    {
        graph[a][b] = w;
        graph[b][a] = w;
    }


    vector<int> v_adj_list;
    int *v_adj_length = new int[nNodes];
    int *v_adj_begin = new int[nNodes];

    int list_idx = 0;
    for (int i = 0; i < nNodes; i++) {
        int numNeighbors = 0;
        for (int j = 0; j < nNodes; j++) {
            if (graph[i][j] != INF && i != j) {
                numNeighbors++;
                v_adj_list.push_back(j);
            }
        }
        v_adj_begin[i] = list_idx;
        v_adj_length[i] = numNeighbors;
        list_idx += numNeighbors;
    }

    /* BFS */
    bool false_val = false;
    bool *running, *running_d;
    int *v_adj_length_d, *v_adj_begin_d, *v_adj_list_d;
    int *dist, *dist_d;

    running = new bool[1];
    *running = true;
    cudaMalloc(&running_d, 1 * sizeof(bool));

    cudaMalloc(&v_adj_length_d, nNodes * sizeof(int));
    cudaMalloc(&v_adj_begin_d, nNodes * sizeof(int));
    cudaMalloc(&v_adj_list_d, v_adj_list.size() * sizeof(int));
    cudaMemcpy(v_adj_length_d, v_adj_length, nNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(v_adj_begin_d, v_adj_begin, nNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(v_adj_list_d, v_adj_list.data(), v_adj_list.size() * sizeof(int), cudaMemcpyHostToDevice);

    dist = new int[nNodes];
    cudaMalloc(&dist_d, nNodes * sizeof(int));
    for (int i = 0; i < nNodes; i++) dist[i] = INF;
    dist[0] = 0;
    cudaMemcpy(dist_d, dist, nNodes * sizeof(int), cudaMemcpyHostToDevice);

    clockBegin = GetTime();
    
	while (*running) {
        cudaMemcpy(running_d, &false_val, 1 * sizeof(bool), cudaMemcpyHostToDevice);
        int blockSize = 32;
        int numBlocks = (nNodes + blockSize - 1) / blockSize;
        bfs_kernel<<<numBlocks, blockSize>>>(nNodes, v_adj_length_d, v_adj_begin_d, v_adj_list_d, dist_d, running_d);
        cudaDeviceSynchronize();
        cudaMemcpy(running, running_d, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
    }

    timeElapsed = (GetTime() - clockBegin)/1000000;        
    printf("%5lf\n", timeElapsed);

	for(int i = 0; i < nNodes; i++)
		free(graph[i]);
    free(running);
    free(dist);
    free(v_adj_begin);
    free(v_adj_length);
    cudaFree(running_d);
    cudaFree(dist_d);
    cudaFree(v_adj_begin_d);
    cudaFree(v_adj_length_d);
    cudaFree(v_adj_list_d);
}

