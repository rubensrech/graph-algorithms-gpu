#include <fstream>
#include <iostream>
// C++ Program for Floyd Warshall Algorithm  
//#include <bits/stdc++.h> 
#include <sys/time.h>
using namespace std; 

#include <cuda_runtime.h>
  
/* Define Infinite as a large enough 
value.This value will be used for  
vertices not connected to each other */
#define INF 99999  
  
// A function to print the solution matrix  
void printSolution(int ** dist, int nNodes);  

double GetTime(void)
{
   struct  timeval time;
   double  Time;

   gettimeofday(&time, (struct timezone *) NULL);
   Time = ((double)time.tv_sec*1000000.0 + (double)time.tv_usec);
   return(Time);
}

// Solves the all-pairs shortest path  
// problem using Floyd Warshall algorithm  
__global__ void floydWarshall_kernel(int *dist, int nNodes) {  
    /* Add all vertices one by one to  
    the set of intermediate vertices.  
    ---> Before start of an iteration,  
    we have shortest distances between all  
    pairs of vertices such that the  
    shortest distances consider only the  
    vertices in set {0, 1, 2, .. k-1} as 
    intermediate vertices.  
    ----> After the end of an iteration,  
    vertex no. k is added to the set of  
    intermediate vertices and the set becomes {0, 1, 2, .. k} */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (k >= nNodes) return;
    if (i >= nNodes) return;

    // Pick all vertices as destination for the  
    // above picked source  
    for (int j = 0; j < nNodes; j++) {  
        // If vertex k is on the shortest path from  
        // i to j, then update the value of dist[i][j]  
        if (dist[i * nNodes + k] + dist[k * nNodes + j] < dist[i * nNodes + j])  
            dist[i * nNodes + j] = dist[i * nNodes + k] + dist[k * nNodes + j];  
    }  
}

int main(int argc, char **argv){
    double timeElapsed, clockBegin;
	int** graph;
    int a, b, w, nNodes;
    
    if (argc < 2) {
        cout << "Usage: ./" << argv[0] << " <graph>" << endl;
        exit(-1);
    }

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

    /* Floyd initialization */
    int **dist, *dist_d;

	dist = new int*[nNodes];
	for (int i = 0; i < nNodes; ++i)
		dist[i] = new int[nNodes];
    /* dist[][] will be the output matrix  
    that will finally have the shortest  
    distances between every pair of vertices */
  
    /* Initialize the solution matrix same  
    as input graph matrix. Or we can say  
    the initial values of shortest distances 
    are based on shortest paths considering  
    no intermediate vertex. */
    for (int i = 0; i < nNodes; i++)  
        for (int j = 0; j < nNodes; j++)  
            dist[i][j] = graph[i][j];

    cudaMalloc(&dist_d, nNodes * nNodes * sizeof(int));
    for (int i = 0; i < nNodes; i++)
        cudaMemcpy(&dist_d[i * nNodes], dist[i], nNodes * sizeof(int), cudaMemcpyHostToDevice);
    
    /* Floyd execution */
    clockBegin = GetTime();
    dim3 blockSize(32, 32);
    dim3 numBlocks((nNodes + blockSize.x - 1) / blockSize.x, (nNodes + blockSize.y - 1) / blockSize.y);
    floydWarshall_kernel<<<numBlocks, blockSize>>>(dist_d, nNodes);
    cudaDeviceSynchronize();
    timeElapsed = (GetTime() - clockBegin)/1000000;
    
    for (int i = 0; i < nNodes; i++)
        cudaMemcpy(dist[i], &dist_d[i * nNodes], nNodes * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the shortest distance matrix  
    printSolution(dist, nNodes);  

    printf("Computation time: %5lf\n", timeElapsed);

    return 0;  
}    
  
/* A utility function to print solution */
void printSolution(int **dist, int nNodes)  
{  
    for (int i = 0; i < nNodes; i++)  
    {  
        for (int j = 0; j < nNodes; j++)  
        {  
            if (dist[i][j] == INF)  
                cout<<"INF"<<"     ";  
            else
                cout<<dist[i][j]<<"     ";  
        }  
        cout<<endl;  
    }  
}  
  
// This code is contributed by rathbhupendra 

