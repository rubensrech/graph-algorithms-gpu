#include <iostream>
#include <fstream>
#include <stack>
#include <queue>
#include <sys/time.h>

using namespace std;

#define INF 99999
#define GOAL 5000

double GetTime(void)
{
   struct  timeval time;
   double  Time;
   
   gettimeofday(&time, (struct timezone *) NULL);
   Time = ((double)time.tv_sec*1000000.0 + (double)time.tv_usec);
   return(Time);
}


int main(int argc, char **argv){
	double timeElapsed, clockBegin;
	int** graph;
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
	
    /* BFS */
	bool *visited = new bool[nNodes];
	for(int i = 0; i < nNodes; i++)
		visited[i] = false;

   	queue<int> q;
	q.push(0);
	visited[0] = true;

    clockBegin = GetTime();

	/* BFS */
	while(!q.empty()){
		int v = q.front();
		q.pop();
        #if DEBUG == 1
		    cout << "visited " << v << endl;
        #endif
		if(v == GOAL)
		{
			cout << "Found " << GOAL << endl;
			timeElapsed = (GetTime() - clockBegin)/1000000;
			printf("Computation time: %5lf\n", timeElapsed);
			return 0;
		}
		for(int i = 0; i < nNodes; i++)
			if(graph[v][i] != INF && v != i)
				if(visited[i] == false)
				{
					visited[i] = true;
					q.push(i);
				}
	}

    timeElapsed = (GetTime() - clockBegin)/1000000;
	printf("%5lf\n", timeElapsed);

	for(int i = 0; i < nNodes; i++)
		free(graph[i]);
	free(visited);
}

