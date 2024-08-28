#ifndef p_disp_h
#define p_disp_h
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
using namespace std;

class MaxDiversityProblem{

    public:

        // Total Number Nodes
        int n;

        // p
        int p;

        // Upper bound
        float upper_bound;

        // Distance Matrix
        vector<vector <float>> distance;

        // Constructor
        MaxDiversityProblem();
        MaxDiversityProblem(string file_name, int ratio = 0){
    
            // Get file
            ifstream input(file_name);
            string line;


            // get first line
            getline(input,line);
            istringstream stream(line);

            // get n
            stream >> n;
            p = (int)ceil(n * 0.01 * ratio);

            // Resize distance matrix
            distance.resize(n);
            for (int i = 0; i < n; i++)
            {
                distance[i].resize(n);
            }
            
            // Fill distance matrix
            upper_bound = 0;    
            int n1, n2;
            float d;
            for (int i = 0; i < (n * (n-1) / 2); i++)
            {
                getline(input,line);
                istringstream stream(line);
                stream >> n1 >> n2 >> d;
                distance[n1][n2] = d;
                distance[n2][n1] = d;
                upper_bound += d;
            }
            input.close();

        }

        // Objective function.  
        // Takes pointer to start of x array, write result to fx
        void f(const vector<int> &x, float &fx){
            // returns f(x)
            // Should pass a pointer to the start of x
            fx = 0;
            for (int n1 = 0; n1<n; n1++){
                for (int n2 = n1 + 1; n2 < n; n2++)
                {
                    fx += distance[n1][n2]*x[n1]*x[n2];
                }
            }
        }

        // Derivative of objective function (overwrites dx)
        void df(const vector<int> &x, vector<float> &dx){
            // re-writes dx
            // Should pass a pointer to the start of x and of dx
            for (int i=0; i<n; i++){
                dx[i] = 0.0;
                for (int j = 0; j < n; j++)
                {
                    dx[i] += distance[i][j]*x[j];
                }
            }
        }
        
};

class MaxDiversityProblemLowMemory{

    public:

        // Total Number Nodes
        int n;
        int coords;

        // p
        int p;

        // Upper bound
        float upper_bound = 1e50;

        // Locations
        vector<vector <float>> locations;

        // Constructor
        MaxDiversityProblemLowMemory();
        MaxDiversityProblemLowMemory(string file_name, int ratio = 0, int coords = 2){
    
            // Get file
            ifstream input(file_name);
            string line;


            // get first line
            getline(input,line);
            istringstream stream(line);

            // get n
            if (ratio == 0){
                stream >> n >> p;
            }
            else
            {
                stream >> n;
                p = (int)ceil(n * 0.01 * ratio);
            }
            this->coords = coords;

            // Resize location matrix
            locations.resize(n);
            for (int i = 0; i < n; i++)
            {
                locations[i].resize(coords);
            }
            
            // Fill location matrix
            int n1;
            double d[coords];
            for (int i = 0; i < n; i++)
            {
                getline(input,line);
                istringstream stream(line);
                stream >> n1;
                for (int s = 0; s < coords; s++)
                {
                    stream >> locations[n1-1][s];
                }
            }
            input.close();

        }        
};

#endif

