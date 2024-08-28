//============================================================================
// Name        : OBLMAMDP.cpp
// Author      : Yangming Zhou
// Version     : edit 02 December 2016
// Copyright   : zhou.yangming@yahoo.com
// Description : Yangming Zhou et al., Opposition-based memetic search for maximum diversity problem,
//               IEEE Transaction on Evolutionary Computation, 21(5):731-745, 2017.
//               https://ieeexplore.ieee.org/abstract/document/7864317
//============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <bits/stdc++.h> // for vector sorting
#include <vector>
#include <math.h>
#include <unistd.h>
#include <time.h>

#define abs(x)(((x) < 0) ? -(x):(x))	// calculate the absolute value of x
#define max_integer 2147483647
#define min_integer -2147483648
#define epsilon 0.000001
#define NP 2				// number of parents
#define PS 10				// population size
#define alpha 15			// tabu tenure factor
#define T 1500				// period for determining the tabu tenure
#define max_iter 50000		// number of iterations in TS
#define scale_factor 2.0    // neighborhood size should not less than 0.5
#define is_obl true

using namespace std;

// Results tracker
struct Result{
    int timelimit;
    double obj_val;
    double solve_time;
};
vector<Result> results;

// running time
double start_time,limit_time;
vector<int> timelimit;
int next_time = 0;

char *dataset;
char *instance_name; 		// instance file name
char instance_path[50];		// instance file path
char result_path[50]; 		// result file path
char statistic_path[50];	// statistical file path
char final_result_file[50];

int allocate_space;			// allocate space for initializing
int total_nbr_v;			// total number of elements
int nbr_v;					// number of selected elements
int delta_nbr_v;			// number of remaining elements

double **diversity;   		// diversity matrix
double max_diversity;		// maximum distance between any two items

int *improved_sol,*best_sol;
double improved_cost,best_cost;			// global best cost
double improved_time,best_time;			// time to find the global best solution

int **pop;					// population with PS individuals
double **sol_distance;			// distance between any two solutions in the population
double *pop_distance;			// the distance from the individual to population
double *pop_cost;			// f value of each individual in the population
double *pop_score;			// score of each solution in the population

int *offspring;				// offspring by crossover operator
int *opposite_sol;			// opposite offspring
int *vertex;				// indicate each item
int nbr_gen;

/*********************basic functions******************/
void calculate_rank(int index_type,int nbr,double *a,int *b)
{
	double *c;
	c = new double[nbr];
	for(int i = 0; i < nbr; i++)
		c[i] = a[i];

	int *flag;
	flag = new int[nbr];
	for(int i = 0; i < nbr; i++)
		flag[i] = 0;

	double temp;
    for(int i = 1; i < nbr; i++)
    	for(int j = nbr-1; j >= i; j--)
        {
    		if(index_type == 0)
    		{
    			// lower score, lower rank
    			if(c[j-1] > c[j])
    			{
    				temp = c[j-1];
    				c[j-1] = c[j];
    				c[j] = temp;
    			}
    		}
    		else if(index_type == 1)
    		{
    			// higher score, lower rank
    			if(c[j-1] < c[j])
    			{
    				temp = c[j-1];
    				c[j-1] = c[j];
    				c[j] = temp;
    			}
    		}
        }

    for(int i = 0; i < nbr; i++)
    	for(int j = 0; j < nbr; j++)
    	{
    		if(flag[j] == 0 && a[i] == c[j])
    		{
    			b[i] = j + 1;
    			flag[j] = 1;
    			break;
    		}
    	}
    delete [] c;
    delete [] flag;
}

/******************read instances******************/
void read_instance(int p_ratio)
{
	ifstream FIC;
    FIC.open(instance_path);
    if(FIC.fail())
    {
    	cout << "### Fail to open file: " << instance_name << endl;
        getchar();
        exit(0);
    }
    if(FIC.eof())
    {
    	cout << "### Fail to open file: " << instance_name << endl;
        exit(0);
    }
    int nbr_pairs = -1;
    int x1,x2;
    double x3;

	/* Files are now read **only** for n, whereas p comes from p ratio given as an argument */
	FIC >> total_nbr_v;
	nbr_pairs = total_nbr_v*(total_nbr_v - 1);
	nbr_pairs = nbr_pairs/2;
	nbr_v = (int)ceil(total_nbr_v * 0.01 * p_ratio);
	while (nbr_v * 2 > total_nbr_v){
		total_nbr_v += 1;
	}
 	delta_nbr_v = total_nbr_v - nbr_v; 
	// Move input to the end of the line
	char next;
	while(FIC.get(next))
	{
		if (next == '\n')  // If the file has been opened in
		{    break;        // text mode then it will correctly decode the
		}                  // platform specific EOL marker into '\n'
	}

    /* cout << "Dataset: "<< dataset << endl;
    if(strcmp(dataset,"MDG-a") == 0 || strcmp(dataset,"MDG-b") == 0 || strcmp(dataset,"MDG-c") == 0) // for SetIMDG-a,MDG-b and MDG-c
    {
    	FIC >> total_nbr_v >> nbr_v;
    	nbr_pairs =(total_nbr_v*(total_nbr_v - 1))/2;
    }
    else if(strcmp(dataset,"b2500") == 0) // for SetIIb2500
    {
    	FIC >> total_nbr_v >> nbr_pairs;
    	nbr_v = 1000;
    }
    else if(strcmp(dataset,"p30005000") == 0)// for SetIIIP3000&P5000
    {
    	FIC >> total_nbr_v >> nbr_pairs;
    	nbr_v = total_nbr_v/2;
    }
    else
    {
    	cout << "data set is wrong!" << endl;
    	exit(0);
    }
 	delta_nbr_v = total_nbr_v - nbr_v; */

    diversity = new double*[total_nbr_v];
    for(int x = 0; x < total_nbr_v; x++)
        diversity[x] = new double[total_nbr_v];
    for(int x = 0; x < total_nbr_v; x++)
    	for(int y = 0; y < total_nbr_v; y++)
    		diversity[x][y] = 0;

    double density = (double)nbr_pairs/(double)(total_nbr_v*(total_nbr_v-1)/2);
    double percent = (double)nbr_v/(double)(total_nbr_v);
    cout << "The statistics of the instance " << instance_path << endl;
    cout << "n = " << total_nbr_v << ", m = " << nbr_v << ", m/n = " << percent << ", and density = " << density << endl;

    max_diversity = 0.0;
    for(int i = 0; i < nbr_pairs; i++)
    {
        FIC >> x1 >> x2 >> x3;
        // if(strcmp(dataset,"b2500") == 0 || strcmp(dataset,"p30005000") == 0)// code for SetIIb2500 and SetIIIP3000&P5000
        // {
        // 	x1--;
        // 	x2--;
        // }
        if ( x1 < 0 || x2 < 0 || x1 >= total_nbr_v || x2 >= total_nbr_v )
        {
            cout << "### Read Date Error : line = "<< x1 << ", column = " << x2 << endl;
            exit(0);
        }

        if(x1 != x2)
        {
        	// if(strcmp(dataset,"b2500") == 0) // code for SetIIb2500
        	// 	x3 = x3*2;

        	diversity[x1][x2] = diversity[x2][x1] = x3;
        	if(diversity[x1][x2] > max_diversity)
        		max_diversity = diversity[x1][x2];
        }
    }
    cout << "Finish loading data!" << endl;
}

void setup_data()
{
    allocate_space = total_nbr_v*sizeof(int);
 	improved_sol = new int[nbr_v];
 	best_sol = new int[nbr_v];
    offspring = new int[nbr_v];
    opposite_sol = new int[nbr_v];
    pop_distance = new double[PS+1];
    pop_cost = new double[PS+1];
    pop_score = new double[PS+1];
    vertex = new int[total_nbr_v];

    pop = new int*[PS+1];
    for(int x = 0; x < PS+1; x++)
    	pop[x] = new int[nbr_v];

    sol_distance = new double*[PS+1];
    for(int x = 0; x < PS+1; x++)
    	sol_distance[x] = new double[PS+1];
}

void clear_data()
{
    delete[] improved_sol;
    delete[] best_sol;
    delete[] pop_cost;
    delete[] pop_distance;
    delete[] pop_score;
    delete[] offspring;
    delete[] opposite_sol;
    delete[] vertex;
    for(int i = 0; i < PS+1; i++)
    {
    	delete pop[i];
    	delete sol_distance[i];
    }
}

// check the solution is correct or not
void check_and_store_result(int *sol,double sol_cost)
{
	FILE *out;
	// check the range of chosen elements in the solution
	for(int i = 0; i < nbr_v; i++)
		if( sol[i] < 0 || sol[i] >= total_nbr_v)
        {
			printf("### element:%d is out of range: %d-%d",sol[i],0,total_nbr_v-1);
            exit(0);
        }

	// check the cost;
	double true_sol_cost = 0.0;
	for(int i = 0; i < nbr_v; i++)
		for(int j = i + 1; j < nbr_v; j++)
			true_sol_cost += diversity[sol[i]][sol[j]];

	if(abs(true_sol_cost - sol_cost) > epsilon)
	{
		printf("Find a error solution: its sol_cost = %f, while its true_sol_ost =%f\n",sol_cost,true_sol_cost);
		exit(0);
	}

	// store the computational result
	out = fopen(result_path, "a+");
	fprintf(out,"   Parameters:                                 \n");
	fprintf(out,"   Population size                        		= %d\n",PS);
	fprintf(out,"   Total number of elements                    = %d\n",total_nbr_v);
	fprintf(out,"   Number of selected elements                	= %d\n",nbr_v);
	fprintf(out,"   Maximum pairwise diversity                  = %lf\n",max_diversity);
	fprintf(out,"   Limit time(second)                          = %lf\n",limit_time);
	fprintf(out,"   Best solution value		                   	= %lf\n",best_cost);
	fprintf(out,"   Time to find the best solution            	= %lf\n",best_time);
	fprintf(out,"   Found best solution:\n");
	fprintf(out,"----------------------------------------------------------------------\n");
	for(int i = 0; i < nbr_v; i++)
		fprintf(out, "%d ",best_sol[i]);
	fprintf(out,"\n");

	fclose(out);
}

void update_improved_sol(int *sol,double sol_cost)
{
    improved_time = (clock() - start_time)/CLOCKS_PER_SEC;
    improved_cost = sol_cost;
    for(int i = 0; i < nbr_v; i++)
    	improved_sol[i] = sol[i];
}

void update_best_sol()
{
	best_time = improved_time;
	best_cost = improved_cost;
	for(int i = 0; i < nbr_v; i++)
		best_sol[i] = improved_sol[i];
}

int determine_tabu_tenure(int iter)
{
	int delta_tenure;
    int temp = iter%T;

	if(temp > 700 && temp <= 800)
		delta_tenure = 8*alpha;
	else if((temp > 300 && temp <= 400)||(temp > 1100 && temp <= 1200))
		delta_tenure = 4*alpha;
	else if((temp > 100 && temp <= 200)||(temp > 500 && temp <= 600)||(temp > 900 && temp <= 1000)||(temp > 1300 && temp <= 1400))
		delta_tenure = 2*alpha;
	else
		delta_tenure = alpha;

	return delta_tenure;
}

bool check_time(){
	//cout << (clock() - start_time)/CLOCKS_PER_SEC << " " << timelimit[next_time] << endl;
	if( (clock() - start_time)/CLOCKS_PER_SEC >= timelimit[next_time]){
		if (next_time < timelimit.size()){
			Result res;
			res.obj_val = best_cost;
			res.solve_time = (clock() - start_time)/CLOCKS_PER_SEC;
			res.timelimit = timelimit[next_time];
			results.push_back(res);
			next_time += 1;
		}
		if (next_time >= timelimit.size()){
			return true;
		}
	}
	return false;
}


void tabu_search(int *sol)
{
	double used_time;
	int iter = 0;
	int no_improve_iter = 0;
	int *outgoing_X;			// set of items with small gain in S (X)
	int *ingoing_Y;				// set of items with large gain out of S (Y)
	int size_X;					// size of the X
	int size_Y;					// size of the Y
    double min_gain_in_S;		// dMinInS
    double max_gain_out_S; 		// dMaxOutS

    int num_best;				// number of best non-tabu moves
    int num_tabu_best;			// number of best tabu moves
    double delta;				// move gain
    double best_delta;			// best move gain for non tabu move
    double best_tabu_delta;		// best move gain for tabu move

	int to_remove,to_add;
	int ad0,ad1;
	int *sol1;
	int *unchosen;
	double *gain;
	int *address;
	int *tenure;
	int *vertex;
	int x1,x2;
	double sol_cost = 0.0;
	int index;
	int max_nbr_moves = 400;
    int best_swap_X[max_nbr_moves];				// set of best and non-tabu items in X
    int best_swap_Y[max_nbr_moves];				// set of best and non-tabu items in Y
    int best_tabu_swap_X[max_nbr_moves];		// set of best and tabu items in X
    int best_tabu_swap_Y[max_nbr_moves]; 		// set of best and tabu items in Y

	outgoing_X = new int[nbr_v];
	ingoing_Y = new int[delta_nbr_v];
	sol1 = new int[nbr_v];
	unchosen = new int[delta_nbr_v];
	gain = new double[total_nbr_v];
	address = new int[total_nbr_v];
	tenure = new int[total_nbr_v];
	vertex = new int[total_nbr_v];

	for(int i = 0; i < total_nbr_v; i++)
	{
		tenure[i] = 0;
		vertex[i] = 0;
		address[i] = 0;
	}

	for(int i = 0; i < nbr_v; i++)
		vertex[sol[i]]++;

	x1 = x2 = 0;
	for(int i = 0; i < total_nbr_v; i++)
	{
		if(vertex[i] == 1)
		{
			sol1[x1] = i;
			address[i] = x1;
			x1++;
		}
		else if(vertex[i] == 0)
		{
			unchosen[x2] = i;
			address[i] = x2;
			x2++;
		}
		else
		{
			cout << "input solution is not correct: " << "vertex[" << i << "] =" << vertex[i] << endl;
			exit(0);
		}
	}

    // calculate the potential contribution of each item to the objective function
    for(int i = 0; i < total_nbr_v; i++)
    {
    	gain[i] = 0.0;
        for(int j = 0; j < nbr_v; j++)
        	gain[i] += diversity[i][sol1[j]];
    }

    // calculate the objective of current solution
    for(int i = 0; i < nbr_v; i++)
    	sol_cost += gain[sol1[i]];
    sol_cost = sol_cost/2.0;

    update_improved_sol(sol1,sol_cost);

	//cout << "Iter = " << iter << ", best cost = " << best_cost << ": f = " << cost << endl;
    while(iter < max_iter)
    {
    	// X: the set of items with small gain (i.e., no more than dMinInS + dmax) in S
    	min_gain_in_S = max_integer;
    	for(int i = 0; i < nbr_v; i++)
    	{
    		if(gain[sol1[i]] < min_gain_in_S)
    			min_gain_in_S = gain[sol1[i]];
    	}
    	min_gain_in_S += scale_factor*max_diversity;

    	size_X = 0;
    	for(int i = 0; i < nbr_v; i++)
    		if(gain[sol1[i]] <= min_gain_in_S)
    		{
    			outgoing_X[size_X] = sol1[i];
    			size_X++;
    		}

    	// Y: the set of items with large gain (i.e., no less than dMaxOutS - dmax) out of S
    	max_gain_out_S = -max_integer;
    	for(int i = 0; i < delta_nbr_v; i++)
    	{
    		if(gain[unchosen[i]] > max_gain_out_S)
    			max_gain_out_S = gain[unchosen[i]];
    	}
    	max_gain_out_S -= scale_factor*max_diversity;

    	size_Y = 0;
    	for(int i = 0; i < delta_nbr_v; i++)
    		if(gain[unchosen[i]] >= max_gain_out_S)
    		{
    			ingoing_Y[size_Y] = unchosen[i];
    			size_Y++;
    		}

    	/* Find out all best tabu or non-tabu moves */
        best_delta = -max_integer;
        best_tabu_delta = -max_integer;
        num_best = 0;
        num_tabu_best = 0;

        for(int i = 0; i < size_X; i++)
        {
        	to_remove = outgoing_X[i];
        	for(int j = 0; j < size_Y; j++)
        	{
        		to_add = ingoing_Y[j];
        	    delta = gain[to_add] - gain[to_remove] - diversity[to_remove][to_add];
        	    if((tenure[to_remove] <= iter) && (tenure[to_add] <= iter))
        	    {
        	    	if(delta > best_delta)
        	    	{
        	    		best_delta = delta;
        	    	    best_swap_X[0] = to_remove;
        	    	    best_swap_Y[0] = to_add;
        	    	    num_best = 1;
        	    	}
        	    	else if(delta == best_delta && (num_best < max_nbr_moves))
        	    	{
        	    		best_swap_X[num_best] = to_remove;
        	    	    best_swap_Y[num_best] = to_add;
        	    	    num_best++;
        	    	}
        	    }
        	    else // tabu move
        	    {
        	    	if(delta > best_tabu_delta)
        	    	{
        	    		best_tabu_delta = delta;
        	    	    best_tabu_swap_X[0] = to_remove;
        	    	    best_tabu_swap_Y[0] = to_add;
        	    	    num_tabu_best = 1;
        	    	}
        	    	else if(delta == best_tabu_delta && (num_tabu_best < max_nbr_moves))
        	    	{
        	    	    best_tabu_swap_X[num_tabu_best] = to_remove;
        	    	    best_tabu_swap_Y[num_tabu_best] = to_add;
        	    	    num_tabu_best++;
        	    	}
        	    }
        	}
        }

        /* Accept a best tabu or non-tabu move */
	    if(((num_tabu_best > 0) && (best_tabu_delta > best_delta) && (best_tabu_delta + sol_cost > improved_cost)) || num_best == 0)
	    {
	    	index = rand()%num_tabu_best;
	        to_remove = best_tabu_swap_X[index];
	        to_add = best_tabu_swap_Y[index];
	        sol_cost += best_tabu_delta;
	    }
	    else
	    {
	    	index = rand()%num_best;
	        to_remove = best_swap_X[index];
	        to_add = best_swap_Y[index];
	        sol_cost += best_delta;
	    }

	    /* Make a move and update */
        ad0 = address[to_remove];
        ad1 = address[to_add];
        sol1[ad0] = to_add;
        address[to_add] = ad0;
        unchosen[ad1] = to_remove;
        address[to_remove] = ad1;

	    for(int i = 0; i < total_nbr_v; i++)
	    	gain[i] += diversity[i][to_add] - diversity[i][to_remove];

	    tenure[to_remove] = iter + determine_tabu_tenure(iter);
	    tenure[to_add] = iter + round(0.7*determine_tabu_tenure(iter));

	    /* Keep the best solution found so far */
	    if(sol_cost > improved_cost)
	    {
	    	update_improved_sol(sol1,sol_cost);
	    	no_improve_iter = 0;
	    }
	    else
	    	no_improve_iter++;

	    iter++;
	    
		//used_time = (clock() - start_time)/CLOCKS_PER_SEC;
	    if(check_time())
	    	break;
    }
    delete []outgoing_X;
    delete []ingoing_Y;
    delete []tenure;
    delete []gain;
    delete []address;
    delete []vertex;
    delete []sol1;
    delete []unchosen;
}

// Check the solution is duplicate or not in the population
int is_duplicate_sol(int **pop1,int index)
{
    int duplicate = 0;
    for(int i = 0; i < index; i++)
    {
    	duplicate = 1;
    	for(int j = 0; j < nbr_v; j++)
    		if(improved_sol[j] != pop1[i][j])
    		{
    			duplicate = 0;
    			break;
    		}

    		if(duplicate == 1)
    			break;
    }
    return duplicate;
}

// Compute the distance between any two solutions in the population
double calculate_sol_distance(int x1,int x2)
{
	double distance;
	int u = 0;
	int v = 0;
    int sharing = 0;
    while((u < nbr_v) && (v < nbr_v))
    {
    	if(pop[x1][u] == pop[x2][v])
    	{
    		sharing ++;
    		u++;
    		v++;
    	}
    	else if(pop[x1][u] < pop[x2][v])
    		u++;
        else if(pop[x1][u] > pop[x2][v])
        	v++;
    }
    distance = 1-(double)sharing/(double)nbr_v;
    return distance;
}

// Sort the elements in the solution in an ascend order
void ascend_sort(int *sol)
{
     int count = 0;
     memset(vertex,0,allocate_space);

     for(int i = 0; i < nbr_v; i++)
    	 vertex[sol[i]] = 1;

     for(int i = 0; i < total_nbr_v; i++)
    	 if(vertex[i] == 1)
    		 sol[count++] = i;
}

// Initialize the population with opposition-based learning
void build_pool_with_OBL()
{
    int nbr_item;
    int nbr_sol;
    int index;
    int *sol;
    int *opp_sol;
    int *elite_sol;
    double elite_sol_cost;

    sol = new int[nbr_v];
    opp_sol = new int[nbr_v];
    elite_sol = new int[nbr_v];

    best_cost = -1;
    nbr_sol = 0;
    while(nbr_sol < PS)
    {
    	// generate a solution
    	memset(vertex,0,allocate_space);
    	nbr_item = 0;
    	while(nbr_item < nbr_v)
    	{
    		index = rand()%total_nbr_v;
    		if(vertex[index] == 0)
    		{
    			sol[nbr_item] = index;
    			nbr_item++;
    			vertex[index] = 1;
    		}
    	}

        tabu_search(sol);
        ascend_sort(improved_sol);

        elite_sol_cost = improved_cost;
        for(int i = 0; i < nbr_v; i++)
        	elite_sol[i] = improved_sol[i];

    	// generate its opposite solution
        if(delta_nbr_v != nbr_v)
        {
        	int *available_items;
            int nbr_available_item = 0;
            available_items = new int[delta_nbr_v];

            for(int i = 0; i < total_nbr_v; i++)
            	if(vertex[i] == 0)
            	{
            		available_items[nbr_available_item] = i;
            		nbr_available_item++;
            	}
            nbr_item = 0;
            while(nbr_item < nbr_v)
            {
            	index = rand()%nbr_available_item;
            	opp_sol[nbr_item] = available_items[index];
            	nbr_item++;

            	// delete this vertex from the array
            	nbr_available_item--;
            	available_items[index] = available_items[nbr_available_item];
            }
            delete [] available_items;
        }
        else
        {
        	nbr_item = 0;
        	for(int i = 0; i < total_nbr_v; i++)
        		if(vertex[i] == 0)
        		{
        			opp_sol[nbr_item] = i;
        			nbr_item++;
        		}
        }

        tabu_search(opp_sol);
        ascend_sort(improved_sol);

        if(improved_cost < elite_sol_cost || (abs(improved_cost-elite_sol_cost) < epsilon && rand()%2 == 0))
        {
        	improved_cost = elite_sol_cost;
        	for(int i = 0; i < nbr_v; i++)
        		improved_sol[i] = elite_sol[i];
        }

        memset(vertex,0,allocate_space);
        for(int i = 0; i < nbr_v; i++)
        	vertex[improved_sol[i]] = 1;

        // modify it if it is same to an existing solution
        int swapin_v,swapout_v;
        int flag;
        double swapout_gain,swapin_gain;
        while(is_duplicate_sol(pop,nbr_sol) == 1)
        {
        	index = rand()%nbr_v;
        	swapout_v = improved_sol[index];

        	swapout_gain = 0.0;
        	for(int i = 0; i < nbr_v; i++)
        		if(improved_sol[i] != swapout_v)
        			swapout_gain += diversity[improved_sol[i]][swapout_v];

        	flag = 0;
        	while(flag == 0)
        	{
        		swapin_v = rand()%total_nbr_v;
        		if(vertex[swapin_v] == 0)
        			flag = 1;
        	}

        	swapin_gain = 0.0;
        	for(int i = 0; i < nbr_v; i++)
        		if(improved_sol[i] != swapout_v)
        			swapin_gain += diversity[improved_sol[i]][swapin_v];

        	// swap
        	vertex[swapin_v] = 1;
        	vertex[swapout_v] = 0;
        	improved_sol[index] = swapin_v;
        	improved_cost += swapin_gain - swapout_gain;
        	ascend_sort(improved_sol);
        }

    	pop_cost[nbr_sol] = improved_cost;
    	for(int i = 0; i < nbr_v; i++)
    		pop[nbr_sol][i] = improved_sol[i];

    	nbr_sol++;

        if(improved_cost > best_cost)
        	update_best_sol();
   }

    // Calculate the distance between any two solutions in the population
    for(int i = 0; i < PS; i++)
    {
    	for(int j = i + 1; j < PS; j++)
    	{
    		sol_distance[i][j] = calculate_sol_distance(i,j);
    		sol_distance[j][i] = sol_distance[i][j];
    	}
    	sol_distance[i][i] = 0.0;
    }

    delete [] sol;
    delete [] opp_sol;
    delete [] elite_sol;
}

// Create an offspring and its opposition by crossover and assign the remaining items greedily
void crossover_with_greedy()
{
    int choose_p;
    int index_p;
    int nbr_p;
    int *is_choose_p;
    int *p;

    int choose_v;
    int index_remaining_v;
    int nbr_added_v;
    int *index_best_v;
    int nbr_best_v;
    int *nbr_remaining_v;
    int **remaining_v;
    double max_v_profit;
    double v_profit;

    is_choose_p = new int[PS];
    p = new int[NP];

    index_best_v = new int[delta_nbr_v];

    nbr_remaining_v = new int[NP];
    remaining_v = new int*[NP];
    for(int i = 0; i < NP; i++)
    	remaining_v[i] = new int[nbr_v];

    // choose two parents
    for(int i = 0; i < PS; i++)
    	is_choose_p[i] = 0;
    nbr_p = 0;
    while(nbr_p < NP)
    {
    	choose_p = rand()%PS;
        if(is_choose_p[choose_p] == 0)
        {
        	p[nbr_p] = choose_p;
        	nbr_p++;
        	is_choose_p[choose_p] = 1;
        }
    }

    // Build a partial solution S0 by preserving the common elements
    memset(vertex,0,allocate_space);
    for(int i = 0; i < NP; i++)
    	for(int j = 0; j < nbr_v; j++)
           vertex[pop[p[i]][j]]++;

    // S1/S0 and S2/S0
    int v;
    for(int i = 0; i < NP; i++)
    {
    	nbr_remaining_v[i] = 0;
    	for(int j = 0; j < nbr_v; j++)
    	{
    		v = pop[p[i]][j];
    		if(vertex[v] == 1)
    		{
    			remaining_v[i][nbr_remaining_v[i]] = v;
    			nbr_remaining_v[i]++;
    			vertex[v] = 0;
    		}
    	}
    }

    // S0
    nbr_added_v = 0;
    for(int i = 0; i < total_nbr_v; i++)
    	if(vertex[i] == NP)
    	{
    		offspring[nbr_added_v] = i;
    		vertex[i] = 1;
    		nbr_added_v++;
    	}

    // generate an offspring by completing the partial solution in a greedy way
    while(nbr_added_v < nbr_v)
    {
    	index_p = nbr_added_v%NP;
    	max_v_profit = min_integer;
    	for(int i = 0; i < nbr_remaining_v[index_p]; i++)
    	{
    		v_profit = 0.0;
    		for(int j = 0; j < nbr_added_v; j++)
    			v_profit += diversity[remaining_v[index_p][i]][offspring[j]];

    		if(v_profit > max_v_profit)
    		{
    			max_v_profit = v_profit;
    			index_best_v[0] = i;
    			nbr_best_v = 1;
    		}
    		else if(abs(v_profit-max_v_profit) < epsilon)
    		{
    			index_best_v[nbr_best_v] = i;
    			nbr_best_v++;
    		}
    	}
    	index_remaining_v = index_best_v[rand()%nbr_best_v];
    	choose_v = remaining_v[index_p][index_remaining_v];

    	offspring[nbr_added_v] = choose_v;
    	nbr_added_v++;
    	vertex[choose_v] = 1;
    	nbr_remaining_v[index_p]--;
    	remaining_v[index_p][index_remaining_v] = remaining_v[index_p][nbr_remaining_v[index_p]];
    }

    // generate an opposite solution
    if(is_obl == true)
    {
    	if(delta_nbr_v > nbr_v)
    	{
    		int index_avaiable_v;
    		int nbr_available_v;
    		int *available_v;
    		available_v = new int[delta_nbr_v];

    		nbr_available_v = 0;
    		for(int i = 0; i < total_nbr_v; i++)
    			if(vertex[i] == 0)
    			{
    				available_v[nbr_available_v] = i;
    				nbr_available_v++;
    			}

    		nbr_added_v = 0;
    		while(nbr_added_v < nbr_v)
    		{
    			index_avaiable_v = rand()%nbr_available_v;
    			opposite_sol[nbr_added_v] = available_v[index_avaiable_v];
    			nbr_added_v++;

    			// delete this vertex from the array
    			nbr_available_v--;
    			available_v[index_avaiable_v] = available_v[nbr_available_v];
    		}
    		delete [] available_v;
    	}
    	else if(delta_nbr_v == nbr_v)
    	{
    		nbr_added_v = 0;
    		for(int i = 0; i < total_nbr_v; i++)
    			if(vertex[i] == 0)
    			{
    				opposite_sol[nbr_added_v] = i;
    				nbr_added_v++;
    			}
    	}
    	else
    	{
    		printf("error occurs in crossover: delta_nbr_v < nbr_v\n");
    		exit(-1);
    	}
    }

    delete []is_choose_p;
    delete []p;
    delete []index_best_v;
    delete []nbr_remaining_v;
    for(int i = 0; i < NP; i++)
    	delete remaining_v[i];
}

void rank_based_pool_updating()
{
    double avg_sol_distance;
    double min_score;
    int index_worst;

    // Insert the offspring into the population
    pop_cost[PS] = improved_cost;
    for(int i = 0; i < nbr_v; i++)
    	pop[PS][i] = improved_sol[i];

    for(int i = 0; i < PS; i++)
    {
    	sol_distance[i][PS] = calculate_sol_distance(i,PS);
    	sol_distance[PS][i] = sol_distance[i][PS];
    }
    sol_distance[PS][PS] = 0.0;

    // Calculate the average distance of each individual with the whole population
    for(int i = 0; i < PS+1; i++)
    {
    	avg_sol_distance = 0.0;
    	for(int j = 0; j < PS+1; j++)
    	{
    		if(j != i)
    			avg_sol_distance += sol_distance[i][j];
    	}
    	pop_distance[i] = avg_sol_distance/PS;
    }


    // Compute the score of each individual in the population
    // Calculate the rank of cost and distance respectively
    int *cost_rank,*distance_rank;
    cost_rank = new int[PS+1];
    distance_rank = new int[PS+1];

    for(int i = 0; i < PS+1; i++)
    {
    	cost_rank[i] = i+1;
    	distance_rank[i] = i+1;
    }

    calculate_rank(0,PS+1,pop_cost,cost_rank);
    calculate_rank(0,PS+1,pop_distance,distance_rank);

    // Compute the score of each individual in the population
    for(int i = 0; i < PS+1; i++)
    	pop_score[i] = alpha*cost_rank[i] + (1.0-alpha)*distance_rank[i];

    min_score = double(max_integer);
    for(int i = 0; i < PS+1; i++)
    	if(pop_score[i] < min_score)
    	{
    		min_score = pop_score[i];
    		index_worst = i;
    	}

    // Insert the offspring
    if(index_worst != PS && is_duplicate_sol(pop,PS) == 0)
    {
    	pop_cost[index_worst] = improved_cost;
        for(int i = 0; i < nbr_v; i++)
           pop[index_worst][i] = improved_sol[i];

        for(int i = 0; i < PS; i++)
        {
        	sol_distance[i][index_worst] = sol_distance[PS][i];
        	sol_distance[index_worst][i] = sol_distance[i][index_worst];
        }
        sol_distance[index_worst][index_worst] = 0.0;
     }
}

// Oppostion-based memetic algorithm (OBMA)
void OBMA()
{
	int no_improve_gen = 0;
	nbr_gen = 0;
    best_cost = -1.0;
    best_time = 0.0;

    // Population Initialization
    printf("using OBL!!!\n");
    build_pool_with_OBL();


    // Population Evolution
    //printf("best_cost = %lf, at gen = %d, no_improve_gen = %d, time to best = %.3f\n",best_cost,gen,no_improve_gen,best_time);
    while(1)
    {
    	//**** Create an offspring and its opposite solution by crossover operator ****
    	crossover_with_greedy();

    	//********************* improve offspring by tabu search **********************
        tabu_search(offspring);
        ascend_sort(improved_sol);

        // Record the best solution
        if(improved_cost > best_cost)
        {
        	update_best_sol();
        	no_improve_gen = 0;
        }
        else
        	no_improve_gen++;

        // Update the population
        rank_based_pool_updating();


        if(is_obl == true)
        {
        	//************ improve the opposite solution by tabu search *****************
        	tabu_search(opposite_sol);
        	ascend_sort(improved_sol);

        	// Record the best solution
        	if(improved_cost > best_cost)
        	{
        		update_best_sol();
        		no_improve_gen = 0;
        	}
        	else
        		no_improve_gen++;

        	// Update the population
        	rank_based_pool_updating();
        }

		// Check the cut off time
		if(check_time())
			break;
        nbr_gen++;
        // Display the intermediate results
        //printf("best_cost = %lf, at gen = %d, no_improve_gen = %d, time to best = %.3f\n",best_cost,gen,no_improve_gen,best_time);
    }
}

int main(int argc,char **argv)
{
	FILE *sum;
	int nbr_repeat = 1; // Only 1 run always
	int nbr_success = 0;
    double avg_best_cost = 0.0;
    double avg_best_time = 0.0;
    double max_best_cost = -1.0;
    double avg_min_best_time;
    double gap_best_cost;
	double sd;

    // Executable options:
    // ct.exe (0)1 
    // instance_set (1)2
    // instance_file (2)3
    // output_file (3)4
    // p_ratio (4)5
    // timelimit (5)6

    // Determine options:
    if (argc < 6){
        cout << "Incorrect arguments [instance_set instance_file output_file p_ratio timelimit(s)]" ;
        return 0;
    }

    // Get file arguments
    string instance_set  = argv[1];
    string instance_file = argv[2];
    string output_file   = argv[3];
    
    int p_ratio;
    istringstream ss4(argv[4]);
    ss4 >> p_ratio;

    for (int i = 5; i < argc; i++)
    {
        int _timelimit;
        istringstream ss5(argv[i]);
        ss5 >> _timelimit;
        timelimit.push_back(_timelimit);
    }
    sort(timelimit.begin(), timelimit.end());

    // Concat file paths
    string input_filepath  = "./data/" + instance_set + "/" + instance_file;
	strcpy(instance_path, input_filepath.c_str());
    string output_filepath = "./results/" + output_file;

    //Read the instance
    read_instance(p_ratio);
    setup_data();
    double best_sol_cost[nbr_repeat];

	// Run the algorithm
	srand((unsigned)time(NULL));
	start_time = clock();
	OBMA();
	printf("finish  nbr_gen = %d,best_cost = %f,best_time = %f\n",nbr_gen,best_cost,best_time);

    // Log...
    ofstream output(output_filepath, ios::app);
	output.precision(17);
    for (int i = 0; i < results.size(); i++)
    {
        output  << instance_set  << " "
                << instance_file << " "
                << total_nbr_v << " "
                << nbr_v << " " 
                << results[i].timelimit << " "
                << "obma" << " "
                << results[i].obj_val << " "
                <<"na" << " "
                << results[i].solve_time << " "
                << "na" << endl;
    }
    output.close();

	/* 
	// Repeat multiple run
	fprintf(sum,"Instance: %s, Limit Time = %.3lf\n",instance_name,limit_time);
	fprintf(sum,"---------------------------------------------------------\n");
	for(int i = 1; i <= nbr_repeat; i++)
    {
	    // strcpy(result_path,"./");
		// strcat(result_path,"results/");
	    // strcat(result_path,dataset);
	    // strcat(result_path,"/");
	    // strcat(result_path,instance_name);
	    // strcat(result_path,".res");
		// char str2[10];
		// sprintf(str2,"%d",i);
	    // strcat(result_path,str2);


		// Check and store the results
		check_and_store_result(best_sol,best_cost);
		fprintf(sum,"%11.3lf    %11.3lf    %d\n",best_cost,best_time,i);

		// Statistical results
		best_sol_cost[i-1] = best_cost;
		avg_best_cost += best_cost;
		avg_best_time += best_time;
		if(best_cost > max_best_cost)
		{
			max_best_cost = best_cost;
			avg_min_best_time = best_time;
			nbr_success = 1;
		}
		else if(abs(best_cost - max_best_cost) < 0.1)
		{
			nbr_success++;
			avg_min_best_time += best_time;
		}
    }

	// Compute the statistical results
	avg_best_cost = avg_best_cost/(double)nbr_repeat;
	avg_best_time = avg_best_time/(double)nbr_repeat;
	avg_min_best_time = avg_min_best_time/(double)nbr_success;
	sd = 0.0;
	for(int i = 0; i < nbr_repeat; i++)
		sd += (best_sol_cost[i] - avg_best_cost)*(best_sol_cost[i] - avg_best_cost);
	sd = sd/(double)nbr_repeat;
	sd = sqrt(sd);
	gap_best_cost = max_best_cost - avg_best_cost;

	fprintf(sum,"---------------------------------------------------------\n");
	fprintf(sum,"%.3lf, %.3lf\n",avg_best_cost,avg_best_time);
	fprintf(sum,"---------------------------------------------------------\n");
	fprintf(sum,"%.3lf, %.3lf, %.3lf, %.3lf, %d\n",max_best_cost,avg_min_best_time,gap_best_cost,sd,nbr_success);
	fclose(sum);

	cout << "finishing...!" << endl;
	cout << "found best objective = " << max_best_cost << endl;
	cout << "number of times to find the best solution = " << nbr_success << endl;
	cout << "average value of the best objective value = " << avg_best_cost << endl;
	cout << "average value of the best time (second) = " << avg_best_time << endl;

	FILE *fin;
    strcpy(final_result_file,dataset);
	strcat(final_result_file,"_result.txt");
	fin = fopen(final_result_file,"at+");
	if(fin != NULL)
	{
		fprintf(fin,"%s, %f, %f, %f, %f, %f, %d\n",instance_name,max_best_cost,avg_best_cost,avg_best_time,avg_min_best_time,sd,nbr_success);
	}
	fclose(fin);

	clear_data(); 
	*/

    return 0;
}
