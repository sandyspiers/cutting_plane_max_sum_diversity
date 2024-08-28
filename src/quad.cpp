#include <iostream>
#include <bits/stdc++.h>
#include <vector>
#include <ilcplex/ilocplex.h>
#include "max_diversity.hpp"
using namespace std;

struct Result{
    int timelimit;
    double obj_val;
    double gap;
    double solve_time;
};

ILOMIPINFOCALLBACK4(TimeLimits, IloNum*, start_time, int*, next_time, vector<int>*, timelimit, vector<Result>*, results){
    if (getCplexTime() - *start_time > (*timelimit)[*next_time]){
        // We have hit the next time limit
        Result res;
        res.timelimit = (*timelimit)[*next_time];
        res.obj_val = getIncumbentObjValue();
        res.gap = getMIPRelativeGap();
        res.solve_time = getCplexTime() - *start_time;
        cout << "result... "    << res.timelimit << " " 
                                << res.obj_val << " "
                                << res.gap << " "
                                << res.solve_time << endl; 
        (*results).push_back(res);
        *next_time += 1;
    }
    return;
}

class QuadraticSolver{

    public:

        // Report
        IloNum obj_val;
        IloNum solve_time;
        IloNum gap;

        // Problem
        MaxDiversityProblem *mdp;

        // Settings
        vector<int>* timelimit = 0;

        // Constructor
        QuadraticSolver(MaxDiversityProblem & problem, vector<int> & time_limits){
            mdp = &problem;
            timelimit = &time_limits;
        }

        void solve(vector<Result> &results) {

            IloEnv env;

            try {
                
                IloModel model(env);

                IloBoolVarArray x(env, mdp->n);
                IloNumVarArray w(env, mdp->n - 1, 0.0, IloInfinity );

                // Choose m
                model.add( IloSum(x) == mdp->p );

                // Objective
                IloExpr obj(env);
                for (int i = 0; i < mdp->n - 1; i++)
                {
                    for (int j = i+1; j < mdp->n; j++)
                    {
                        obj += mdp->distance[i][j] * x[i] * x[j];
                    }
                }
                
                // Objective
                model.add(IloMaximize(env, obj));

                // Solver
                int next_time = 0;
                IloNum time1, time2;
                IloCplex cplex(model);

                // Parameters
                cplex.setParam(IloCplex::Param::Threads, 1); 
                cplex.setParam(IloCplex::Param::ClockType, 1);
                cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, 0);
                if (timelimit->back() > 0){
                    cplex.setParam(IloCplex::Param::TimeLimit, timelimit->back());
                    if (timelimit->size() > 1){
                        cplex.use(TimeLimits(env,&time1,&next_time,timelimit,&results));
                    }
                }
                //cplex.setParam(IloCplex::Param::MIP::Strategy::HeuristicFreq, -1);
                //cplex.setParam(IloCplex::Param::MIP::Cuts::MIRCut, -1);
                //cplex.setParam(IloCplex::Param::MIP::Cuts::Implied, -1);
                //cplex.setParam(IloCplex::Param::MIP::Cuts::Gomory, -1);
                //cplex.setParam(IloCplex::Param::MIP::Cuts::FlowCovers, -1);
                //cplex.setParam(IloCplex::Param::MIP::Cuts::PathCut, -1);
                //cplex.setParam(IloCplex::Param::MIP::Cuts::LiftProj, -1);
                //cplex.setParam(IloCplex::Param::MIP::Cuts::ZeroHalfCut, -1);
                //cplex.setParam(IloCplex::Param::MIP::Cuts::Cliques, -1);
                //cplex.setParam(IloCplex::Param::MIP::Cuts::Covers, -1);  

                // Solve
                time1 = cplex.getCplexTime();
                cplex.solve();
                time2 = cplex.getCplexTime();

                // Report for the remaining time limits
                for (int i = next_time; i < timelimit->size(); i++)
                {
                    Result res;
                    res.timelimit = (*timelimit)[i];
                    res.obj_val = cplex.getObjValue();
                    res.gap = cplex.getMIPRelativeGap();
                    res.solve_time = time2-time1;
                    results.push_back(res);
                }
            }
            catch (IloException& ex) {
                cerr << "Error: " << ex << endl;
            }
            catch (...) {
                cerr << "Error" << endl;
            }
            env.end();

        }
    
};

int main(int argc,char* argv[]){
    // Executable options:
    // quad.exe (0)1 
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

    vector<int> timelimit;
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
    string output_filepath = "./results/" + output_file;

    // Setup...
    MaxDiversityProblem mdp(input_filepath, p_ratio);
    QuadraticSolver quad(mdp, timelimit);

    // Solve...
    vector<Result> results;
    quad.solve(results);

    // Log...
    ofstream output(output_filepath, ios::app);
	output.precision(17);
    for (int i = 0; i < results.size(); i++)
    {
        output  << instance_set  << " "
                << instance_file << " "
                << mdp.n << " "
                << mdp.p << " " 
                << results[i].timelimit << " "
                << "quad" << " "
                << results[i].obj_val << " "
                << results[i].gap << " "
                << results[i].solve_time << " "
                << "na" << endl;
    }
    output.close();

    
    return 0;
}
