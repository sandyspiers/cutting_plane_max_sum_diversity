#include <iostream>
#include <bits/stdc++.h>
#include <vector>
#include <ilcplex/ilocplex.h>
#include "max_diversity.hpp"
using namespace std;

struct Result{
    int timelimit;
    int num_cuts;
    double obj_val;
    double gap;
    double solve_time;
};

ILOMIPINFOCALLBACK5(TimeLimits, IloNum*, start_time, int*, next_time, vector<int>*, timelimit, vector<Result>*, results, IloInt*, num_cuts){
    if (getCplexTime() - *start_time > (*timelimit)[*next_time]){
        // We have hit the next time limit
        Result res;
        res.timelimit = (*timelimit)[*next_time];
        res.obj_val = getIncumbentObjValue();
        res.gap = getMIPRelativeGap();
        res.solve_time = getCplexTime() - *start_time;
        res.num_cuts = *num_cuts;
        cout << "result... "    << res.timelimit << " " 
                                << res.obj_val << " "
                                << res.gap << " "
                                << res.solve_time << endl; 
        (*results).push_back(res);
        *next_time += 1;
    }
    return;
}

ILOLAZYCONSTRAINTCALLBACK5(CutCallback, IloBoolVarArray, x, IloNumVar, theta, MaxDiversityProblem*, mdp, IloNum*, rho, IloInt*, num_cuts){
    
    // Get environment
    IloEnv env = getEnv();

    // Get x solution
    IloNumArray xSol(env,mdp->n);
    getValues(xSol,x);

    // Generate and add cut
    // cut: f(xSol) + <df(xSol),x - xSol>
    IloNum fx = 0.0;
    IloNum dx;
    IloExpr cut(env);
    for (int i=0; i< mdp->n; i++){
        dx = 0.0;
        for (int j = 0; j < i; j++)
        {
            dx += mdp->distance[i][j]*xSol[j];
        }
        dx += (*rho) * (0.5-xSol[i]);
        for (int j = i+1; j < mdp->n; j++)
        {
            dx += mdp->distance[i][j]*xSol[j];
            fx += mdp->distance[i][j]*xSol[i]*xSol[j];
        }
        cut += dx * (x[i] - xSol[i]);
    }
    cut += fx;
    *num_cuts += 1;
    add(theta <= cut).end();
    xSol.end();   

    return;
}

class OuterApproximationSolver{

    public:

        // Report
        IloInt num_cuts = 0;
        IloNum obj_val;
        IloNum solve_time;
        IloNum gap;

        // Problem
        MaxDiversityProblem *mdp;
        IloNum *rho;

        // Settings
        vector<int>* timelimit = 0;

        // Constructor
        OuterApproximationSolver(MaxDiversityProblem & problem, double & eigen_value, vector<int> & time_limits){
            mdp = &problem;
            timelimit = &time_limits;
            rho = &eigen_value;
        }

        void solve(vector<Result> &results) {

            void get_rho();

            IloEnv env;

            try {
                
                IloModel model(env);

                IloBoolVarArray x(env, mdp->n);
                IloNumVar theta(env,0,mdp->upper_bound, "theta");

                // Choose m
                model.add( IloSum(x) == mdp->p );
                
                // Objective
                model.add(IloMaximize(env,theta));

                // Solver
                int next_time = 0;
                IloNum time1, time2;
                IloCplex cplex(model);
                cplex.use(CutCallback(env, x, theta, mdp, rho, &num_cuts));

                // Parameters
                cplex.setParam(IloCplex::Param::Threads, 1); 
                cplex.setParam(IloCplex::Param::ClockType, 1);
                cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, 0);
                if (timelimit->back() > 0){
                    cplex.setParam(IloCplex::Param::TimeLimit, timelimit->back());
                    if (timelimit->size() > 1){
                        cplex.use(TimeLimits(env,&time1,&next_time,timelimit,&results,&num_cuts));
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
                    res.num_cuts = num_cuts;
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
    // oa.exe (0)1 
    // instance_set (1)2
    // instance_file (2)3
    // output_file (3)4
    // p_ratio (4)5
    // eigen_value (5)4
    // timelimit (6)7

    // Determine options:
    if (argc < 7){
        cout << "Incorrect arguments [instance_set instance_file output_file p_ratio eigen_value timelimit(s)]" ;
        return 0;
    }

    // Get file arguments
    string instance_set  = argv[1];
    string instance_file = argv[2];
    string output_file   = argv[3];
    
    int p_ratio;
    istringstream ss4(argv[4]);
    ss4 >> p_ratio;

    double eigen_value;
    istringstream ss5(argv[5]);
    ss5 >> eigen_value;

    vector<int> timelimit;
    for (int i = 6; i < argc; i++)
    {
        int _timelimit;
        istringstream ss6(argv[i]);
        ss6 >> _timelimit;
        timelimit.push_back(_timelimit);
    }
    sort(timelimit.begin(), timelimit.end());

    // Concat file paths
    string input_filepath  = "./data/" + instance_set + "/" + instance_file;
    string output_filepath = "./results/" + output_file;

    // Setup...
    MaxDiversityProblem mdp(input_filepath, p_ratio);
    OuterApproximationSolver oa(mdp, eigen_value, timelimit);

    // Solve...
    vector<Result> results;
    oa.solve(results);

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
                << "oa" << " "
                << results[i].obj_val << " "
                << results[i].gap << " "
                << results[i].solve_time << " "
                << results[i].num_cuts << endl;
    }
    output.close();

    return 0;
}
