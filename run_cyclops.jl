#! /usr/bin/env julia

println("run_cyclops.jl version 0.1");
println("Author: Hang.Ruan@uth.tmc.edu")

if length(ARGS) != 3
	error("Usage: <seed_symbol_lst> <expr_level> <out_prefix> ")
end

SEED_SYMBOL_FP = ARGS[1];
EXPR_LEVEL_FP = ARGS[2];
out_prefix = ARGS[3];

#### Should be modifed accordingly

srcdir = 
	"/extraspace/hruan/codes/CYCLOPS/src";

####

LOAD_PATH;
LOAD_PATH = LOAD_PATH[1:2];
push!(LOAD_PATH,srcdir);

using StatsBase
using MultivariateStats
using Distributions

include(string(srcdir,"/CYCLOPS_2a_AutoEncoderModule.jl"))
include(string(srcdir,"/CYCLOPS_2a_PreNPostprocessModule.jl"))
include(string(srcdir,"/CYCLOPS_2a_Seed.jl"))
include(string(srcdir,"/CYCLOPS_2a_MCA.jl"))
include(string(srcdir,"/CYCLOPS_2a_MultiCoreModule_Smooth.jl"))

include(string(srcdir,"/CYCLOPS_2a_CircularStats.jl"))
#include(string(srcdir,"/CYCLOPS_2a_CircularStats_U.jl"))

using CYCLOPS_2a_AutoEncoderModule
using CYCLOPS_2a_PreNPostprocessModule

using CYCLOPS_2a_MCA
using CYCLOPS_2a_MultiCoreModule_Smooth
using CYCLOPS_2a_Seed
#using CYCLOPS_2a_CircularStats_U
using CYCLOPS_2a_CircularStats

#using PyPlot

####

currdir=pwd();
outdir=string(currdir,"/", out_prefix);
mkdir(outdir);

#### SETTING SVD PARAMETERS
Frac_Var = .85; # Set Number of Dimensions of SVD to maintain this fraction of variance
DFrac_Var = .03; # Set Number of Dimensions of SVD to so that incremetal fraction of variance of var is at least this much
N_best =  20;  # Number of random initial conditions to try for each optimization

total_background_num = 40; # Number of background runs for global background refrence statistic
n_cores = 5; # Number of machine cores 

####

#### SETTING SEEDING PARAMETERS

Seed_MinCV = .14;
Seed_MaxCV = .75;
Seed_Blunt = .95;
MaxSeeds = 10000;

####

seed_symbol_file = readdlm(SEED_SYMBOL_FP);
seed_symbol_list = seed_symbol_file[:,1];

exp_data_file = readdlm(EXPR_LEVEL_FP);
exp_data = convert(Array{Float64,2}, exp_data_file[2:end,2:end]);

cd(outdir);

nprobes = size(exp_data)[1];

println(string("# Probes/Symbols: ", nprobes));

####

println("\nGetting seed data...");

####

cutrank = nprobes-MaxSeeds;
Seed_MinMean = (sort(vec(mean(convert(Array{Float64,2}, exp_data),2))))[cutrank];

println(string("Seed_MinMean: "), Seed_MinMean);

seed_symbols, seed_data =
	 getseed_test(exp_data_file,seed_symbol_list,
		Seed_MaxCV,Seed_MinCV,Seed_MinMean,Seed_Blunt);

writedlm(string(out_prefix, ".seed_symbols.txt"), seed_symbols);
writedlm(string(out_prefix, ".seed_data.txt"), seed_data);

seed_data = dispersion!(seed_data);

writedlm(string(out_prefix,".seed_data_disp.txt"), seed_data);

####

println("\nGetting eigengenes ...");

####

eigens_out, norm_seed_data = GetEigenGenes(seed_data,Frac_Var,DFrac_Var,30);

writedlm(string(out_prefix,".eigen_outs.txt"), eigens_out);
writedlm(string(out_prefix,".norm_seed_data.txt"), norm_seed_data);

println("\nEstimated eigen out: ", eigens_out);

####

println("\nEstimating phases ...");

####

estimated_phaselist,bestnet,global_var_metrics = CYCLOPS_Order(eigens_out,norm_seed_data,N_best);

writedlm(string(out_prefix,".estimated_phaselist.txt"), estimated_phaselist);

println("\nEstimated best neural net:");
println(bestnet);

println(string("best_stat_error "),global_var_metrics[1]);

writedlm(string(out_prefix,".global_var_metrics.txt"), global_var_metrics);

println("\nEvaluating smoothness ...");

estimated_phaselist1 = mod(estimated_phaselist + 2*pi,2*pi);
global_smooth_metrics = smoothness_measures(seed_data,norm_seed_data,estimated_phaselist1);

println(string("smooth_measure_eigen "), global_smooth_metrics[1]);
println(string("smooth_measure_eigen_sqrt "), global_smooth_metrics[2]);
println(string("smooth_measure_raw "), global_smooth_metrics[3]);
println(string("smooth_measure_raw_sq "), global_smooth_metrics[4]);

println();

pval=[1,1,1];

if global_smooth_metrics[1] <1
	println("\nEvaluating pval ...")
	pval = 
		multicore_backgroundstatistics_global_eigen(seed_data,
		eigens_out, N_best, total_background_num, global_var_metrics);
end

println(pval);


println("\nCyclops done.");

