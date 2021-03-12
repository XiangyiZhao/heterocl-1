#=============================================================================
# run-fixed.tcl 
#=============================================================================
# @brief: A Tcl script for fixed-point experiments.
#
# @desc: This script runs a batch of simulation & synthesis experiments
# to explore trade-offs between accuracy, performance, and area for 
# fixed-point implementation of the CORDIC core.
#
# 1. The user specifies a list of bitwidth pairs, i.e., (TOT_WIDTH, INT_WIDTH)
# 2. Results are collected in a single file ./result/fixed-result.csv
# 3. out.dat from each individual experiment is also copied to ./result

# Project name
set hls_prj "hls.prj"

# Open/reset the project
open_project ${hls_prj} -reset
# Top function of the design is "cordic"
set_top default_function

# Add design and testbench files
add_files test.cpp
#add_files -tb mmult_accel.cpp

open_solution "solution1"
# Use Zynq device
set_part {xc7z020clg484-1}

# Target clock period is 10ns
create_clock -period 10

### INSERT OPTIMIZATION DIRECTIVES HERE ###






###########################################

# Simulate the C++ design
#csim_design
# Synthesize the design
csynth_design

# We will skip C-RTL cosimulation for now
#cosim_design

#---------------------------------------------
# Collect & dump out results from HLS reports
#---------------------------------------------

quit
