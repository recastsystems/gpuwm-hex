#pragma once
#include "heat.hpp"
#include "conversions.hpp"
namespace direct {
// Completed tendency cut only. This is NOT rrfs_mynnedmf_run_l16's whole backward.
// One serial owner per column; P=float production operands, double cotangents.
// Reads input HNI and seed HNO; adds to bar HNI. Buffers must not alias.
template<class P>bool heat_column_backward(int n,P dt,const P*input,const double*seed,double*bar){if(n<3||n>59)return false;HeatState<P>s{};heat_forward(n,dt,input,static_cast<P*>(nullptr),s);heat_reverse(n,dt,input,s,seed,bar);return true;}
}
