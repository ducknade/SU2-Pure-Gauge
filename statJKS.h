#ifndef FUNC_H
#define FUNC_H

double autoCorrelation(double* data, int num);
double mean(double* data, int num, double& average);
double mean1(double* data, int num, double& average);
double correlator(double* data, int num, int m, double average);
double jackknife(double* data, int num, int binSize, double& average);
double jackknife_2_log_dividing(double* data1, double* data2, int num, int binSize, double& average);
double jackknife_2_power_dividing(double* data1, double* data2, int num, int binSize, double power1, double power2, double& average);
double jackknife_data_output(double* data, int num, int binSize, double& average, double* modified, int& modified_num);

#endif
