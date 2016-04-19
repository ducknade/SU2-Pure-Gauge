#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include "statJKS.h"
#include <ctime>
#include <vector>
#include <omp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_bessel.h>
#include <sstream>
#include <cstring>
#include <fenv.h>

using namespace std;

#define PI 3.14159265358979323846

#define NTHREADS 32

#define NDIM 4 				// dimension of the system
#define NSIDE 16		// number of sites per dimension

#define PARTITION_NUMBER_0 4
#define PARTITION_NUMBER_1 2
#define PARTITION_NUMBER_2 2
#define PARTITION_NUMBER_3 2

// const int NUM_LINK_PER_THREAD = NUM_OF_LINK / NTHREADS;
// const int NUM_SITE_PER_THREAD = NUM_OF_SITE / NTHREADS;

#define NUM_LINK_PER_THREAD 8192
#define NUM_SITE_PER_THREAD 2048

int NUM_OF_SITE = (int)pow((double)NSIDE, (double)NDIM);
int NUM_OF_LINK = NUM_OF_SITE * NDIM;

class SU2
{
public:
	double a0;
	double a1;
	double a2;
	double a3;
	SU2();
	SU2(double b0, int rng_i); // set a0 to b0 and set random numbers to a1, a2 and a3.
	SU2(double b0, double b1, double b2, double b3);
	SU2(const SU2& SU2Object);
	SU2& operator =(const SU2& rightSide);
	SU2 dagger();
	SU2 transpose();
	SU2 star();
	void normalize();
};

SU2 operator *(const SU2& mat1, const SU2& mat2);
SU2 operator +(const SU2& mat1, const SU2& mat2);
SU2 operator -(const SU2& mat1, const SU2& mat2);
SU2 operator *(const SU2& mat1, const double lambda);
SU2 operator *(const double lambda, const SU2& mat2);

// simulation parameters
// const int Npara = 4;				// number of parameters for each SU(2) element on each site
double beta;         // coupling constant: \beta * e_0^2 = 4

SU2 *gauge_field;     // gauge field on each site: gauge_field[i][j] = a_j of site i.
SU2 *gauge_field_smear;

gsl_rng **rng;

vector<omp_nest_lock_t> omp_nest_locks(NUM_OF_LINK);
vector<omp_nest_lock_t> omp_nest_locks_smear(NUM_OF_LINK);

int updating_list[NTHREADS][NUM_LINK_PER_THREAD];
int measuring_list[NTHREADS][NUM_SITE_PER_THREAD];

ofstream error_output;

// declare some functions

void initialize();
void finalize();
int x_mu2site_id(const vector<int>& x_mu);
vector<int> site_id2x_mu(int site_id);
int site_dir2link_id(const vector<int>& x_mu, int direction);
int link_id2site_dir(int link_id, vector<int>& site);
SU2 site_dir2mat(const vector<int>& x_mu, int alpha);
SU2 site_dir2mat_smear(const vector<int>& x_mu, int alpha);

double collecting_nearby_info(vector<int>& x_mu, int direction, SU2& U_bar);
double collecting_nearby_info_spatial(vector<int>& x_mu, int direction, SU2& U_bar);

double measure_local_Wilson(const vector<int> &x_mu, int loop_size1, int loop_size2);
double measure_global_Wilson(int loop_size1, int loop_size2);

double measure_plaquette(vector<int> &x_mu);
double measure_all();

double measure_local_Wilson_RT(vector<int> x_mu, vector<int>R_vector, int T);
double measure_global_Wilson_RT(vector<int>R_vector, int T);

double measure_creutz(int I, int J);

double measure_polyakov(vector<int> x_mu);
double measure_polyakov_correlation(int num);

void measure_potential(ifstream& input, int start_config, int end_config, int step_interval, double beta_set, \
	double smear_epsilon, int smear_iteration, vector<int> dR, int T, \
	double& R, double& VR, double& sigma_VR, double& CR, double& sigma_CR);

SU2 temporal_link_integration(vector<int> x_mu, int alpha);

void smearing(double epsilon, int iteration);

void update();

void configuration2fits(ofstream &output);
void fits2configuration(ifstream &input, int step_num);
string int2str(int num);
string int2str(int num, int width);
string double2str(double num, int num_decimal_place);

void create_configuration(int start, int end);

void any_nan(int config);

int main(){
	
/*------------------------------	GENERATE LINK VARIABLE CONFIGURATION :)	------------------------------*/
	
	//
	//
	//
		
	// int NUM = 30000;
	//
	// beta = 2.5;
	//
	// string FILENAME = "/home/jt2798/SU2-Pure-Gauge/source/SU2_beta_25_03.jks";
	//
	// ofstream outstream;
	// outstream.open(FILENAME.c_str());
	// outstream << "### --- SU2 Pure Gauge  --- ###" << endl;
	// outstream << "### --- BETA --- ###" << "\t" << double2str(beta, 2) << endl;
	// outstream << "### --- NDIM --- ###" << "\t" << NDIM << endl;
	// outstream << "### --- NSIDE --- ###" << "\t" << NSIDE << endl;
	// outstream << "### --- NSITE --- ###" << "\t" << NUM_OF_SITE << endl;
	// outstream << "### --- NLINK --- ###" << "\t" << NUM_OF_LINK << endl;
	// outstream << "### --- NSTEP --- ###" << "\t" << NUM << endl;
	//
	// outstream << "### --- START CONFIGURATION --- ###" << endl;
	//
	// initialize();
	//
	// for(int j = 0; j < NUM; j++){
	// 	update();
	// 	any_nan(j);
	// 	configuration2fits(outstream);
	// 	cout << "config " << j << endl;
	// }
	//
	// outstream.close();
	//
	// finalize();

	//
	//
	//
	
/*------------------------------	GENERATE LINK VARIABLE CONFIGURATION ^^^	------------------------------*/
	
//
//
//
	
/*------------------------------	READ LINK VARIABLE CONFIGURATION :)	------------------------------*/
	
	//
	//
	//
	
	cout.setf(ios::fixed);
	cout.setf(ios::showpoint);
	cout.precision(8);

	string input_FILENAME = "/home/jt2798/SU2-Pure-Gauge/source/SU2_beta_25_03.jks";
	ifstream instream;
	instream.open(input_FILENAME.c_str());

	string output_FILENAME = "/home/jt2798/SU2-Pure-Gauge/output/SU2_beta_25_VR_output_03.dat";
	ofstream outstream;
	outstream.open(output_FILENAME.c_str(), ofstream::app);
	outstream.setf(ios::fixed);
	outstream.setf(ios::showpoint);
	outstream.precision(8);

	time_t now = time(0);
	char* time_cstr = ctime(&now);

	string time_str(time_cstr);
	time_str = string(time_str, 0, time_str.length() - 1);

	outstream << "### --- date and time: " << time_str << " --- ###" << endl;

	double R, VR, sigma_VR, CR, sigma_CR;
	int vec[33][4];

	int count = 0;
	for(int i = 0; i < 5; i++){
		for(int j = 0; j <= i; j++){
			for(int k = 0; k <= j; k++){

				if(i * i + j * j + k * k < 2) continue;

				vec[count][1] = k;
				vec[count][2] = j;
				vec[count][3] = i;

				count++;

			}
		}
	}

	for(int i = 0; i < 33; i++){

		vector<int> dR(4);
		dR[0] = 0;
		dR[1] = vec[i][1];
		dR[2] = vec[i][2];
		dR[3] = vec[i][3];

		double previous_VR = 0.;
		int T_extent = 3 + ceil(sqrt(dR[1] * dR[1] + dR[2] * dR[2] + dR[3] * dR[3]));

		for(int j = T_extent; j < T_extent + 1; j++){

			measure_potential(instream, 20001, 30001, 20, 2.5, \
				0.5, 10, dR, j, \
				R, VR, sigma_VR, CR, sigma_CR);

			outstream << R << "\t" << j << "\t"
				<< VR << "\t"
					<< sigma_VR << "\t"
						<< CR << "\t"
							<< sigma_CR << endl;

			cout << i << "\t" << R << "\t" << j << "\t"
				<< VR << "\t"
					<< sigma_VR << "\t"
						<< CR << "\t"
							<< sigma_CR << endl;

			if(abs(VR - previous_VR) < sigma_VR){
				break;
			}

			previous_VR = VR;
		}
	}

	instream.close();
	outstream.close();

	//
	//
	//
	
/*------------------------------	READ LINK VARIABLE CONFIGURATION ^^^	------------------------------*/
	
//
//
//
	
/*------------------------------	TESTING CASE	------------------------------*/
	
	//
	//
	//
	
	// for(int i = 10; i < 11; i++){
	//
	// 	initialize();
	//
	// 	beta = 1.0;
	// 	string FILENAME = "/home/jt2798/SU2Gauge/SU2_beta_10_02.jks";
	// 	ifstream instream;
	// 	instream.open(FILENAME.c_str());
	//
	// 		for(int j = 1; j < 20; j++){
	// 			fits2configuration(instream, j);
	// 			cout << "step " << j << "\t";
	// 			cout << measure_global_Wilson(1, 1) << endl;
	// 		}
	//
	// 	finalize();
	// 	instream.close();
	// }
	
	//
	//
	//
	
/*------------------------------	TESTING CASE ^^^	------------------------------*/

//
//
//

/*------------------------------ MEASURE WHILE UPDATING	------------------------------*/
	
	//
	//
	//
	
	// ofstream thermal("/home/jt2798/SU2Gauge/thermal.dat");
	// ofstream thermal_report("/home/jt2798/SU2Gauge/thermal_report.dat");
	// thermal.setf(ios::fixed);
	// thermal.setf(ios::showpoint);
	// thermal.precision(6);
	// thermal_report.setf(ios::fixed);
	// thermal_report.setf(ios::showpoint);
	// thermal_report.precision(6);
	//
	// beta = 2.5;
	// int T = 5;
	//
	// vector<int> dR(4);
	// dR[0] = 0;
	// dR[1] = 4;
	// dR[2] = 0;
	// dR[3] = 0;
	//
	// int thermaliztion_sweep = 18000;
	// int sweep_bw_measurements = 100;
	//
	// int num_measurements = 2000;
	//
	// int starting_point = 19000;
	//
	// initialize();
	//
	// int count_measurements = 0;
	// int config = 0;
	//
	// double value_list1[num_measurements];
	// double value_list2[num_measurements];
	//
	// // for(int j = 0; j < thermaliztion_sweep; j++){
	// // 	update();
	// // 	config++;
	// // 	// cout << j << endl;
	// // }
	//
	// string FILENAME = "/home/jt2798/SU2Gauge/SU2_beta_25_03.jks";
	// ifstream instream;
	// instream.open(FILENAME.c_str());
	//
	// fits2configuration(instream, starting_point);
	// // cout << gauge_field[120].a0 << endl;
	//
	// for(int i = 0; i < num_measurements; i++){
	//
	// 	memcpy(gauge_field_smear, gauge_field, NUM_OF_LINK * sizeof(SU2));
	// 	smearing(0.5, 20);
	//
	// 	// cout << gauge_field[120].a0 << endl;
	//
	// 	double measurement1 = measure_global_Wilson_RT(dR, T);
	// 	double measurement2 = measure_global_Wilson_RT(dR, T + 1);
	//
	// 	value_list1[i] = measurement1;
	// 	value_list2[i] = measurement2;
	//
	// 	thermal << config + starting_point << "\t" << measurement1 << "\t" << measurement2 << endl;
	// 	cout << config + starting_point << "\t" << measurement1 << "\t" << measurement2 << endl;
	//
	// 	for(int j = 0; j < sweep_bw_measurements; j++){
	// 		update();
	// 		config++;
	// 	}
	// }
	//
	// double R2 = dR[1] * dR[1] + dR[2] * dR[2] + dR[3] * dR[3];
	// double R = sqrt(R2);
	// double VR, CR;
	//
	// // VR = log(avg1 / avg2);
	// double sigma_VR = jackknife_2_log_dividing(value_list1, value_list2, num_measurements, 2, VR);
	// // CR = pow(avg1, T + 1) / pow(avg2, T);
	// double sigma_CR = jackknife_2_power_dividing(value_list1, value_list2, num_measurements, 2, T + 1, T, CR);
	//
	// finalize();
	//
	// thermal_report << R << "\t" << VR << "\t" << sigma_VR << "\t" << CR << "\t" << sigma_CR << endl;
	
	//
	//
	//
	
/*------------------------------	MEASURE WHILE UPDATING ^^_^	------------------------------*/

//
//
//

/*------------------------------	STATISTICS TESTING CASE	------------------------------*/
	
	//
	//
	//
	
	// string FILENAME = "/vega/astro/users/jt2798/SU2/SU2_beta_25_VR_output_03_21.dat";
	// ifstream instream;
	// instream.open(FILENAME);
	//
	// double value_list1[80];
	// double value_list2[80];
	//
	// for(int j = 0; j < 80; j++){
	// 	instream >> value_list1[j];
	// 	instream >> value_list2[j];
	// }
	//
	// double avg;
	//
	// cout << autoCorrelation(value_list1, 80) << endl;
	// cout << autoCorrelation(value_list2, 80) << endl;
	//
	// cout << jackknife_2_power_dividing(value_list1, value_list2, 80, 4, 8., 7., avg) << endl;
	// cout << avg << endl;
	//
	// instream.close();
	
	//
	//
	//
	
/*------------------------------	STATISTICS TESTING CASE ^^^	------------------------------*/
	
}

void measure_potential(ifstream& input, int start_config, int end_config, int step_interval, double beta_set, \
	double smear_epsilon, int smear_iteration, vector<int> dR, int T, \
	double& R, double& VR, double& sigma_VR, double& CR, double& sigma_CR){
	 	
	ofstream outstatus("/home/jt2798/SU2Gauge/status_03.dat", ofstream::app);
	outstatus.setf(ios::fixed);
	outstatus.setf(ios::showpoint);
	outstatus.precision(6);
		
	initialize();

	beta = beta_set;
	
	int num_measurements = (end_config - start_config) / step_interval + 1;
	int count_measurements = 0;

	double value_list1[num_measurements];
	double value_list2[num_measurements];
	
	for(int j = start_config; j < end_config;){

		fits2configuration(input, j);
		outstatus << "config " << "\t" << j << ":\t";
		
		const clock_t check_point_1 = clock();
		
		smearing(smear_epsilon, smear_iteration);

		const clock_t check_point_2 = clock();
		
		double measurement1 = measure_global_Wilson_RT(dR, T);
		double measurement2 = measure_global_Wilson_RT(dR, T + 1);

		if(j > 2000){
			value_list1[count_measurements] = measurement1;
			value_list2[count_measurements] = measurement2;
			
			count_measurements++;
		}

		const clock_t check_point_3 = clock();
		
		// outstatus << "smearing = " << double( check_point_2 - check_point_1 ) / CLOCKS_PER_SEC
		// 	<< ", " << "measurement = " << double( check_point_3 - check_point_2 ) / CLOCKS_PER_SEC << ".\t"
		// 		<< measurement1 << "\t" << measurement2 << endl;
		
		outstatus << measurement1 << "\t" << measurement2 << endl;
		
		// cout << "smearing = " << double( check_point_2 - check_point_1 ) / CLOCKS_PER_SEC
		// 	<< ", " << "measurement = " << double( check_point_3 - check_point_2 ) / CLOCKS_PER_SEC << ".\t"
		// 		<< value_list1[count_measurements] << "\t" << value_list2[count_measurements] << endl;
		
		// cout << value_list1[j - start_config] << "\t" << value_list2[j - start_config] << endl;
		
		j += step_interval;
	}

	// double avg1;
	// double std1 = jackknife(value_list1, end_config - start_config, 2, avg1);
	// double avg2;
	// double std2 = jackknife(value_list2, end_config - start_config, 2, avg2);
		
	double R2 = dR[1] * dR[1] + dR[2] * dR[2] + dR[3] * dR[3];
	
	R = sqrt(R2);
	
	// VR = log(avg1 / avg2);
	sigma_VR = jackknife_2_log_dividing(value_list1, value_list2, count_measurements, 2, VR);
	// CR = pow(avg1, T + 1) / pow(avg2, T);
	sigma_CR = jackknife_2_power_dividing(value_list1, value_list2, count_measurements, 2, T + 1, T, CR);

	finalize();
}

void create_configuration(int start, int end){
	
	int NUM = 100;

	for(int i = start; i < end; i++){

		beta = i * 0.2;

		ostringstream os;
	    os.setf(ios::fixed);
	    os.setf(ios::showpoint);
	    os.precision(2);
			
		os << beta;
		string beta_str = os.str();

		string FILENAME = "/vega/astro/users/jt2798/SU2/SU2_beta_" + int2str(i * 2, 2) + ".jks";

		ofstream outstream;
		outstream.open(FILENAME.c_str());
		outstream << "SU2 Pure Gauge" << endl;
		outstream << "BETA" << "\t" << double2str(beta, 2) << endl;
		outstream << "NDIM" << "\t" << NDIM << endl;
		outstream << "NSIDE" << "\t" << NSIDE << endl;
		outstream << "NSITE" << "\t" << NUM_OF_SITE << endl;
		outstream << "NLINK" << "\t" << NUM_OF_LINK << endl;
		outstream << "NSTEP" << "\t" << NUM << endl;
		
		outstream << "### --- START CONFIGURATION --- ###" << endl;

		initialize();

		for(int j = 0; j < NUM; j++){
			update();
			configuration2fits(outstream);
			cout << "step " << j << endl;
		}

		outstream.close();
		
		finalize();
	}
}

SU2::SU2(){
	a0 = 1.;
	a1 = 0.;
	a2 = 0.;
	a3 = 0.;
}

SU2::SU2(double b0, int rng_i){

	if(!(b0 * b0 < 1.)){
		error_output << "CONSTRUCTOR INITIAL VALUE GREATER THAN ONE." << endl;
		exit(-2);
	}

	this->a0 = b0;
	double r = sqrt(1. - b0 * b0);
	double u = gsl_rng_uniform(rng[rng_i]) * 2. - 1.;
	double phi = gsl_rng_uniform(rng[rng_i]) * PI * 2.;
	
	if(u < -1. || u > 1. || isnan(u)){
		error_output << "u and phi NAN." << endl;
		error_output << u << "\t" << phi << endl;
		exit(-2);
	}
	
	this->a1 = r * sqrt(1. - u * u) * cos(phi);
	this->a2 = r * sqrt(1. - u * u) * sin(phi);
	this->a3 = r * u;
}

SU2::SU2(double b0, double b1, double b2, double b3){
	this->a0 = b0;
	this->a1 = b1;
	this->a2 = b2;
	this->a3 = b3;
}

SU2::SU2(const SU2& SU2Object){
	a0 = SU2Object.a0;
	a1 = SU2Object.a1;
	a2 = SU2Object.a2;
	a3 = SU2Object.a3;
}

SU2& SU2::operator =(const SU2& rightSide){
	a0 = rightSide.a0;
	a1 = rightSide.a1;
	a2 = rightSide.a2;
	a3 = rightSide.a3;
	
	return *this;
}

SU2 SU2::dagger(){
	return SU2(a0, -a1, -a2, -a3);
}

SU2 SU2::transpose(){
	return SU2(a0, a1, -a2, a3);
}

SU2 SU2::star(){
	return SU2(a0, -a1, a2, -a3);
}

void SU2::normalize(){
	double r = sqrt(this->a0 * this->a0 + this->a1 * this->a1 + this->a2 * this->a2 + this->a3 * this->a3);
	
	this->a0 /= r;
	this->a1 /= r;
	this->a2 /= r;
	this->a3 /= r;
}

SU2 operator *(const SU2& mat1, const SU2& mat2){
	double c0 = mat1.a0 * mat2.a0 - mat1.a1 * mat2.a1 - mat1.a2 * mat2.a2 - mat1.a3 * mat2.a3;
	double c1 = mat1.a0 * mat2.a1 + mat1.a1 * mat2.a0 - mat1.a2 * mat2.a3 + mat1.a3 * mat2.a2;
	double c2 = mat1.a0 * mat2.a2 + mat1.a2 * mat2.a0 - mat1.a3 * mat2.a1 + mat1.a1 * mat2.a3;
	double c3 = mat1.a0 * mat2.a3 + mat1.a3 * mat2.a0 - mat1.a1 * mat2.a2 + mat1.a2 * mat2.a1;
	
	return SU2(c0, c1, c2, c3);
}

SU2 operator +(const SU2& mat1, const SU2& mat2){	
	return SU2(mat1.a0 + mat2.a0, mat1.a1 + mat2.a1, mat1.a2 + mat2.a2, mat1.a3 + mat2.a3);
}

SU2 operator -(const SU2& mat1, const SU2& mat2){	
	return SU2(mat1.a0 - mat2.a0, mat1.a1 - mat2.a1, mat1.a2 - mat2.a2, mat1.a3 - mat2.a3);
}

SU2 operator *(const SU2& mat1, const double lambda){	
	return SU2(mat1.a0 * lambda, mat1.a1 * lambda, mat1.a2 * lambda, mat1.a3 * lambda);
}

SU2 operator *(const double lambda, const SU2& mat2){	
	return SU2(mat2.a0 * lambda, mat2.a1 * lambda, mat2.a2 * lambda, mat2.a3 * lambda);
}

void partition(){
	
	// ofstream outtest("partition_test.dat");
	
	vector<int> partition_scheme(4);
	partition_scheme[0] = PARTITION_NUMBER_0;
	partition_scheme[1] = PARTITION_NUMBER_1;
	partition_scheme[2] = PARTITION_NUMBER_2;
	partition_scheme[3] = PARTITION_NUMBER_3;
	
	vector<int> nside_per_partition(4);
	nside_per_partition[0] = NSIDE / partition_scheme[0];
	nside_per_partition[1] = NSIDE / partition_scheme[1];
	nside_per_partition[2] = NSIDE / partition_scheme[2];
	nside_per_partition[3] = NSIDE / partition_scheme[3];
	
	vector<int> Region_Distribution(NDIM);
	
	int ncount = 0;
	
	for(int i = 0; i < NTHREADS; i++){
		
		int rep = i;
		
		for(int j = 0; j < NDIM; j++){
			Region_Distribution[j] = rep % partition_scheme[j];
			rep /= partition_scheme[j];
		}
		
		int ncount_partition_link = 0;
		int ncount_partition_site = 0;
		vector<int> x_mu(4);
		
		for(int j = 0; j < nside_per_partition[0]; j++){
			for(int k = 0; k < nside_per_partition[1]; k++){
				for(int l = 0; l < nside_per_partition[2]; l++){
					for(int m = 0; m < nside_per_partition[3]; m++){
						
						x_mu[0] = j + Region_Distribution[0] * nside_per_partition[0];
						x_mu[1] = k + Region_Distribution[1] * nside_per_partition[1];
						x_mu[2] = l + Region_Distribution[2] * nside_per_partition[2];
						x_mu[3] = m + Region_Distribution[3] * nside_per_partition[3];
					
						measuring_list[i][ncount_partition_site] = x_mu2site_id(x_mu);
					
						ncount_partition_site++;
						
						// outtest << i << "\t" << measuring_list[i][ncount_partition_site - 1] << "\t"
						// 	<< x_mu[0] << "\t"
						// 		<< x_mu[1] << "\t"
						// 			<< x_mu[2] << "\t"
						// 				<< x_mu[3] << "\t" << endl;
					
					}
				}
			}
		}
		
		for(int n = 0; n < NDIM; n++){
			for(int o = 0; o < NUM_SITE_PER_THREAD; o++){
				updating_list[i][ncount_partition_link] = measuring_list[i][o] * NDIM + n;
				ncount_partition_link++;
				ncount++;
			}
		}
		
		// outtest << ncount_partition_link << "\t" << NUM_LINK_PER_THREAD << endl;
		// cout << ncount_partition_site << "\t" << NUM_SITE_PER_THREAD << endl;
		// cout << ncount << endl;
		
	}
}

void initialize(){
	
	feenableexcept(FE_INEXACT || FE_DIVBYZERO || FE_UNDERFLOW || FE_OVERFLOW || FE_INVALID);
	
	gsl_rng_env_setup();
	
	rng = new gsl_rng* [NUM_OF_LINK];
	for(int i = 0; i < NUM_OF_LINK; i++){
		rng[i] = gsl_rng_alloc(gsl_rng_ranlxs2);
		gsl_rng_set(rng[i], 10798 + i * 29);
		
		gsl_rng_uniform(rng[i]);
		gsl_rng_uniform(rng[i]);
		gsl_rng_uniform(rng[i]);
		gsl_rng_uniform(rng[i]);
		gsl_rng_uniform(rng[i]);
		
	}
	
	gauge_field = new SU2[NUM_OF_LINK];
	gauge_field_smear = new SU2[NUM_OF_LINK];
	
	for(int i = 0; i < NUM_OF_LINK; i++){
		omp_init_nest_lock(&(omp_nest_locks[i]));
		omp_init_nest_lock(&(omp_nest_locks_smear[i]));
	}
	
	partition();
	
	time_t now = time(0);
	char* dt = ctime(&now);
	string other(dt);
	string time_str = string(other, 0, other.length() - 1);
	
	error_output.open("/home/jt2798/SU2Gauge/ERROR.log", ios::app);
	error_output << "### --- date and time: " << time_str << " --- ###" << endl;
}

void finalize(){
	delete [] gauge_field;
	delete [] gauge_field_smear;

	for(int i = 0; i < NUM_OF_LINK; i++){
		omp_destroy_nest_lock(&(omp_nest_locks[i]));
		omp_destroy_nest_lock(&(omp_nest_locks_smear[i]));
	}
	
	for(int i = 0; i < NUM_OF_LINK; i++){
		gsl_rng_free(rng[i]);
	}
	
	delete [] rng;
	
	error_output.close();
}

int x_mu2site_id(const vector<int>& x_mu){
	
	// x_mu, mu = 0, 1, 2, 3, agrees with cpp indice convention
	
	int j = NDIM - 1;
	int output = 0;
	for(int i = 0; i < NDIM; i++){
		output *= NSIDE;
		output += (x_mu[j] % NSIDE + NSIDE) % NSIDE; // avoid negative module
		j--;
	}
	
	if(output < 0 || output >= NUM_OF_SITE){
		error_output << "SITE ID OUT OF BOUND." << endl;
		exit(-1);
	}
	
	return output;
}

vector<int> site_id2x_mu(int site_id){
	
	// x_mu, mu = 0, 1, 2, 3, agrees with cpp indice convention
	
	vector<int> output(NDIM);
	for(int i = 0; i < NDIM; i++){
		output[i] = site_id % NSIDE;
		// cout << i << endl;
		site_id /= NSIDE;
	}
	
	return output;
}

int site_dir2link_id(const vector<int>& x_mu, int direction){
	
	// x_mu, mu = 0, 1, 2, 3; direction = 0, 1, 2, 3; agrees with cpp indice convention
	
	if(direction < 0 || direction >= NDIM){
		error_output << "direction parameter out of range." << endl;
		exit(0);
	}
	int link_id = x_mu2site_id(x_mu) * NDIM + direction;
	return link_id; 
}

int link_id2site_dir(int link_id, vector<int>& site){
	
	// x_mu, mu = 0, 1, 2, 3, agrees with cpp indice convention
	
	site = site_id2x_mu(link_id / NDIM);
	return link_id % NDIM;
}

SU2 site_dir2mat(const vector<int>& x_mu, int alpha){
	
	// x_mu, mu = 0, 1, 2, 3, agrees with cpp indice convention; alpha = 1, 2, 3, 4 indicates the direction of the link variable
	
	if(alpha == 0){
		error_output << "ALPHA OUT OF RANGE" << endl;
		error_output << alpha << endl;
		exit(0);
	}
	
	int pos;
	SU2 return_SU2;
	
	if(alpha > 0){
		
		pos = site_dir2link_id(x_mu, alpha - 1);
		
		omp_set_nest_lock(&(omp_nest_locks[pos]));
		return_SU2 = gauge_field[pos];
		omp_unset_nest_lock(&(omp_nest_locks[pos]));
		
		return return_SU2;
	}
	else{
		vector<int> y_mu(x_mu);
		y_mu[-alpha - 1]--;

		pos = site_dir2link_id(y_mu, -alpha - 1);

		omp_set_nest_lock(&(omp_nest_locks[pos]));
		return_SU2 = gauge_field[pos].dagger();
		omp_unset_nest_lock(&(omp_nest_locks[pos]));

		return return_SU2;
	}
}

SU2 site_dir2mat_smear(const vector<int>& x_mu, int alpha){
	
	// x_mu, mu = 0, 1, 2, 3, agrees with cpp indice convention; alpha = 1, 2, 3, 4 indicates the direction of the link variable
	
	if(alpha == 0){
		error_output << "alpha out of range" << endl;
		error_output << alpha << endl;
		exit(0);
	}
	
	int pos;
	SU2 return_SU2;
	
	if(alpha > 0){
		
		pos = site_dir2link_id(x_mu, alpha - 1);
		
		omp_set_nest_lock(&(omp_nest_locks[pos]));
		return_SU2 = gauge_field_smear[pos];
		omp_unset_nest_lock(&(omp_nest_locks[pos]));
		
		return return_SU2;
	}
	else{
		vector<int> y_mu(x_mu);
		y_mu[-alpha - 1]--;

		pos = site_dir2link_id(y_mu, -alpha - 1);

		omp_set_nest_lock(&(omp_nest_locks_smear[pos]));
		return_SU2 = gauge_field_smear[pos].dagger();
		omp_unset_nest_lock(&(omp_nest_locks_smear[pos]));

		return return_SU2;
	}
}

double collecting_nearby_info(vector<int>& x_mu, int direction, SU2& U_bar){
	
	// return square root of the determinant k and set U_bar equal to the remainning SU2 matrix
	
	double d0 = 0.;
	double d1 = 0.;
	double d2 = 0.;
	double d3 = 0.;
	
	vector<int> y1_mu(x_mu);
	y1_mu[direction]++;
	
	int iter = 0;
	
	for(int i = 1; i < NDIM + 1; i++){
		
		if(i == direction + 1) continue;

		vector<int> y_mu(x_mu);
		SU2 MUL1(1., 0., 0., 0.);
		
		y_mu[direction]++;
		MUL1 = MUL1 * site_dir2mat(y_mu, i);
		
		y_mu[i - 1]++;
		MUL1 = MUL1 * site_dir2mat(y_mu, -(direction + 1));
		
		y_mu[direction]--;
		MUL1 = MUL1 * site_dir2mat(y_mu, -i);
		
		
		vector<int> z_mu(x_mu);
		SU2 MUL2(1., 0., 0., 0.);
		
		z_mu[direction]++;
		MUL2 = MUL2 * site_dir2mat(z_mu, -i);
		
		z_mu[i - 1]--;
		MUL2 = MUL2 * site_dir2mat(z_mu, -(direction + 1));
		
		z_mu[direction]--;
		MUL2 = MUL2 * site_dir2mat(z_mu, i);
		
		// vector<int> y2_mu(x_mu);
		// vector<int> y3_mu(x_mu);
		//
		// y2_mu[direction]++;
		// y2_mu[i - 1]++;
		// y3_mu[i - 1]++;
		//
		// SU2 U_alpha_p((site_dir2mat(y1_mu, i) * site_dir2mat(y2_mu, -(direction + 1))) * site_dir2mat(y3_mu, -i));
		//
		// vector<int> y4_mu(x_mu);
		// vector<int> y5_mu(x_mu);
		//
		// y4_mu[direction]++;
		// y4_mu[i - 1]--;
		// y5_mu[i - 1]--;
		//
		// SU2 U_alpha_m((site_dir2mat(y1_mu, -i) * site_dir2mat(y4_mu, -(direction + 1))) * site_dir2mat(y5_mu, i));

		d0 += MUL1.a0 + MUL2.a0;
		d1 += MUL1.a1 + MUL2.a1;
		d2 += MUL1.a2 + MUL2.a2;
		d3 += MUL1.a3 + MUL2.a3;		
	}
	
	double detroot = sqrt(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3);
	
	U_bar = SU2(d0 / detroot, d1 / detroot, d2 / detroot, d3 / detroot);
	
	return detroot;
}

double collecting_nearby_info_spatial_smear(vector<int>& x_mu, int direction, SU2& U_bar){
	
	// return square root of the determinant k and set U_bar equal to the remainning SU2 matrix
	
	double d0 = 0.;
	double d1 = 0.;
	double d2 = 0.;
	double d3 = 0.;
	
	vector<int> y1_mu(x_mu);
	y1_mu[direction]++;
	
	int iter = 0;
	
	for(int i = 1; i < NDIM + 1; i++){
		if(i == direction + 1) continue;
		if(i == 1) continue;

		vector<int> y_mu(x_mu);
		
		SU2 MUL1(1., 0., 0., 0.);
		
		y_mu[direction]++;
		
		MUL1 = MUL1 * site_dir2mat_smear(y_mu, i);
		
		y_mu[i - 1]++;

		MUL1 = MUL1 * site_dir2mat_smear(y_mu, -(direction + 1));
		
		y_mu[direction]--;

		MUL1 = MUL1 * site_dir2mat_smear(y_mu, -i);
		
		vector<int> z_mu(x_mu);
		
		SU2 MUL2(1., 0., 0., 0.);
		
		z_mu[direction]++;

		MUL2 = MUL2 * site_dir2mat_smear(z_mu, -i);
		
		z_mu[i - 1]--;

		MUL2 = MUL2 * site_dir2mat_smear(z_mu, -(direction + 1));
		
		z_mu[direction]--;

		MUL2 = MUL2 * site_dir2mat_smear(z_mu, i);

		// vector<int> y2_mu(x_mu);
		// vector<int> y3_mu(x_mu);
		//
		// y2_mu[direction]++;
		// y2_mu[i - 1]++;
		// y3_mu[i - 1]++;
		//
		// SU2 U_alpha_p((site_dir2mat(y1_mu, i) * site_dir2mat(y2_mu, -(direction + 1))) * site_dir2mat(y3_mu, -i));
		//
		// vector<int> y4_mu(x_mu);
		// vector<int> y5_mu(x_mu);
		//
		// y4_mu[direction]++;
		// y4_mu[i - 1]--;
		// y5_mu[i - 1]--;
		//
		// SU2 U_alpha_m((site_dir2mat(y1_mu, -i) * site_dir2mat(y4_mu, -(direction + 1))) * site_dir2mat(y5_mu, i));

		d0 += MUL1.a0 + MUL2.a0;
		d1 += MUL1.a1 + MUL2.a1;
		d2 += MUL1.a2 + MUL2.a2;
		d3 += MUL1.a3 + MUL2.a3;
		
	}
	
	double detroot = sqrt(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3);
	
	U_bar = SU2(d0 / detroot, d1 / detroot, d2 / detroot, d3 / detroot);
	
	return detroot;
}

double measure_plaquette(vector<int> &x_mu){
	
	double d0 = 0.;
	
	for(int i = 1; i < NDIM + 1; i++){
		for(int j = 1; j < i; j++){
			
			vector<int> y1_mu(x_mu);
			vector<int> y2_mu(x_mu);
			vector<int> y3_mu(x_mu);
	
			y1_mu[i - 1]++;
			y2_mu[i - 1]++;
			y2_mu[j - 1]++;
			y3_mu[j - 1]++;
			
			d0 += ((site_dir2mat(x_mu, i) * site_dir2mat(y1_mu, j)) * (site_dir2mat(y2_mu, -i) * site_dir2mat(y3_mu, -j))).a0;
		}
	}
	
	return NDIM * (NDIM - 1) / 2. - d0;
}

double measure_all(){
	double sum = 0.;
	int mul = NUM_OF_SITE * NDIM * (NDIM - 1) / 2;
	vector<int> x_mu;
	for(int i = 0; i < NUM_OF_SITE; i++){
		x_mu = site_id2x_mu(i);
		sum += measure_plaquette(x_mu);
		// cout << i << endl;
	}
	
	return sum / mul;
}

double measure_local_Wilson_RT(vector<int> x_mu, vector<int>R_vector, int T){
			
	SU2 MUL(1., 0., 0., 0.);
			
	vector<int> y_mu(x_mu);
			
	for(int k = 0; k < R_vector[1]; k++){
		MUL = MUL * site_dir2mat_smear(y_mu, 2);
		// MUL = MUL * site_dir2mat(y_mu, 2);
		y_mu[1]++;
	}
			
	for(int k = 0; k < R_vector[2]; k++){
		MUL = MUL * site_dir2mat_smear(y_mu, 3);
		// MUL = MUL * site_dir2mat(y_mu, 3);
		y_mu[2]++;
	}
			
	for(int k = 0; k < R_vector[3]; k++){
		MUL = MUL * site_dir2mat_smear(y_mu, 4);
		// MUL = MUL * site_dir2mat(y_mu, 4);
		y_mu[3]++;
	}
			
	for(int k = 0; k < T; k++){
		MUL = MUL * temporal_link_integration(y_mu, 1);
		// MUL = MUL * site_dir2mat_smear(y_mu, 1);
		y_mu[0]++;
	}
	
	for(int k = 0; k < R_vector[3]; k++){
		MUL = MUL * site_dir2mat_smear(y_mu, -4);
		// MUL = MUL * site_dir2mat(y_mu, -4);
		y_mu[3]--;
	}
	
	for(int k = 0; k < R_vector[2]; k++){
		MUL = MUL * site_dir2mat_smear(y_mu, -3);
		// MUL = MUL * site_dir2mat(y_mu, -3);
		y_mu[2]--;
	}
			
	for(int k = 0; k < R_vector[1]; k++){
		MUL = MUL * site_dir2mat_smear(y_mu, -2);
		// MUL = MUL * site_dir2mat(y_mu, -2);
		y_mu[1]--;
	}
	
	for(int k = 0; k < T; k++){
		MUL = MUL * temporal_link_integration(y_mu, -1);
		// MUL = MUL * site_dir2mat_smear(y_mu, -1);
		y_mu[0]--;
	}		
			
	return MUL.a0;
}

double measure_global_Wilson_RT(vector<int> R_vector, int T){
	
	double sum = 0.;
	int count = 0;
	
	omp_set_num_threads(NTHREADS);
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		
		for(int j = 0; j < NUM_SITE_PER_THREAD; j++){
						
			int i = measuring_list[tid][j];
			// test[i] = true;
		
			vector<int> x_mu(4);
			x_mu = site_id2x_mu(i);

			double individual = measure_local_Wilson_RT(x_mu, R_vector, T);
			
			if(isnan(individual)){
				error_output << "INDIVIDUAL NAN." << endl;
				exit(-1);
			}
		
			#pragma omp critical
			{
				sum += individual;
				count++;
			}
				
		}
		
	}
	
	if(count == 0){
		error_output << "COUNT EQUALS ZERO." << endl;
		exit(-1);
	}
	
	// cout << "count = " << count << endl;
	
	// cout << sum / count << endl;
	
	return sum / count;

}



double measure_local_Wilson(const vector<int> &x_mu, int loop_size1, int loop_size2){
	double d0 = 0.;
	
	for(int i = 1; i < NDIM + 1; i++){
		for(int j = 1; j < i; j++){
			
			int dir1 = i - 1;
			int dir2 = j - 1;
			
			SU2 MUL(1., 0., 0., 0.);
			
			vector<int> y_mu(x_mu);
			
			for(int k = 0; k < loop_size1; k++){
				MUL = MUL * site_dir2mat(y_mu, i);
				y_mu[dir1]++;
			}
			
			for(int k = 0; k < loop_size2; k++){
				MUL = MUL * site_dir2mat(y_mu, j);
				y_mu[dir2]++;
			}
			
			for(int k = 0; k < loop_size1; k++){
				MUL = MUL * site_dir2mat(y_mu, -i);
				y_mu[dir1]--;
			}
			
			for(int k = 0; k < loop_size2; k++){
				MUL = MUL * site_dir2mat(y_mu, -j);
				y_mu[dir2]--;
			}
			
			d0 += MUL.a0;
		}
	}
	
	return d0;
}

double measure_global_Wilson(int loop_size1, int loop_size2){
	double sum = 0.;
	int mul = NUM_OF_SITE * NDIM * (NDIM - 1) / 2;
	
	
	
	omp_set_num_threads(NTHREADS);
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		
		for(int j = 0; j < NUM_SITE_PER_THREAD; j++){
						
			int i = measuring_list[tid][j];
			// test[i] = true;
		
			vector<int> x_mu;
			x_mu = site_id2x_mu(i);

			double individual = measure_local_Wilson(x_mu, loop_size1, loop_size2);
		
			#pragma omp atomic
			sum += individual;	
		}
		
	}
	
	// cout << sum / mul << endl;
	
	return sum / mul;
}

double measure_creutz(int I, int J){
	
	if(I < 2 || J < 2){
		error_output << "Creutz Ratio" << endl;
		exit(0);
	}
	
	double EXP = measure_global_Wilson(I, J) * measure_global_Wilson(I - 1, J - 1) \
		 / (measure_global_Wilson(I - 1, J) * measure_global_Wilson(I, J - 1));
	
	// cout << EXP << "\t";
	
	return -0.5 * log(EXP * EXP);
}

double measure_polyakov(vector<int> x_mu){
	double d0 = 0.;
	SU2 MUL(1., 0., 0., 0.);
	
	vector<int> y_mu(x_mu);
	
	for(int k = 0; k < NSIDE; k++){
		MUL = MUL * site_dir2mat(y_mu, 1);
		y_mu[0]++;
	}
	
	return MUL.a0;
}

double measure_polyakov_correlation(int num){
	vector<int> x_mu(4);
	vector<int> x1_mu(4);
	vector<int> x2_mu(4);
	vector<int> x3_mu(4);
	x_mu[0] = 0;
	x1_mu[0] = 0;
	x2_mu[0] = 0;
	x3_mu[0] = 0;
	
	int count = 0;
	
	double pavg = 0.;
	double pcor = 0.;
	
	double p0, p1, p2, p3;
	
	for(int i = 0; i < NSIDE; i++){
		for(int j = 0; j < NSIDE; j++){
			for(int k = 0; k < NSIDE; k++){
				count++;
				
				x_mu[1] = i; x_mu[2] = j; x_mu[3] = k;
				x1_mu[1] = i; x1_mu[2] = j; x1_mu[3] = k;
				x2_mu[1] = i; x2_mu[2] = j; x2_mu[3] = k;
				x3_mu[1] = i; x3_mu[2] = j; x3_mu[3] = k;
				
				p0 = measure_polyakov(x_mu);
				
				x1_mu[1] += num;
				p1 = measure_polyakov(x1_mu);
				x2_mu[2] += num;
				p2 = measure_polyakov(x2_mu);
				x3_mu[3] += num;
				p3 = measure_polyakov(x3_mu);
				
				pcor += p0 * (p1 + p2 + p3) / 3.;
				
				pavg += p0;
				
			}
		}
	}
	
	pcor /= count;
	pavg /= count;
	
	// cout << pavg << endl;
	
	return pcor - pavg * pavg;
	
}

SU2 temporal_link_integration(vector<int> x_mu, int alpha){
	
	if(alpha * alpha != 1){
		error_output << "NON TEMPORAL LINK" << endl;
		exit(-1);
	}

	vector<int> y_mu(x_mu);

	SU2 U_bar;
	double detrootk;
	double fac;
	
	if(alpha > 0){
		
		detrootk = collecting_nearby_info(y_mu, 0, U_bar);
		fac = gsl_sf_bessel_In(2, beta * detrootk) / gsl_sf_bessel_In(1, beta * detrootk);
		
		return U_bar.dagger() * fac;
		
	}else{
		y_mu[0]--;
		detrootk = collecting_nearby_info(y_mu, 0, U_bar);
		fac = gsl_sf_bessel_In(2, beta * detrootk) / gsl_sf_bessel_In(1, beta * detrootk);
		
		return U_bar * fac;
		
	}
	
	// return site_dir2mat(x_mu, alpha);
}

void smearing(double epsilon, int iteration){
		
	// int count = 0;
		
	for(int it = 0; it < iteration; it++){
		
		omp_set_num_threads(NTHREADS);
		#pragma omp parallel
		{

			int tid = omp_get_thread_num();
			
			// if(tid == 15 || tid == 0)
			// 	cout << omp_get_num_threads() << endl;
			
			// #pragma omp critical
			for(int j = 0; j < NUM_LINK_PER_THREAD; j++){

				int i = updating_list[tid][j];
				// test[i] = true;

				vector<int> x_mu;
				int dir;
				SU2 U_bar;
				double detrootk;

				dir = link_id2site_dir(i, x_mu);

				if(dir == 0)
					continue;

				// #pragma omp critical
				detrootk = collecting_nearby_info_spatial_smear(x_mu, dir, U_bar);

				// detrootk = 0.78;

				omp_set_nest_lock(&(omp_nest_locks_smear[i]));

				gauge_field_smear[i] = gauge_field_smear[i] + epsilon * U_bar.dagger() * detrootk;
				gauge_field_smear[i].normalize();

				omp_unset_nest_lock(&(omp_nest_locks_smear[i]));

				#pragma omp barrier

			}
		}
		
			// for(int i = 0; i < NUM_OF_LINK; i++){
			//
			// 	vector<int> x_mu;
			// 	int dir;
			// 	SU2 U_bar;
			// 	double detrootk;
			//
			// 	dir = link_id2site_dir(i, x_mu);
			//
			// 	if(dir == 0)
			// 		continue;
			//
			// 	detrootk = collecting_nearby_info_spatial_smear(x_mu, dir, U_bar);
			//
			// 	omp_set_nest_lock(&(omp_nest_locks_smear[i]));
			//
			// 	gauge_field_smear[i] = gauge_field_smear[i] + epsilon * U_bar.dagger() * detrootk;
			// 	gauge_field_smear[i].normalize();
			//
			// 	omp_unset_nest_lock(&(omp_nest_locks_smear[i]));
			//
			// }
		
	}		
}

void update(){
	
	// bool test[NUM_OF_LINK];
	// for(int i = 0; i < NUM_OF_LINK; i++){
	// 	test[i] = false;
	// }
	
	omp_set_num_threads(NTHREADS);
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		
		// if (tid == 0){
		//   	    	int nthreads = omp_get_num_threads();
		// 	cout << "There are " << nthreads << " threads." << endl;
		// }
		
		for(int j = 0; j < NUM_LINK_PER_THREAD; j++){
						
			int i = updating_list[tid][j];
			// test[i] = true;
		
			vector<int> x_mu;
			int dir;
			SU2 U_bar;
			double detrootk;
			double z, p, w, weight;
			double b0;
			double lo, delta;
			double rn1, rn2;
								
			dir = link_id2site_dir(i, x_mu);
		
			// #pragma omp critical(sample)
			detrootk = collecting_nearby_info(x_mu, dir, U_bar);
		
			// cout << detrootk << endl;
			lo = exp(-2. * beta * detrootk);
			delta = 1. - lo;
			do{
				rn1 = gsl_rng_uniform(rng[i]);
				rn2 = gsl_rng_uniform(rng[i]);
				
				z = rn1 * delta + lo;
				p = rn2;
				w = log(z) / (beta * detrootk);
				
				weight = sqrt(-2. * w - w * w);
				
			}while(p > weight);

			b0 = w + 1.;
					
			SU2 Umat(b0, i);
		
			omp_set_nest_lock(&(omp_nest_locks[i]));
			gauge_field[i] = Umat * U_bar.dagger();
			omp_unset_nest_lock(&(omp_nest_locks[i]));

			#pragma omp barrier
		}
		
	}
	
	// for(int i = 0; i < NUM_OF_LINK; i++){
	// 	if(!test[i]){
	// 		cout << i << endl;
	// 		exit(-1);
	// 	}
	// }
}

void any_nan(int config){
	bool cri;
	for(int i = 0; i < NUM_OF_LINK; i++){
		cri = isnan(gauge_field[i].a0) || isnan(gauge_field[i].a1) \
			|| isnan(gauge_field[i].a2) || isnan(gauge_field[i].a3);
		
		if(cri){
			error_output << "config " << config << ": " << i << endl;
			exit(-1);
		}
	}
}

void configuration2fits(ofstream &output){
	
	output.write((char*)gauge_field, sizeof(SU2) * NUM_OF_LINK);

}

void fits2configuration(ifstream &input, int step_num){
	
	if(!input.good()){
		error_output << "IFSTREAM ERROR" << endl;
		exit(-1);
	}
	
	string line;
	string block_delimit = "START CONFIGURATION";
	
	int pos = -1;
	
	input.seekg(0, input.beg);
	
	while(getline(input, line))
	{
		// cout << line << endl;
		if(line.find(block_delimit) != string::npos){
			pos = input.tellg();
			break;
		}
	}
	
	// cout << pos << endl;
	
	if(pos == -1){
		error_output << "NO CONFIGUATION DATA FOUND" << endl;
		exit(-1);
	}
	
	input.seekg(NUM_OF_LINK * sizeof(SU2) * (step_num - 1), input.cur);
	
	if(input.eof()){
		error_output << "END OF FILE" << endl;
		exit(-1);
	}
	
	input.read((char*)gauge_field, NUM_OF_LINK * sizeof(SU2));
	
	memcpy(gauge_field_smear, gauge_field, NUM_OF_LINK * sizeof(SU2));
}

string int2str(int num)
{
    char tmp;
    string str = "";
    do
    {
        tmp = 48 + num % 10;
        str = tmp + str;
        num /= 10; 
    }while(num);
    return str;
}

string int2str(int num, int width){
	char tmp;
	int num_length = 0;
	std::string str = "";
	do
	{
		tmp = 48 + num % 10;
		str = tmp + str;
		num /= 10;
		num_length++; 
	}while(num);
	for(int i = 0; i < width - num_length; i++){
		str = "0" + str;
	}
	return str;
}

string double2str(double num, int num_decimal_place){
	ostringstream os;
    os.setf(ios::fixed);
    os.setf(ios::showpoint);
    os.precision(num_decimal_place);
		
	os << num;
	return os.str();
}
