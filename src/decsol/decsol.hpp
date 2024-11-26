/***************************************************************************
                                decsol.h
                             -------------------
    modified for C by:   : Blake Ashby
    last modified        : Nov 15, 2002
    email                : bmashby@stanford.edu

	modified by  		 : Vitaly Nikolaev
    last modified        : 2024
    email                : vs.nikolaev.bmstu@gmail.com

Helper routines for StiffIntegratorT class. Modified from original Fortran
code provided by:

	E. Hairer & G. Wanner
	Universite de Geneve, dept. de Mathematiques
	CH-1211 GENEVE 4, SWITZERLAND
	E-mail : HAIRER@DIVSUN.UNIGE.CH, WANNER@DIVSUN.UNIGE.CH

 ***************************************************************************/

#ifndef _DECSOL_NEW_H_
#define _DECSOL_NEW_H_

#include <cmath>
#include <utility>
#include <concepts>

#include <omp.h>

#define DECSOL_THREADS_NUM 8

// Matrix Triangularization by Gaussian Elimination
int dec_row(const int n, double *A, int *ip);

// Solution of linear system A*x = b
void sol_row(const int n, double *A, double *b, int *ip);

/*-----------------------------------------------------------------------
	Matrix Triangularization by Gaussian Elimination

	Input:
		n		order of matrix
		A		matrix to be triangularized, column-wise storage model

	Output:
		A[i][j], i <= j		upper triangular factor, U
		A[i][j], i > j		multipliers = lower triangular factor, i - l
		ip[k], k < n - 1	index of k-th pivot row
		ip[n-1]				(-1)^(number of interchanges) or 0
		ier 				0 if matrix A is nonsingular, or k if found
								to be singular at stage k

	Use sol to obtain solution of linear system

	determ(A) = ip[n-1]*A[0][0]*A[1][1]*...*A[n-1][n-1]

	If ip[n-1] = 0, A is singular, sol will divide by zero

	Reference:
		C.B. Moler, Algorithm 423, Linear Equation Solver,
		C.A.C.M. 15 (1972), p. 274.

-----------------------------------------------------------------------*/
template<std::floating_point T>
int dec_column(const int n, T* A, int *ip)
{
	int m, ier;
	T t;

	ier = 0;
	ip[n-1] = 1;
	if (n != 1) {
		for (int k = 0; k < n - 1; ++k) {

			struct MyMax {
				T max;
				int index;
			} myMaxStruct {A[k + n * k], k};

			#pragma omp declare reduction(maximo : struct MyMax : omp_out = omp_in.max > omp_out.max ? omp_in : omp_out)

			#pragma omp parallel for reduction(maximo : myMaxStruct)
			for (int i = k + 1; i < n; ++i) {
				if (fabs(A[i + n * k]) > fabs(myMaxStruct.max)) {
					myMaxStruct.max = A[i + n * k];
					myMaxStruct.index = i;
				};
			}

			m = myMaxStruct.index;
			t = myMaxStruct.max;

			ip[k] = m;
			if (m != k) {
				ip[n-1] = -ip[n-1];
				std::swap(A[k + n * k], A[m + n * k]);
			}
			if (t == 0.0) {
				ier = k;
				ip[n-1] = 0;
				return ier;
			}
			t = 1.0/t;

			#pragma omp parallel for num_threads(DECSOL_THREADS_NUM)
			for (std::size_t i = k + 1; i < n; ++i)
				A[i + n * k] *= -t;

			#pragma omp parallel for num_threads(DECSOL_THREADS_NUM)
			for (int j = k + 1; j < n; ++j) {
				const T t = A[m + n * j];
				std::swap(A[m + n * j], A[k + n * j]);
				if (t != 0.0)
					for (int i = k + 1; i < n; ++i)
						A[i + n * j] += A[i + n * k] * t;
			}
		}
	}

	if (A[(n-1) + n * (n-1)] == 0.0) {
		ier = n;
		ip[n-1] = 0;
	}

	return ier;

} //dec_column

/*-----------------------------------------------------------------------
	Solution of linear system A*x = b

	Input:
		n		order of matrix
		A		triangularized matrix obtained from dec
		b		right hand side vector
		ip		pivot vector obtained from dec

	Do not use if dec has set ier != 0

	Output:
		b		solution vector, x

-----------------------------------------------------------------------*/
template<typename T>
void sol_column(const int n, T* A, T* b, int *ip)
{
	int m;
	T t;

	if (n != 1) {
		for (int k = 0; k < n - 1; ++k) {

			m = ip[k];
			t = b[m];

			std::swap(b[m], b[k]);

			#pragma omp parallel for num_threads(DECSOL_THREADS_NUM)
			for (int i = k + 1; i < n; ++i)
				b[i] += A[i + n * k] * t;

		}

		for (int k = 0; k < n - 1; ++k) {
			const int kb = n - k - 1;
			b[kb] = b[kb]/A[kb + n * kb];
			t = -b[kb];
			#pragma omp parallel for num_threads(DECSOL_THREADS_NUM)
			for (int i = 0; i < n - k - 1; ++i) 
				b[i] += A[i + n * kb] * t;
		}
	}
	
	b[0] = b[0]/A[0];
	
	return;
	
} //sol_column

#endif /* _DECSOL_NEW_H_ */
