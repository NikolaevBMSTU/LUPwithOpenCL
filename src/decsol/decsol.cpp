/***************************************************************************
                                decsol.c
                             -------------------
    modified for C by:   : Blake Ashby
    last modified        : Nov 15, 2002
    email                : bmashby@stanford.edu

	fixed bugs and modified by  : Vitaly Nikolaev
    last modified        		: 2024
    email                		: vs.nikolaev.bmstu@gmail.com

Helper routines for StiffIntegratorT class. Modified from original Fortran
code provided by:

	E. Hairer & G. Wanner
	Universite de Geneve, dept. de Mathematiques
	CH-1211 GENEVE 4, SWITZERLAND
	E-mail : HAIRER@DIVSUN.UNIGE.CH, WANNER@DIVSUN.UNIGE.CH

 ***************************************************************************/

#include "decsol.hpp"


int dec_row(const int n, double *A, int *ip)
{

/*-----------------------------------------------------------------------
	Matrix Triangularization by Gaussian Elimination

	Input:
		n		order of matrix
		A		matrix to be triangularized

	Output:
		A[i][j], i <= j		upper triangular factor, U
		A[i][j], i > j		multipliers = lower triangular factor, i - l
		ip[k], k < n - 1	index of k-th pivot rowf
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

	int kp1, m, nm1, k, ier;
	double t;

	ier = 0;
	ip[n-1] = 1;
	if (n != 1) {
		nm1 = n - 1;
		for (k = 0; k < nm1; ++k) {
			kp1 = k + 1;
			m = k;
			for (int i = kp1; i < n; ++i) {
				if (fabs(A[i * n + k]) > fabs(A[m * n + k])) m = i;
			}
			ip[k] = m;
			t = A[m * n + k];
			if (m != k) {
				ip[n-1] = -ip[n-1];
				A[m * n + k] = A[k * n + k];
				A[k * n + k] = t;
			}
			if (t == 0.0) {
				ier = k;
				ip[n-1] = 0;
				return (ier);
			}
			t = 1.0/t;

			for (int i = kp1; i < n; ++i)
				A[i * n + k] *= -t;

			for (int j = kp1; j < n; ++j) {
				t = A[m * n + j];
				A[m * n + j] = A[k * n + j];
				A[k * n + j] = t;
				if (t != 0.0)
					for (int i = kp1; i < n; ++i)
						A[i * n + j] += A[i * n + k]*t;
			}
		}
	}
	k = n;
	if (A[(n-1) * n + n-1] == 0.0) {
		ier = k;
		ip[n-1] = 0;
	}

	return (ier);

} //dec_row

void sol_row(const int n, double *A, double *b, int *ip)
{

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

	int m, kb, km1, nm1, kp1;
	double t;

	if (n != 1) {
		nm1 = n - 1;
		for (int k = 0; k < nm1; ++k) {
			kp1 = k + 1;
			m = ip[k];
			t = b[m];
			b[m] = b[k];
			b[k] = t;
			for (int i = kp1; i < n; ++i)
				b[i] += A[i * n + k]*t;
		}
		for (int k = 0; k < nm1; ++k) {
			km1 = n - k - 2;
			kb = km1 + 1;
			b[kb] = b[kb]/A[kb * n + kb];
			t = -b[kb];
			for (int i = 0; i <= km1; ++i) 
				b[i] += A[i * n + kb]*t;
		}
	}
	b[0] = b[0]/A[0];
	
	return;
	
} //sol_row
