/***************************************************************************
                                decsol.h
                             -------------------
    modified for C by:   : Blake Ashby
    last modified        : Nov 15, 2002
    email                : bmashby@stanford.edu

Helper routines for StiffIntegratorT class. Modified from original Fortran
code provided by:

	E. Hairer & G. Wanner
	Universite de Geneve, dept. de Mathematiques
	CH-1211 GENEVE 4, SWITZERLAND
	E-mail : HAIRER@DIVSUN.UNIGE.CH, WANNER@DIVSUN.UNIGE.CH

 ***************************************************************************/

#ifndef _DECSOL_NEW_H_
#define _DECSOL_NEW_H_

// Matrix Triangularization by Gaussian Elimination
int dec_row(const int n, double *A, int *ip);

// Solution of linear system A*x = b
void sol_row(const int n, double *A, double *b, int *ip);

// Matrix Triangularization by Gaussian Elimination
int dec_column(const int n, double *A, int *ip);

// Solution of linear system A*x = b
void sol_column(const int n, double *A, double *b, int *ip);

// // Matrix Triangularization by Gaussian Elimination for complex matrices
// int decc(const int n, double **AR, double **AI, int *ip);

// // Solution of linear system A*x = b -- complex matrices
// void solc(const int n, double **AR, double **AI, double *br,
// 	double *bi, int *ip);

#endif /* _DECSOL_NEW_H_ */
