/*
    It is a part of origin decsol.h that contains only functions for
    LUP decomposition of dense squere matrix for real (doulbe) and complex
    numbers (two matrix of [A + i*B], both double)
*/

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

#ifndef _DECSOL_H_
#define _DECSOL_H_

// Matrix Triangularization by Gaussian Elimination
int dec(const int n, double **A, int *ip);

// Solution of linear system A*x = b
void sol(const int n, double **A, double *b, int *ip);

// Matrix Triangularization by Gaussian Elimination for complex matrices
int decc(const int n, double **AR, double **AI, int *ip);

// Solution of linear system A*x = b -- complex matrices
void solc(const int n, double **AR, double **AI, double *br,
	double *bi, int *ip);

#endif /* _DECSOL_H_ */
