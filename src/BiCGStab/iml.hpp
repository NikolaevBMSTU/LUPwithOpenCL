#ifndef IML_BICGSTAB_H_
#define IML_BICGSTAB_H_

#include <tuple>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

namespace ublas = boost::numeric::ublas;

// ===============================================================
// FROM IML++ (Iterative Methods Library) v. 1.2a
// https://math.nist.gov/iml++/
// https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
// Modifyed for UBLAS
// ===============================================================

//*****************************************************************
// Iterative template routine -- BiCGSTAB
//
// BiCGSTAB solves the unsymmetric linear system Ax = b 
// using the Preconditioned BiConjugate Gradient Stabilized method
//
// BiCGSTAB follows the algorithm described on p. 27 of the 
// SIAM Templates book.
//
// The return value indicates convergence within max_iter (input)
// iterations (0), or no convergence within max_iter iterations (1).
// The return value will be 2, if the preconditioner is singular.
//
// Upon successful return, output arguments have the following values:
//  
//        x  --  approximate solution to Ax = b
// max_iter  --  the number of iterations performed before the
//               tolerance was reached
//      tol  --  the residual after the final iteration
//  
//*****************************************************************


template <class T>
std::tuple<unsigned int, unsigned int, double>
BiCGSTAB(const ublas::matrix<T> &A, ublas::vector<T> &x, const ublas::vector<T> &b, unsigned int const &max_iter, T const &tol)
{
	T resid;
	ublas::vector<T> rho(max_iter), alpha(max_iter), beta(max_iter), omega(max_iter); // list of scalars
	ublas::vector<T> p, s, t, v;

	T normb = norm_2(b);
	ublas::vector<T> r = b - prod(A, x);
	ublas::vector<T> rtilde = r;

	// if b == 0
	if (normb == 0.0)
		normb = 1;
	
	// if alredy solved
	if ((resid = norm_2(r) / normb) <= tol) {
		return std::tuple<unsigned int, unsigned int, double>(0, 0, resid);
	}

	for (unsigned int i = 1; i < max_iter; i++) {
		rho(i) = inner_prod(rtilde, r);
		if (rho(i) == 0) {
			return std::tuple<unsigned int, unsigned int, double>(2, i, norm_2(r) / normb);;
		}
		if (i == 1)
			p = r;
		else {
			beta(i) = (rho(i)/rho(i-1)) * (alpha(i-1)/omega(i-1));
			p = r + beta(i) * (p - omega(i-1) * v);
		}
		v = prod(A, p);
		alpha(i) = rho(i) / inner_prod(rtilde, v);
		s = r - alpha(i) * v;
		if ((resid = norm_2(s)/normb) < tol) {
			x += alpha(i) * p;
			return std::tuple<unsigned int, unsigned int, double>(0, i, resid);
		}
		t = prod(A, s);
		omega(i) = inner_prod(t, s) / inner_prod(t, t);
		x += alpha(i) * p + omega(i) * s;
		r = s - omega(i) * t;

		if ((resid = norm_2(r) / normb) < tol) {
			return std::tuple<unsigned int, unsigned int, double>(0, i, resid);
		}

		if (omega(i) == 0) {
			return std::tuple<unsigned int, unsigned int, double>(3, i, norm_2(r) / normb);
		}
	}

	return std::tuple<unsigned int, unsigned int, double>(1, max_iter, resid);
}

#endif
