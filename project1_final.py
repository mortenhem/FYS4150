#!/usr/bin/env python

"""
Program for solving the 1D Poisson equation
in n steps from x=0 until x=1 by using forward and backward substitution.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time



n = int(sys.argv[1])	# Number of internal mesh points
h = 1./(n+1)			# Step size






def solve_linear_system_general(RHS, a, b, c):
	
	### unfinished
	
	"""
	Solve Ax=f using by the Thomas Algorithm
	general case.

			
	a, b, c : arrays of diagonal elements
	"""
	
		
	# Internal mesh points
	x = np.linspace(0+h, 1, n, endpoint=False)	

	# Create coefficient matrix A
	A = np.zeros((n, n))

	# Fill matrix with values
	for i in range(n):
	
		A[i, i] = b[i]
		A[i, i-1] = a[i-1]
		A[1, 0] = a[1]
		A[i-1, i] = c[i-1]
		A[n-2, n-1] = c[-2]	

	
	f = h*h*RHS(x)


	b_new = np.zeros(n)		# New main diagonal
	f_new = np.zeros(n)		# New right hand side

	# Forward substitution

	i = 0
	b_new[i] = b[i]

	f_new[i] = f[i]

	u = np.zeros(n)			# Unknowns to be computed
	
	t0 = time.time()
	for i in range(1, n):

		b_new[i] = b[i] - c[i-1]*a[i-1]/b_new[i-1]
		f_new[i] = f[i] - f_new[i-1]*a[i-1]/b_new[i-1]


	# Back substitution

	u[n-1] = f_new[n-1]/b_new[n-1]		# Last value in u


	# Computing unknonws with backward substitution

	for i in reversed(range(1, n-2)):
		u[i-1] = (f_new[i-1] - u[i]*c[i-1])/(b_new[i-1])
	

	
	t1 = time.time()


	CPU_time = t1 - t0

	return u, x, CPU_time



def solve_linear_system(RHS, a_, b_, c_):
	"""
	Solve -u''(x)=f(x) for x in(0,1) in n steps 
	by converting it to a linear system Au=f, and 
	using forward and backward
	substitution.
		
			
	a_ : coefficients just below the main diagonal.
	b_ : coefficients along main diagonal.
	c_ : coefficients above main diagonal.
	
	RHS: Source term

	In this example: a_ == c_
	"""
	
	
	# Values of diagonal below main diagonal
	a = np.array([a_ for i in range(1, n)])
	a[0] = 0

	# Main diagonal
	b = np.array([b_ for i in range(n)])

	# Values above main diagonal
	c = np.array([-1 for i in range(n-1)])
	c[-1] = 0

	
	# Internal mesh points
	x = np.linspace(0+h, 1, n, endpoint=False)	

	# Create coefficient matrix A
	A = np.zeros((n, n))

	# Fill matrix with values
	for i in range(n):
	
		A[i, i] = b[i]
		A[i, i-1] = a[i-1]
		A[1, 0] = a[1]
		A[i-1, i] = c[i-1]
		A[n-2, n-1] = c[-2]	


		
	f = h*h*RHS(x)


	b_new = np.zeros(n)		# New main diagonal
	f_new = np.zeros(n)		# New right hand side

	# Forward substitution

	i = 0
	b_new[i] = b[i]

	f_new[i] = f[i]

	u = np.zeros(n)			# Unknowns to be computed
	
	t0 = time.time()
	for i in range(1, n):

		b_new[i] = b[i] - c[i-1]*a[i-1]/b_new[i-1]
		f_new[i] = f[i] - f_new[i-1]*a[i-1]/b_new[i-1]


	# Back substitution

	u[n-1] = f_new[n-1]/b_new[n-1]		# Last value in u


	# Computing unknonws with backward substitution

	for i in reversed(range(1, n-2)):
		u[i-1] = (f_new[i-1] - u[i]*c[i-1])/(b_new[i-1])
	

	
	# FLOPS: 10N

	t1 = time.time()


	CPU_time = t1 - t0

	return u, x, CPU_time



def plot_and_compare(u, plot=True):
	"""
	Plot exact and numerical solution.
	With source term f(x)=100*exp(-10x), the exact solution
	is u = 1 - (1 - np.exp(-10))*x - np.exp(-10*x).
	"""

	def u_exact(x):
		"""Exact solution of -u''(x)=100exp(-10x)."""
	
		return 1 - (1 - np.exp(-10))*x - np.exp(-10*x)


	x_fine = np.linspace(0, 1, 101)		# Fine mesh
	u_e = u_exact(x_fine)				# Exact solution

	plt.plot(x_fine, u_e, x, u, '--')
	plt.legend(['Exact', 'Numerical'])
	plt.title("Solution of -u''(x)=f(x), n=%g" % n)
	plt.xlabel('x')
	plt.ylabel('u')
	plt.savefig('plotn=%g.png' % n)
	if plot:
		plt.show()



def compute_error():
	"""Compute relative error."""
	

	def source_term(x):
		"""Right hand side of equation -u''(x)=f(x)."""
	
		return 100*np.exp(-10*x)


	
	# Call solver function
	v, x, CPU_time = solve_linear_system(RHS=source_term, a_=-1, b_=2, c_=-1)

	def u_exact(x):
	 
		return 1 - (1 - np.exp(-10))*x - np.exp(-10*x)


	u_e = u_exact(x)

	error = np.zeros(n)

	error_max = []
		
	for i in range(n):
		
		error[i] = np.log(np.abs((v[i] - u_e[i])/u_e[i]))
		
	
	return error_max
	

compute_error()

def LU_solver(n):
	import scipy.linalg
	A = np.zeros((n, n))
		
	for i in range(n):
	
		A[i, i] = 2
		A[i, i-1] = -1
		A[1, 0] = -1
		A[i-1, i] = -1
		A[n-2, n-1] = 2	


	P, L, U = scipy.linalg.lu(A)



	return P, L, U





	








if __name__=='__main__':	

	#os.system('clear')
	
	None
	#RHS = lambda x: 100*np.exp(-10*x)

	#u, x, CPU_time = solve_linear_system(RHS, a_=-1, b_=2, c_=-1)

	#print CPU_time, 'n=%g' % n	
	
	
	#plot_and_compare(u=u, plot=True)

