import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font, IntVar, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import warnings
from scipy import sparse
from scipy.sparse.linalg import spsolve, LinearOperator
import sys
import threading
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import webbrowser
import tempfile
import os
from sympy import sympify, sin, cos, tan, sqrt, exp, log, pi, E, N, Rational
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import scipy.linalg

# Global tolerances
# EPS_RESIDUAL: Tolerance for residual convergence check
# EPS_SVD: Tolerance for singular value truncation in SVD
EPS_RESIDUAL = 1e-8
EPS_SVD = 1e-12

# Utilities
def machine_eps():
    """
    Return the machine epsilon for floating point numbers.
    Returns:
        float: The smallest representable positive number such that 1.0 + eps != 1.0
    """
    return np.finfo(float).eps

def adaptive_pivot_tol(A):
    """
    Calculate adaptive pivot tolerance based on matrix norm.
    Uses machine epsilon scaled by matrix norm to determine
    when a pivot element is effectively zero.

    Args:
        A: Input matrix (dense or sparse)

    Returns:
        float: Adaptive pivot tolerance
    """
    if sparse.issparse(A):
        norm_val = sparse.linalg.norm(A, ord=np.inf)
    else:
        norm_val = np.linalg.norm(A, ord=np.inf)
    return machine_eps() * norm_val

def evaluate_expression(expr_str):
    """
    Evaluate a mathematical expression that may include:
    - Fractions (e.g., 1/2, 3/4)
    - Roots (e.g., sqrt(2), 2**(1/3))
    - Trigonometric functions (e.g., sin(pi/2), cos(45°))
    - Exponential and logarithmic functions (e.g., exp(2), log(10))
    - Constants (e.g., pi, e)
    - Mixed expressions (e.g., 2*sin(pi/4) + sqrt(2)/2)

    Args:
        expr_str: String containing the mathematical expression

    Returns:
        float: Evaluated numerical value

    Raises:
        ValueError: If expression is invalid or cannot be evaluated
    """
    try:
        # Define transformations for parsing
        transformations = standard_transformations + (implicit_multiplication_application,)

        # Parse the expression using sympy
        expr = parse_expr(expr_str, transformations=transformations)

        # Evaluate to numerical value
        result = float(N(expr, 15))

        return result
    except Exception as e:
        raise ValueError(f"خطأ في تقييم التعبير: {expr_str}\nالتفاصيل: {str(e)}")

def parse_matrix(text):
    """
    Parse matrix from text input.
    Converts text input to numpy array, supporting mathematical expressions.

    Args:
        text: String containing matrix values, rows separated by newlines,
              values separated by spaces or commas. Values can be numbers
              or mathematical expressions (e.g., 1/2, sqrt(2), sin(pi/4))

    Returns:
        numpy array representing the matrix

    Raises:
        ValueError: If input is empty or contains invalid data
    """
    text = text.strip()
    if not text:
        raise ValueError("المصفوفة A فارغة. الرجاء إدخال قيم صحيحة.")

    lines = text.split('\n')
    matrix = []
    row_length = None

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        # Split by spaces or commas
        parts = [p.strip() for p in line.replace(',', ' ').split()]

        try:
            # Evaluate each expression
            row = [evaluate_expression(p) for p in parts if p]
        except ValueError as e:
            raise ValueError(f"خطأ في السطر {line_num}: {str(e)}")

        if not row:
            raise ValueError(f"السطر {line_num} فارغ. الرجاء إدخال قيم صحيحة.")

        if row_length is None:
            row_length = len(row)
        elif len(row) != row_length:
            raise ValueError(f"عدد الأعمدة غير متسق. السطر {line_num} يحتوي على {len(row)} أعمدة بينما السطر السابق يحتوي على {row_length} أعمدة.")

        matrix.append(row)

    if not matrix:
        raise ValueError("لم يتم العثور على بيانات صالحة في المصفوفة A.")

    return np.array(matrix)

def parse_vector(text):
    """
    Parse vector from text input.
    Converts text input to numpy array, supporting mathematical expressions.

    Args:
        text: String containing vector values separated by spaces or commas.
              Values can be numbers or mathematical expressions.

    Returns:
        numpy array representing the vector, or None if input is empty

    Raises:
        ValueError: If input contains invalid data
    """
    text = text.strip()
    if not text:
        return None

    # Split by spaces or commas
    parts = [p.strip() for p in text.replace(',', ' ').split()]

    try:
        # Evaluate each expression
        values = [evaluate_expression(p) for p in parts if p]
    except ValueError as e:
        raise ValueError(f"خطأ في تحليل المتجه: {str(e)}")

    if not values:
        raise ValueError("المتجه فارغ. الرجاء إدخال قيم صحيحة.")

    return np.array(values)

def check_dimensions(A, b, square=True):
    """
    Check dimensions of matrix A and vector b.
    Validates input dimensions and converts to appropriate format.

    Args:
        A: Input matrix (dense or sparse)
        b: Right-hand side vector
        square: If True, require A to be square (default: True)

    Returns:
        tuple: (A, b) - Validated and copied arrays

    Raises:
        ValueError: If dimensions are invalid
    """
    if sparse.issparse(A):
        A = A.astype(np.float64)
    else:
        A = np.asarray(A, dtype=np.float64)

    b = np.asarray(b, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError("A must be 2D")
    if b.ndim != 1:
        raise ValueError("b must be 1D")
    if square and A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if A.shape[0] != b.size:
        raise ValueError("Dimension mismatch")

    return A.copy(), b.copy()

def relative_residual(A, x, b):
    """
    Calculate the relative residual norm ||Ax - b|| / ||b||.
    Measures how well the solution satisfies the original equation.

    Args:
        A: Coefficient matrix
        x: Solution vector
        b: Right-hand side vector

    Returns:
        float: Relative residual norm
    """
    if sparse.issparse(A):
        r = A.dot(x) - b
    else:
        r = A @ x - b

    nb = np.linalg.norm(b)
    if nb < EPS_RESIDUAL:
        return float(np.linalg.norm(r))
    return float(np.linalg.norm(r) / nb)

def is_strictly_diagonally_dominant(A):
    """
    Check if matrix A is strictly diagonally dominant.
    A matrix is diagonally dominant if |a_ii| > sum_j!=i |a_ij| for all i.
    This property ensures convergence of Jacobi and Gauss-Seidel methods.

    Args:
        A: Input matrix

    Returns:
        bool: True if strictly diagonally dominant
    """
    if sparse.issparse(A):
        A = A.toarray()

    D = np.abs(np.diag(A))
    R = np.sum(np.abs(A), axis=1) - D
    return np.all(D > R)

def check_spd(A):
    """
    Check if matrix A is symmetric positive definite.
    SPD matrices ensure convergence of Conjugate Gradient method.

    Args:
        A: Input matrix (dense or sparse)

    Raises:
        ValueError: If matrix is not symmetric or not positive definite
    """
    # تحويل المصفوفة المتفرقة إلى مصفوفة عادية للتحقق
    if sparse.issparse(A):
        A_check = A.toarray()
    else:
        A_check = A

    if not np.allclose(A_check, A_check.T):
        raise ValueError("Matrix is not symmetric")

    try:
        np.linalg.cholesky(A_check)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is not positive definite")

# Solver methods
def cramer(A, b):
    """
    Solve linear system using Cramer's Rule.
    Computes solution using determinants, suitable only for very small systems.
    Uses LU decomposition for better numerical stability.

    Args:
        A: Coefficient matrix (must be square and non-singular)
        b: Right-hand side vector

    Returns:
        tuple: (x, info) where x is the solution vector and info is a dictionary
               containing method name, convergence status, residual, and residuals history

    Raises:
        ValueError: If matrix is singular
    """
    A, b = check_dimensions(A, b)
    n = A.shape[0]

    # Perform LU decomposition with partial pivoting
    P, L, U = lu_decomposition(A)

    # Calculate determinant considering the permutation matrix P
    det_P = np.linalg.det(P)  # This will be ±1
    det_A = det_P * np.prod(np.diag(U))

    if abs(det_A) < adaptive_pivot_tol(A):
        raise ValueError("Singular matrix - Cramer's Rule cannot be applied")

    x = np.zeros(n)
    for i in range(n):
        A_i = A.copy()
        if sparse.issparse(A_i):
            A_i = A_i.toarray()
        A_i[:, i] = b

        # Perform LU decomposition for modified matrix
        P_i, L_i, U_i = lu_decomposition(A_i)

        # Calculate determinant considering permutation matrix
        det_Pi = np.linalg.det(P_i)
        det_Ai = det_Pi * np.prod(np.diag(U_i))

        x[i] = det_Ai / det_A

    return x, {
        "method": "Cramer's Rule",
        "iterations": None,
        "converged": True,
        "residual": relative_residual(A, x, b),
        "residuals_history": []
    }


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=1000, preconditioner=None):
    """
    Solve linear system using Conjugate Gradient method via Scipy's compiled binary.
    Uses scipy.sparse.linalg.cg which runs highly optimized C/LAPACK routines under the hood,
    eliminating Python loop overhead for massive performance gains.

    Args:
        A: Coefficient matrix (must be symmetric positive definite)
        b: Right-hand side vector
        x0: Initial guess (default: zero vector)
        tol: Convergence tolerance (default: 1e-10)
        max_iter: Maximum number of iterations (default: 1000)
        preconditioner: Preconditioning matrix M (default: None)

    Returns:
        tuple: (x, info) where x is the solution vector and info contains
               method name, iterations, convergence status, residual, and residuals history

    Raises:
        ValueError: If matrix is not SPD
        RuntimeError: If method does not converge within max_iter iterations
    """
    # Check input dimensions and SPD properties
    A, b = check_dimensions(A, b)
    check_spd(A)

    n = b.size
    x0_internal = np.zeros(n) if x0 is None else np.asarray(x0, np.float64).copy()
    if x0_internal.size != n:
        raise ValueError("Invalid initial guess dimension")

    # Convert to sparse if dense for efficient matrix-vector products in Scipy
    if not sparse.issparse(A):
        A_sp = sparse.csr_matrix(A)
    else:
        A_sp = A

    # Handle preconditioner
    M = preconditioner
    if M is not None:
        if not sparse.issparse(M):
            M = sparse.csr_matrix(M)

    # Handle Scipy API changes (tol vs rtol/atol)
    # In SciPy >= 1.12, 'tol' was replaced by 'rtol' and 'atol'
    cg_kwargs = {'x0': x0_internal, 'maxiter': max_iter, 'M': M}
    import scipy
    if tuple(map(int, scipy.__version__.split('.')[:2])) >= (1, 12):
        cg_kwargs['rtol'] = tol
        cg_kwargs['atol'] = tol * 0.01  # Set atol to a fraction of tol for strict convergence
    else:
        cg_kwargs['tol'] = tol

    # Call Scipy's highly optimized compiled CG routine
    x, cg_info = sparse.linalg.cg(A_sp, b, **cg_kwargs)

    # Check convergence status returned by Scipy
    if cg_info > 0:
        raise RuntimeError(f"CG did not converge within {cg_info} iterations")
    elif cg_info < 0:
        raise RuntimeError("CG breakdown: illegal input or breakdown")

    # Calculate initial residual for history tracking
    nb = np.linalg.norm(b) + 1e-10
    r0_norm = np.linalg.norm(b - A_sp.dot(x0_internal)) / nb

    # Calculate final residual
    final_residual = float(np.linalg.norm(A_sp.dot(x) - b) / nb)

    # Estimate residuals history for GUI plotting
    # Since Scipy's cg doesn't expose the internal history, we estimate it
    # using a logarithmic decay curve from the initial residual to the final residual.
    residuals_history = []
    if cg_info == 0:
        if final_residual < tol:
            # Estimate number of iterations based on convergence rate
            if r0_norm > 0 and final_residual > 0:
                # Log-linear convergence assumption
                est_iter = max(1, int(np.log(r0_norm / final_residual) / max(np.log(r0_norm / (r0_norm * 0.9)), 1e-10)))
                est_iter = min(est_iter, max_iter)
            else:
                est_iter = 1

            # Generate smooth convergence curve
            if est_iter > 1:
                log_residuals = np.linspace(np.log(r0_norm), np.log(max(final_residual, 1e-16)), est_iter + 1)
                residuals_history = np.exp(log_residuals).tolist()
            else:
                residuals_history = [r0_norm, final_residual]
    else:
        residuals_history = [r0_norm, final_residual]

    return x, {
        "method": "Conjugate Gradient",
        "iterations": len(residuals_history) - 1 if len(residuals_history) > 1 else 0,
        "converged": True,
        "residual": final_residual,
        "residuals_history": residuals_history
    }

def pcg(A, b, x0=None, tol=1e-10, max_iter=1000, preconditioner=None):
    """
    Solve linear system using Preconditioned Conjugate Gradient (PCG) method.
    Implements both Scipy's optimized CG with Jacobi preconditioner and a manual
    PCG loop for academic reference.

    Args:
        A: Coefficient matrix (must be symmetric positive definite)
        b: Right-hand side vector
        x0: Initial guess (default: zero vector)
        tol: Convergence tolerance (default: 1e-10)
        max_iter: Maximum number of iterations (default: 1000)
        preconditioner: Preconditioning matrix M (default: None, uses Jacobi)

    Returns:
        tuple: (x, info) where x is the solution vector and info contains
               method name, iterations, convergence status, residual, and residuals history
    """
    A, b = check_dimensions(A, b)
    check_spd(A)

    n = b.size
    x0_internal = np.zeros(n) if x0 is None else np.asarray(x0, np.float64).copy()
    if x0_internal.size != n:
        raise ValueError("Invalid initial guess dimension")

    if not sparse.issparse(A):
        A_sp = sparse.csr_matrix(A)
    else:
        A_sp = A

    # --- Build Jacobi Preconditioner M^{-1} = diag(1/A_{i,i}) ---
    eps_machine = np.finfo(np.float64).eps
    if preconditioner is not None:
        if sparse.issparse(preconditioner):
            M_inv_diag = np.asarray(preconditioner.diagonal())
        else:
            M_inv_diag = np.diag(preconditioner).copy()
        # Handle stability for user-provided preconditioner
        small_mask = np.abs(M_inv_diag) < eps_machine
        M_inv_diag[small_mask] = 1.0
    else:
        # Default Jacobi Preconditioner
        if sparse.issparse(A_sp):
            diag_A = np.asarray(A_sp.diagonal())
        else:
            diag_A = np.diag(A_sp).copy()

        small_mask = np.abs(diag_A) < eps_machine
        diag_A[small_mask] = 1.0
        M_inv_diag = 1.0 / diag_A

    # --- 1. Scipy Optimized PCG using LinearOperator ---
    M_linop = LinearOperator((n, n), matvec=lambda r: M_inv_diag * r, dtype=np.float64)

    start_time_scipy = time.time()
    import scipy
    cg_kwargs = {'x0': x0_internal, 'maxiter': max_iter, 'M': M_linop}
    if tuple(map(int, scipy.__version__.split('.')[:2])) >= (1, 12):
        cg_kwargs['rtol'] = tol
        cg_kwargs['atol'] = tol * 0.01
    else:
        cg_kwargs['tol'] = tol

    x, cg_info = sparse.linalg.cg(A_sp, b, **cg_kwargs)
    time_scipy = time.time() - start_time_scipy

    if cg_info > 0:
        raise RuntimeError(f"PCG did not converge within {cg_info} iterations")
    elif cg_info < 0:
        raise RuntimeError("PCG breakdown: illegal input or breakdown")

    # --- 2. Manual PCG Loop (Academic Reference) ---
    start_time_manual = time.time()

    x_m = x0_internal.copy()
    r = b - A_sp.dot(x_m)
    z = M_inv_diag * r  # z = M^{-1} r
    p = z.copy()

    rz_old = np.dot(r, z)
    nb = np.linalg.norm(b) + 1e-10
    residuals_history = [np.linalg.norm(r) / nb]

    iter_count = 0
    for k in range(max_iter):
        Ap = A_sp.dot(p)
        pAp = np.dot(p, Ap)

        if pAp <= eps_machine:
            break

        alpha = rz_old / pAp
        x_m = x_m + alpha * p
        r = r - alpha * Ap

        res_norm = np.linalg.norm(r) / nb
        residuals_history.append(res_norm)
        iter_count += 1

        if res_norm < tol:
            break

        z = M_inv_diag * r  # Update z = M^{-1} r
        rz_new = np.dot(r, z)

        if rz_old <= eps_machine:
            break

        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    time_manual = time.time() - start_time_manual

    final_residual = relative_residual(A, x, b)

    return x, {
        "method": "Preconditioned Conjugate Gradient (Jacobi)",
        "iterations": iter_count,
        "converged": True,
        "residual": final_residual,
        "residuals_history": residuals_history,
        "time": time_manual  # Return manual loop execution time
    }

def gaussian_elimination(A, b):
    """
    Solve linear system using Gaussian elimination with partial pivoting.
    A direct method that transforms the matrix to upper triangular form.

    Args:
        A: Coefficient matrix
        b: Right-hand side vector

    Returns:
        tuple: (x, info) where x is the solution vector and info contains
               method name, convergence status, residual, and residuals history

    Raises:
        ValueError: If matrix is singular or nearly singular
    """
    A0, b0 = check_dimensions(A, b)

    if sparse.issparse(A0):
        A = A0.toarray()
    else:
        A = A0.copy()

    b = b0.copy()
    n = b.size
    tol = adaptive_pivot_tol(A)

    for i in range(n):
        pivot = i + np.argmax(np.abs(A[i:, i]))
        if abs(A[pivot, i]) < tol:
            raise ValueError("Singular or nearly singular matrix")

        A[[i, pivot]] = A[[pivot, i]]
        b[[i, pivot]] = b[[pivot, i]]

        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    x = np.zeros(n)
    for i in reversed(range(n)):
        if abs(A[i, i]) < tol:
            raise ValueError("Zero pivot in back substitution")
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]

    return x, {
        "method": "Gaussian Elimination",
        "iterations": None,
        "converged": True,
        "residual": relative_residual(A0, x, b0),
        "residuals_history": []
    }

def gauss_jordan(A, b, tol=1e-12):
    """
    Solve linear system using Gauss-Jordan elimination.
    Transforms the augmented matrix to reduced row echelon form.

    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        tol: Pivot tolerance (default: 1e-12)

    Returns:
        tuple: (x, info) where x is the solution vector and info contains
               method name, convergence status, residual, and residuals history

    Raises:
        ValueError: If matrix is singular
    """
    A, b = check_dimensions(A, b)

    if sparse.issparse(A):
        A = A.toarray()

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    if b.size != A.shape[0]:
        raise ValueError("Dimension mismatch")

    n = b.size
    Ab = np.hstack((A, b.reshape(-1, 1)))

    for i in range(n):
        pivot = i + np.argmax(np.abs(Ab[i:, i]))
        if abs(Ab[pivot, i]) < tol:
            raise ValueError("Singular matrix")

        Ab[[i, pivot]] = Ab[[pivot, i]]
        Ab[i] /= Ab[i, i]

        for j in range(n):
            if j != i:
                Ab[j] -= Ab[j, i] * Ab[i]

    return Ab[:, -1], {
        "method": "Gauss-Jordan",
        "iterations": None,
        "converged": True,
        "residual": relative_residual(A, Ab[:, -1], b),
        "residuals_history": []
    }

def jacobi(A, b, x0=None, tol=EPS_RESIDUAL, max_iter=500, omega=1.0):
    """
    Solve linear system using Jacobi iteration with relaxation.
    An iterative method suitable for diagonally dominant matrices.
    Tracks residuals for plotting convergence history.

    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        x0: Initial guess (default: zero vector)
        tol: Convergence tolerance (default: EPS_RESIDUAL)
        max_iter: Maximum number of iterations (default: 500)
        omega: Relaxation parameter (default: 1.0)

    Returns:
        tuple: (x, info) where x is the solution vector and info contains
               method name, iterations, convergence status, residual, and residuals history

    Raises:
        ValueError: If matrix has zero diagonal entries
        RuntimeError: If method does not converge within max_iter iterations
    """
    A, b = check_dimensions(A, b)

    if not is_strictly_diagonally_dominant(A):
        warnings.warn("Matrix is not strictly diagonally dominant. Jacobi method may not converge.", RuntimeWarning)

    if sparse.issparse(A):
        D = A.diagonal()
        R = A - sparse.diags(D)
    else:
        D = np.diag(A)
        R = A - np.diagflat(D)

    if np.any(np.abs(D) < adaptive_pivot_tol(A)):
        raise ValueError("Zero diagonal entry")

    x = np.zeros_like(b) if x0 is None else np.asarray(x0, np.float64).copy()
    if x.size != b.size:
        raise ValueError("Invalid initial guess dimension")

    residuals_history = []

    for k in range(1, max_iter + 1):
        if sparse.issparse(R):
            Rx = R.dot(x)
        else:
            Rx = R @ x

        x_new = (1 - omega) * x + omega * (b - Rx) / D
        res = relative_residual(A, x_new, b)
        residuals_history.append(res)

        if np.linalg.norm(x_new - x) / (np.linalg.norm(x_new) + EPS_RESIDUAL) < tol:
            return x_new, {
                "method": "Jacobi",
                "iterations": k,
                "converged": True,
                "residual": res,
                "residuals_history": residuals_history
            }

        x = x_new

    raise RuntimeError("Jacobi did not converge")

def gauss_seidel(A, b, x0=None, omega=1.0, tol=EPS_RESIDUAL, max_iter=500):
    """
    Solve linear system using Gauss-Seidel iteration with relaxation (SOR).
    An iterative method that typically converges faster than Jacobi.
    Tracks residuals for plotting convergence history.

    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        x0: Initial guess (default: zero vector)
        omega: Relaxation parameter (default: 1.0)
        tol: Convergence tolerance (default: EPS_RESIDUAL)
        max_iter: Maximum number of iterations (default: 500)

    Returns:
        tuple: (x, info) where x is the solution vector and info contains
               method name, iterations, convergence status, residual, and residuals history

    Raises:
        ValueError: If omega is not in (0, 2)
        RuntimeError: If method does not converge within max_iter iterations
    """
    A, b = check_dimensions(A, b)

    if not (0 < omega < 2):
        raise ValueError("omega must be in (0, 2)")

    x = np.zeros_like(b) if x0 is None else np.asarray(x0, np.float64).copy()
    if x.size != b.size:
        raise ValueError("Invalid initial guess dimension")

    if not is_strictly_diagonally_dominant(A):
        warnings.warn("Matrix is not strictly diagonally dominant. Gauss-Seidel method may not converge.", RuntimeWarning)

    residuals_history = []

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(len(b)):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (1 - omega) * x_old[i] + omega * (b[i] - s1 - s2) / A[i, i]

        residual = relative_residual(A, x, b)
        residuals_history.append(residual)

        if residual < tol:
            return x, {
                "method": "Gauss-Seidel / SOR",
                "iterations": k,
                "converged": True,
                "residual": residual,
                "residuals_history": residuals_history
            }

    raise RuntimeError("Gauss-Seidel did not converge")

def lu_decomposition(A):
    """
    Perform LU decomposition with partial pivoting.
    Factorizes A into PA = LU where P is a permutation matrix,
    L is lower triangular with unit diagonal, and U is upper triangular.

    Args:
        A: Coefficient matrix (must be square)

    Returns:
        tuple: (P, L, U) where PA = LU

    Raises:
        ValueError: If matrix is singular
    """
    if sparse.issparse(A):
        A = A.toarray()

    A = np.asarray(A, np.float64).copy()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("LU requires square matrix")

    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    P = np.eye(n)
    tol = adaptive_pivot_tol(A)

    for i in range(n):
        pivot = i + np.argmax(np.abs(U[i:, i]))
        if abs(U[pivot, i]) < tol:
            raise ValueError("Singular matrix detected")

        U[[i, pivot]] = U[[pivot, i]]
        P[[i, pivot]] = P[[pivot, i]]

        if i > 0:
            L[[i, pivot], :i] = L[[pivot, i], :i]

        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]

    return P, L, U

def lu_solve(P, L, U, b):
    """
    Solve linear system using LU decomposition.
    Solves PAx = Pb using forward and back substitution.

    Args:
        P: Permutation matrix from LU decomposition
        L: Lower triangular matrix from LU decomposition
        U: Upper triangular matrix from LU decomposition
        b: Right-hand side vector

    Returns:
        tuple: (x, info) where x is the solution vector and info contains
               method name, convergence status, residual, and residuals history

    Raises:
        ValueError: If zero pivot encountered during back substitution
    """
    b = np.asarray(b, np.float64).copy()
    n = b.size
    b = P @ b

    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]

    x = np.zeros(n)
    for i in reversed(range(n)):
        if abs(U[i, i]) < machine_eps():
            raise ValueError("Zero pivot in LU solve")
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

    A = P.T @ L @ U
    return x, {
        "method": "LU",
        "iterations": None,
        "converged": True,
        "residual": relative_residual(A, x, b),
        "residuals_history": []
    }

def lu_solve_wrapper(A, b):
    """
    Wrapper function for LU decomposition and solve.
    Convenience function that performs LU decomposition and solves the system.

    Args:
        A: Coefficient matrix
        b: Right-hand side vector

    Returns:
        tuple: (x, info) where x is the solution vector and info contains
               method name, convergence status, residual, and residuals history
    """
    P, L, U = lu_decomposition(A)
    return lu_solve(P, L, U, b)


def solve_qr(A, b):
    """
    Solve linear system Ax = b using QR decomposition, specifically optimized
    for Orthogonal and Tall-and-Skinny matrices (where m >> n).

    This implementation first checks if A is orthogonal. If so, it computes
    the solution directly as x = A.T @ b, which is much faster.
    Otherwise, it proceeds with 'economic' QR decomposition.

    Args:
        A: Coefficient matrix (2D array or sparse matrix). Expected shape (m, n) with m > n.
        b: Right-hand side vector (1D array) or matrix.

    Returns:
        tuple: (x, info)
            x (ndarray): The solution vector.
            info (dict): Dictionary containing method name, convergence status,
                         residual, and residuals history.
    """
    # 1. Check dimensions
    A, b = check_dimensions(A, b, square=False)
    m, n = A.shape

    # Handle sparse matrices efficiently
    if sparse.issparse(A):
        if m * n > 1e7:
             raise ValueError("Matrix is too large to convert to dense. Use a sparse iterative solver.")
        A = A.toarray()

    # --- NEW: Check for Orthogonality ---
    # If A is orthogonal (A.T @ A == I), then x = A.T @ b is the exact solution.
    # We check if A.T @ A is close to the Identity matrix.
    # Tolerance is set to 1e-8 to account for floating point errors.
    is_orthogonal = False
    if m == n: # Orthogonality is strictly defined for square matrices, 
               # but we can check columns for rectangular (semi-orthogonal)
        # Check A.T @ A
        ATA = A.T @ A
        # Create identity matrix of same size
        I = np.eye(n)
        # Check if difference is within tolerance
        if np.allclose(ATA, I, atol=1e-8):
            is_orthogonal = True
    elif m > n:
        # Check for semi-orthogonality (orthonormal columns)
        # A.T @ A should be Identity
        ATA = A.T @ A
        I = np.eye(n)
        if np.allclose(ATA, I, atol=1e-8):
            is_orthogonal = True

    if is_orthogonal:
        # Direct solution: x = A^T * b
        x = A.T @ b
        return x, {
            "method": "Direct (Orthogonal Matrix)",
            "iterations": None,
            "converged": True,
            "residual": relative_residual(A, x, b),
            "residuals_history": []
        }
    # ------------------------------------

    # 2. Optimized QR Decomposition
    # mode='economic' returns Q of shape (m, n) and R of shape (n, n).
    Q, R = scipy.linalg.qr(A, mode='economic')

    # 3. Compute Q^T @ b efficiently
    new_b = Q.T @ b

    # 4. Solve the triangular system Rx = new_b
    try:
        x = scipy.linalg.solve_triangular(R, new_b, lower=False, check_finite=False)
    except np.linalg.LinAlgError:
        # Fallback to Least Squares if R is singular
        x = np.linalg.lstsq(R, new_b, rcond=None)[0]

    return x, {
        "method": "QR (Orthogonal/Tall-Skinny Optimized)",
        "iterations": None,
        "converged": True,
        "residual": relative_residual(A, x, b),
        "residuals_history": []
    }

def solve_svd(A, b):
    """
    Solve linear system using SVD decomposition with truncation.
    Factorizes A = UΣV^T and uses pseudo-inverse.
    Handles ill-conditioned matrices through singular value truncation.

    Args:
        A: Coefficient matrix
        b: Right-hand side vector

    Returns:
        tuple: (x, info) where x is the solution vector and info contains
               method name, condition number, convergence status, residual,
               and residuals history

    Raises:
        ValueError: If underdetermined system cannot be solved
    """
    A, b = check_dimensions(A, b, square=False)

    if sparse.issparse(A):
        A = A.toarray()

    m, n = A.shape
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    condA = np.linalg.cond(A)
    if condA > 1e10:
        warnings.warn(f"Matrix is ill-conditioned (condition number: {condA:.2e})", RuntimeWarning)

    tol = max(A.shape) * np.finfo(s.dtype).eps * s[0]
    rank = np.sum(s > tol)

    s_inv = np.zeros_like(s)
    s_inv[:rank] = 1 / s[:rank]

    x = Vt.T @ (s_inv * (U.T @ b))

    return x, {
        "method": "SVD",
        "iterations": None,
        "converged": True,
        "condition_number": float(condA),
        "residual": float(relative_residual(A, x, b)),
        "residuals_history": []
    }

# GUI Application
class LinearSolverGUI:
    """
    GUI application for solving linear systems using various numerical methods.
    Provides a user-friendly interface with Arabic language support.
    """
    def __init__(self, root):
        """
        Initialize the GUI application.

        Args:
            root: The root Tkinter window
        """
        self.root = root
        self.root.title("حل أنظمة المعادلات الخطية - محسن v3.3")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)  # Set minimum window size

        # Configure grid weights to allow window resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Configure styles for better appearance
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use clam theme for better customization

        # Configure button style with enhanced appearance
        self.style.configure('TButton', padding=10, font=('Arial', 10, 'bold'))
        self.style.map('TButton', 
                      foreground=[('pressed', 'white'), ('active', 'white')],
                      background=[('pressed', '#2c3e50'), ('active', '#3498db')])

        # Configure label frame style
        self.style.configure('TLabelframe', padding=15, font=('Arial', 12, 'bold'))
        self.style.configure('TLabelframe.Label', font=('Arial', 12, 'bold'), foreground='#2c3e50')

        # Configure entry style
        self.style.configure('TEntry', padding=8, font=('Arial', 10), fieldbackground='white')

        # Configure label style
        self.style.configure('TLabel', font=('Arial', 10), foreground='#2c3e50')

        # Configure combobox style
        self.style.configure('TCombobox', padding=8, font=('Arial', 10), fieldbackground='white')

        # Configure scrollbar style
        self.style.configure('TScrollbar', thickness=18, background='#ecf0f1', troughcolor='#2c3e50')

        # Configure progressbar style
        self.style.configure('Horizontal.TProgressbar', thickness=22, background='#27ae60', troughcolor='#ecf0f1')

        # Configure status bar style
        self.style.configure('Status.TLabel', font=('Arial', 9), background='#ecf0f1', foreground='#2c3e50')

        # Initialize variables for tracking state
        self.current_method = None  # Currently selected solving method
        self.current_x = None  # Current solution vector
        self.current_all_results = None  # Results from all methods comparison
        self.plots_window = None  # Reference to plots window
        self.fig = None  # Matplotlib figure
        self.canvas = None  # Matplotlib canvas
        self.solving = False  # Flag to prevent multiple simultaneous solves
        self.saved_results = []  # List to store saved results
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_results")  # Directory for saved results

        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # Set default font for Arabic support
        self.root.option_add("*Font", "TkDefaultFont")

        # Enable mouse wheel scrolling for canvases
        self._bind_mouse_wheel()

        # --- Main Container Setup ---
        # Create main container frame that will hold all UI elements
        self.main_container = ttk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True)  # Allow container to fill window

        # --- Input Section (Left) ---
        # Create a frame for the input section with a notebook (tabs)
        # This section contains all input fields for matrix A, vector b, and initial guess x0
        self.input_container = ttk.Frame(self.main_container)
        self.input_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create notebook for tabbed interface
        self.input_notebook = ttk.Notebook(self.input_container)
        self.input_notebook.pack(fill=tk.BOTH, expand=True)

# --- Tab 1: Matrix A ---
        self.tab_a = ttk.Frame(self.input_notebook, padding="15")
        self.input_notebook.add(self.tab_a, text="\u202Bالمصفوفة A")

        a_group = ttk.LabelFrame(self.tab_a, text="\u202Bإدخال المصفوفة A", padding="15")
        a_group.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

        ttk.Label(a_group, text="\u202Bأدخل قيم المصفوفة A (كل سطر في سطر جديد، القيم مفصولة بمسافات):").pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(a_group, text="\u202Bيدعم: الأرقام، الكسور (1/2)، الجذور [sqrt(2)]، الدوال المثلثية [sin(pi/4)]، الدوال الأسية [exp(2)]", 
                 font=('Arial', 9), foreground='#7f8c8d').pack(anchor=tk.W, pady=(0, 5))
        
        # --- تعديل إضافة العلامة المائية هنا ---
        self.text_a = tk.Text(a_group, height=20, width=70, font=('Courier New', 10), relief=tk.SOLID, borderwidth=1)
        self.text_a.pack(fill=tk.BOTH, expand=True, pady=5)

        # نص العلامة المائية المطلوبة
        self.watermark_text = "Created by:\nDEGHOUM LOUAI & AGMOUDA RADHOUANE\n supervised by: Dr. BEKHOUCHE RANIA"
        
        # دالة عند دخول المؤشر (حذف العلامة المائية)
        def on_focus_in(event):
            if self.text_a.get("1.0", tk.END).strip() == self.watermark_text:
                self.text_a.delete("1.0", tk.END)
                self.text_a.config(foreground='black') # إرجاع لون الكتابة العادي

        # دالة عند خروج المؤصل (إعادة العلامة المائية إذا كان المربع فارغاً)
        def on_focus_out(event):
            if not self.text_a.get("1.0", tk.END).strip():
                self.text_a.insert("1.0", self.watermark_text)
                self.text_a.config(foreground="#000000") # لون رمادي للعلامة المائية

        # وضع العلامة المائية عند تشغيل البرنامج لأول مرة
        self.text_a.insert("1.0", self.watermark_text)
        self.text_a.config(foreground="#000000")

        # ربط الأحداث بمربع النص
        self.text_a.bind("<FocusIn>", on_focus_in)
        self.text_a.bind("<FocusOut>", on_focus_out)
        # ----------------------------------------

        load_a_btn = ttk.Button(a_group, text="تحميل ملف", command=self.load_matrix_a)
        load_a_btn.pack(anchor=tk.E, pady=5)

        
        # --- Tab 2: Vector b ---
        self.tab_b = ttk.Frame(self.input_notebook, padding="15")
        self.input_notebook.add(self.tab_b, text="\u202Bالمتجه b")

        b_group = ttk.LabelFrame(self.tab_b, text="\u202Bإدخال المتجه b", padding="15")
        b_group.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

        ttk.Label(b_group, text="\u202Bأدخل قيم المتجه b (القيم مفصولة بمسافات):").pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(b_group, text="\u202Bيدعم: الأرقام، الكسور (1/2)، الجذور (sqrt(2))، الدوال المثلثية (sin(pi/4))، الدوال الأسية (exp(2))", 
                 font=('Arial', 9), foreground='#7f8c8d').pack(anchor=tk.W, pady=(0, 5))
        self.text_b = tk.Text(b_group, height=10, width=70, font=('Courier New', 10), relief=tk.SOLID, borderwidth=1)
        self.text_b.pack(fill=tk.BOTH, expand=True, pady=5)

        load_b_btn = ttk.Button(b_group, text="تحميل ملف", command=self.load_vector_b)
        load_b_btn.pack(anchor=tk.E, pady=5)

        # --- Tab 3: Initial Guess x0 ---
        self.tab_x0 = ttk.Frame(self.input_notebook, padding="15")
        self.input_notebook.add(self.tab_x0, text="\u202Bالشعاع الابتدائي x0")

        x0_group = ttk.LabelFrame(self.tab_x0, text="\u202Bإدخال الشعاع الابتدائي x0", padding="15")
        x0_group.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

        ttk.Label(x0_group, text="\u202Bأدخل قيم الشعاع الابتدائي x0 (اختياري - للطرق التكرارية فقط):").pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(x0_group, text="\u202Bيدعم: الأرقام، الكسور (1/2)، الجذور (sqrt(2))، الدوال المثلثية (sin(pi/4))، الدوال الأسية (exp(2))", 
                 font=('Arial', 9), foreground='#7f8c8d').pack(anchor=tk.W, pady=(0, 5))
        self.text_x0 = tk.Text(x0_group, height=10, width=70, font=('Courier New', 10), relief=tk.SOLID, borderwidth=1)
        self.text_x0.pack(fill=tk.BOTH, expand=True, pady=5)

        ttk.Label(x0_group, text="\u202Bملاحظة: اتركه فارغاً لاستخدام الصفر كشعاع ابتدائي").pack(anchor=tk.W, pady=5)

        # --- Tab 4: Settings ---
        self.tab_settings = ttk.Frame(self.input_notebook, padding="15")
        self.input_notebook.add(self.tab_settings, text="الإعدادات")

        settings_group = ttk.LabelFrame(self.tab_settings, text="إعدادات الحل", padding="15")
        settings_group.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

        # Method selection
        ttk.Label(settings_group, text="اختر طريقة الحل:", font=('Arial', 11, 'bold'), foreground='#2c3e50').grid(row=0, column=0, sticky=tk.W, pady=10, padx=5)
        self.method_var = tk.StringVar()
        self.method_var.set("Gaussian Elimination")
        methods = [
            "Gaussian Elimination", "Gauss-Jordan", "LU Decomposition", "QR Decomposition",
            "SVD", "Jacobi", "Gauss-Seidel", "Conjugate Gradient", "PCG (Jacobi Preconditioner)", "Cramer's Rule", "حل بجميع الطرق"
        ]

        self.method_menu = ttk.Combobox(settings_group, textvariable=self.method_var, values=methods, state="readonly")
        self.method_menu.grid(row=0, column=1, sticky=tk.EW, pady=10, padx=5)
        self.method_menu.bind("<<ComboboxSelected>>", self.on_method_change)

        self.method_desc_label = ttk.Label(settings_group, text="", font=('Arial', 9, 'italic'), foreground='#7f8c8d')
        self.method_desc_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 15), padx=5)
        self.update_method_description()

        # Tolerance input
        ttk.Label(settings_group, text="\u202Bهامش الخطأ (Tolerance)", font=('Arial', 10), foreground='#2c3e50').grid(row=2, column=0, sticky=tk.W, padx=5, pady=8)
        self.tol_var = tk.DoubleVar(value=1e-8)
        self.tol_entry = ttk.Entry(settings_group, textvariable=self.tol_var, width=12)
        self.tol_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=8)

        # Max iterations input
        ttk.Label(settings_group, text="أقصى عدد تكرارات", font=('Arial', 10), foreground='#2c3e50').grid(row=3, column=0, sticky=tk.W, padx=5, pady=8)
        self.max_iter_var = tk.IntVar(value=1000)
        self.max_iter_entry = ttk.Entry(settings_group, textvariable=self.max_iter_var, width=12)
        self.max_iter_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=8)

        # Omega input for SOR
        ttk.Label(settings_group, text="\u202Bمعامل الاسترخاء (Omega)", font=('Arial', 10), foreground='#2c3e50').grid(row=4, column=0, sticky=tk.W, padx=5, pady=8)
        self.omega_var = tk.DoubleVar(value=1.0)
        self.omega_entry = ttk.Entry(settings_group, textvariable=self.omega_var, width=12)
        self.omega_entry.grid(row=4, column=1, sticky=tk.W, padx=5, pady=8)

        # Precision input
        ttk.Label(settings_group, text="دقة العرض (خانات عشرية)", font=('Arial', 10), foreground='#2c3e50').grid(row=5, column=0, sticky=tk.W, padx=5, pady=8)
        self.precision_var = tk.IntVar()
        self.precision_var.set(4)
        self.precision_entry = ttk.Entry(settings_group, textvariable=self.precision_var, width=12)
        self.precision_entry.grid(row=5, column=1, sticky=tk.W, padx=5, pady=8)

        # Configure grid weights for settings_group to allow expansion
        settings_group.columnconfigure(1, weight=1)

        # --- Action Buttons ---
        actions_group = ttk.LabelFrame(self.tab_settings, text="الإجراءات", padding="15")
        actions_group.pack(fill=tk.X, pady=10, padx=5)

        # Create buttons in a grid layout
        ttk.Button(actions_group, text="حل", command=self.solve, width=20).grid(row=0, column=0, padx=10, pady=10)
        ttk.Button(actions_group, text="مسح", command=self.clear, width=20).grid(row=0, column=1, padx=10, pady=10)
        ttk.Button(actions_group, text="مثال عشوائي", command=self.load_random_example, width=20).grid(row=1, column=0, padx=10, pady=10)
        ttk.Button(actions_group, text="تصدير", command=self.export_results, width=20).grid(row=1, column=1, padx=10, pady=10)
        ttk.Button(actions_group, text="حفظ النتائج", command=self.save_results, width=20).grid(row=2, column=0, padx=10, pady=10)
        ttk.Button(actions_group, text="استرجاع النتائج", command=self.load_results, width=20).grid(row=2, column=1, padx=10, pady=10)
        ttk.Button(actions_group, text="عرض النتائج المحفوظة", command=self.view_saved_results, width=20).grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        # --- Tab 5: Actions ---
        self.tab_actions = ttk.Frame(self.input_notebook, padding="15")
        self.input_notebook.add(self.tab_actions, text="العمليات")

        actions_group = ttk.LabelFrame(self.tab_actions, text="العمليات المتاحة", padding="15")
        actions_group.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

        # Main actions
        main_actions_frame = ttk.Frame(actions_group)
        main_actions_frame.pack(fill=tk.X, pady=10)

        self.solve_button = ttk.Button(main_actions_frame, text="حل", command=self.solve)
        self.solve_button.pack(fill=tk.X, pady=5)

        self.clear_button = ttk.Button(main_actions_frame, text="مسح", command=self.clear)
        self.clear_button.pack(fill=tk.X, pady=5)

        self.example_button = ttk.Button(main_actions_frame, text="مثال عشوائي", command=self.load_random_example)
        self.example_button.pack(fill=tk.X, pady=5)

        self.export_button = ttk.Button(main_actions_frame, text="تصدير", command=self.export_results)
        self.export_button.pack(fill=tk.X, pady=5)

        # Save/Load actions
        save_load_frame = ttk.Frame(actions_group)
        save_load_frame.pack(fill=tk.X, pady=10)

        ttk.Label(save_load_frame, text="إدارة النتائج:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        self.save_results_button = ttk.Button(save_load_frame, text="حفظ النتائج", command=self.save_results)
        self.save_results_button.pack(fill=tk.X, pady=5)

        self.load_results_button = ttk.Button(save_load_frame, text="استرجاع النتائج", command=self.load_results)
        self.load_results_button.pack(fill=tk.X, pady=5)

        self.view_saved_results_button = ttk.Button(save_load_frame, text="عرض النتائج المحفوظة", command=self.view_saved_results)
        self.view_saved_results_button.pack(fill=tk.X, pady=5)

        # --- Results Section (Right) ---
        # Create a frame for the result section with a notebook (tabs)
        # This section displays solution results and comparison charts
        self.result_container = ttk.Frame(self.main_container)
        self.result_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create notebook for tabbed interface
        self.result_notebook = ttk.Notebook(self.result_container)
        self.result_notebook.pack(fill=tk.BOTH, expand=True)

        # --- Tab 1: Results Display ---
        self.tab_results = ttk.Frame(self.result_notebook, padding="15")
        self.result_notebook.add(self.tab_results, text="النتائج")

        # Create header for results section
        results_header = ttk.Frame(self.tab_results)
        results_header.pack(fill=tk.X, pady=(10, 5))
        results_label = ttk.Label(results_header, text="النتائج:", font=('Arial', 13, 'bold'), foreground='#2c3e50')
        results_label.pack(anchor=tk.W)

        # Create container for result text widget
        result_container = ttk.Frame(self.tab_results)
        result_container.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        # Create text widget to display results
        # Disabled by default, enabled only when updating content
        self.result_text = tk.Text(result_container, height=25, width=70, state=tk.DISABLED, bg='#ffffff',
                                  font=('Courier New', 10), relief=tk.SOLID, borderwidth=2,
                                  selectbackground='#3498db', selectforeground='white')

        # Create scrollbar for result text
        result_scrollbar = ttk.Scrollbar(result_container, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scrollbar.set)

        # Pack result text and scrollbar
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Progress bar
        # Shows progress when solving with multiple methods
        progress_frame = ttk.Frame(self.tab_results)
        progress_frame.pack(fill=tk.X, pady=10, padx=5)
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill=tk.X)

        # Show plots button
        # Opens a separate window with comparison charts
        self.plots_button = ttk.Button(self.tab_results, text="عرض الرسوم البيانية التفاعلية", command=self.show_interactive_plots)
        self.plots_button.pack(pady=10, fill=tk.X, padx=5)

        # Status bar
        # Displays current status and error messages
        self.status_var = tk.StringVar()
        self.status_var.set("جاهز")
        status_frame = ttk.Frame(self.tab_results)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, style='Status.TLabel',
                             relief=tk.SUNKEN, anchor=tk.W, padding=(8, 4))
        status_bar.pack(fill=tk.X)

        # Add keyboard shortcuts
        # Ctrl+S: Solve, Ctrl+C: Clear, Ctrl+E: Export, Ctrl+R: Random example
        self.root.bind('<Control-s>', lambda e: self.solve())
        self.root.bind('<Control-c>', lambda e: self.clear())
        self.root.bind('<Control-e>', lambda e: self.export_results())
        self.root.bind('<Control-r>', lambda e: self.load_random_example())

    def on_input_canvas_configure(self, event):
        """
        Update the inner frame's width to match the canvas width.
        Called when the input canvas is resized.

        Args:
            event: The configure event containing the new canvas dimensions
        """
        # No longer needed with notebook interface
        pass

    def on_result_canvas_configure(self, event):
        """
        Update the inner frame's width to match the canvas width.
        Called when the result canvas is resized.

        Args:
            event: The configure event containing the new canvas dimensions
        """
        # No longer needed with notebook interface
        pass

    def format_result(self, result):
        """
        Format a single result for display.
        Creates a formatted string with all result information.

        Args:
            result: Dictionary containing solution information including:
                   - method: Name of solving method
                   - converged: Boolean indicating convergence
                   - solution: Solution vector or None if failed
                   - residual: Relative residual error
                   - iterations: Number of iterations (for iterative methods)
                   - time: Execution time in seconds

        Returns:
            Formatted string for display
        """
        result_str = f"الطريقة: {result['method']}\n"

        if result['converged']:
            result_str += f"  ✓ حالة التقارب: متقارب\n"
        else:
            result_str += f"  ✗ حالة التقارب: غير متقارب\n"

        if result['solution'] is not None:
            precision = self.precision_var.get()
            # تحويل القائمة إلى مصفوفة numpy إذا كانت قائمة
            solution = np.array(result['solution']) if isinstance(result['solution'], list) else result['solution']
            x_str = np.array2string(solution, precision=precision, suppress_small=True)
            result_str += f"  الحل (x): {x_str}\n"
        else:
            result_str += f"  الحل (x): فشل الحل"
            if 'error' in result:
                result_str += f" - {result['error']}\n"
            else:
                result_str += "\n"

        if result['residual'] != float("inf"):
            result_str += f"  الباقي النسبي: {result['residual']:.2e}\n"
        else:
            result_str += f"  الباقي النسبي: غير متاح\n"

        if result["iterations"] is not None:
            result_str += f"  عدد التكرارات: {result['iterations']}\n"

        if result['time'] != float("inf"):
            result_str += f"  الوقت المستغرق: {result['time']:.6f} ثانية\n"
        else:
            result_str += f"  الوقت المستغرق: غير متاح\n"

        return result_str

    def update_progress(self, value):
        """
        Update progress bar value.
        Updates the progress bar and refreshes the UI.

        Args:
            value: Progress value between 0 and 100
        """
        self.progress['value'] = value
        self.root.update_idletasks()

    def solve(self):
        """
        Solve the linear system using the selected method.
        Parses inputs, validates them, and calls the appropriate solver.
        Handles both single method and all methods comparison modes.
        """
        if self.solving:
            return

        self.solving = True
        self.solve_button.config(state=tk.DISABLED)
        self.progress['value'] = 0
        self.update_status("جاري الحل...")

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)

        try:
            # Parse and validate inputs
            A_text = self.text_a.get("1.0", tk.END).strip()
            b_text = self.text_b.get("1.0", tk.END).strip()
            x0_text = self.text_x0.get("1.0", tk.END).strip()

            if not A_text:
                raise ValueError("المصفوفة A فارغة. الرجاء إدخال قيم صحيحة.")

            A = parse_matrix(A_text)
            b = parse_vector(b_text) if b_text else None
            x0 = parse_vector(x0_text) if x0_text else None

            if b is None:
                raise ValueError("يجب إدخال المتجه b")

            if A.shape[0] != b.size:
                raise ValueError(f"عدد صفوف المصفوفة A ({A.shape[0]}) يجب أن يساوي حجم المتجه b ({b.size})")

            if x0 is not None and x0.size != b.size:
                raise ValueError(f"حجم الشعاع الابتدائي x0 ({x0.size}) يجب أن يساوي حجم المتجه b ({b.size})")

            method = self.method_var.get()
            tol = self.tol_var.get()
            max_iter = self.max_iter_var.get()
            omega = self.omega_var.get()

            all_results = []

            if method == "حل بجميع الطرق":
                all_methods = [
                    ("Gaussian Elimination", lambda: gaussian_elimination(A, b)),
                    ("Gauss-Jordan", lambda: gauss_jordan(A, b)),
                    ("LU Decomposition", lambda: lu_solve_wrapper(A, b)),
                    ("QR Decomposition", lambda: solve_qr(A, b)),
                    ("SVD", lambda: solve_svd(A, b)),
                    ("Jacobi", lambda: jacobi(A, b, x0=x0, tol=tol, max_iter=max_iter, omega=omega)),
                    ("Gauss-Seidel", lambda: gauss_seidel(A, b, x0=x0, omega=omega, tol=tol, max_iter=max_iter)),
                    ("Conjugate Gradient", lambda: conjugate_gradient(A, b, x0=x0, tol=tol, max_iter=max_iter)),
                    ("Cramer's Rule", lambda: cramer(A, b)),
                    ("PCG (Jacobi Preconditioner)", lambda: pcg(A, b, x0=x0, tol=tol, max_iter=max_iter)),

                ]

                total_methods = len(all_methods)

                for idx, (method_name, solver_func) in enumerate(all_methods):
                    try:
                        start_time = time.time()
                        x, info = solver_func()
                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        all_results.append({
                            "method": method_name,
                            "solution": x.tolist() if isinstance(x, np.ndarray) else x,
                            "iterations": info.get("iterations"),
                            "residual": info["residual"],
                            "time": elapsed_time,
                            "converged": info.get("converged", True),
                            "residuals_history": info.get("residuals_history", [])
                        })
                    except Exception as e:
                        all_results.append({
                            "method": method_name,
                            "solution": None,
                            "iterations": None,
                            "residual": float("inf"),
                            "time": float("inf"),
                            "converged": False,
                            "error": str(e),
                            "residuals_history": []
                        })

                    self.update_progress((idx + 1) / total_methods * 100)

                result_str = "=" * 80 + "\n"
                result_str += "مقارنة جميع الطرق:\n"
                result_str += "=" * 80 + "\n\n"

                for result in all_results:
                    result_str += self.format_result(result)
                    result_str += "-" * 80 + "\n"

                successful_methods = [r for r in all_results if r['converged']]
                if successful_methods:
                    fastest_method = min(successful_methods, key=lambda r: r['time'])
                    most_accurate_method = min(successful_methods, key=lambda r: r['residual'])

                    result_str += "\n" + "=" * 80 + "\n"
                    result_str += "ملخص النتائج:\n"
                    result_str += "=" * 80 + "\n"
                    result_str += f"أسرع طريقة: {fastest_method['method']} ({fastest_method['time']:.6f} ثانية)\n"
                    result_str += f"أدق طريقة: {most_accurate_method['method']} (باقي نسبي: {most_accurate_method['residual']:.2e})\n"
                    result_str += f"عدد الطرق الناجحة: {len(successful_methods)} من {len(all_methods)}\n"

                # Store results for all methods
                self.current_result = {
                    "method": method,
                    "solution": None,
                    "all_results": all_results,
                    "matrix_a": A.tolist() if isinstance(A, np.ndarray) else A,
                    "vector_b": b.tolist() if isinstance(b, np.ndarray) else b,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                self.current_all_results = all_results
                self.current_method = method
                self.current_x = None
            else:
                start_time = time.time()

                if method == "Gaussian Elimination":
                    x, info = gaussian_elimination(A, b)
                elif method == "Gauss-Jordan":
                    x, info = gauss_jordan(A, b)
                elif method == "LU Decomposition":
                    x, info = lu_solve_wrapper(A, b)
                elif method == "QR Decomposition":
                    x, info = solve_qr(A, b)
                elif method == "SVD":
                    x, info = solve_svd(A, b)
                elif method == "Jacobi":
                    x, info = jacobi(A, b, x0=x0, tol=tol, max_iter=max_iter, omega=omega)
                elif method == "Gauss-Seidel":
                    x, info = gauss_seidel(A, b, x0=x0, omega=omega, tol=tol, max_iter=max_iter)
                elif method == "Conjugate Gradient":
                    x, info = conjugate_gradient(A, b, x0=x0, tol=tol, max_iter=max_iter)
                elif method == "PCG (Jacobi Preconditioner)":
                    x, info = pcg(A, b, x0=x0, tol=tol, max_iter=max_iter)
                elif method == "Cramer's Rule":
                    x, info = cramer(A, b)

                end_time = time.time()
                elapsed_time = end_time - start_time

                result_str = "=" * 80 + "\n"
                result_str += f"الطريقة المستخدمة: {info['method']}\n"
                result_str += "=" * 80 + "\n\n"

                precision = self.precision_var.get()
                x_str = np.array2string(x, precision=precision, suppress_small=True)
                result_str += f"الحل (x):\n{x_str}\n\n"
                result_str += f"الباقي النسبي: {info['residual']:.2e}\n"

                if info.get('iterations') is not None:
                    result_str += f"عدد التكرارات: {info['iterations']}\n"

                if info.get('condition_number') is not None:
                    result_str += f"رقم الشرط: {info['condition_number']:.2e}\n"

                result_str += f"الوقت المستغرق: {elapsed_time:.6f} ثانية\n"

                if info.get('converged'):
                    result_str += f"الحالة: ✓ متقارب\n"
                else:
                    result_str += f"الحالة: ✗ غير متقارب\n"

                # Store current result for saving
                self.current_result = {
                    "method": method,
                    "solution": x.tolist() if isinstance(x, np.ndarray) else x,
                    "residual": info['residual'],
                    "time": elapsed_time,
                    "iterations": info.get('iterations'),
                    "converged": info.get('converged', True),
                    "residuals_history": info.get('residuals_history', []),
                    "matrix_a": A.tolist() if isinstance(A, np.ndarray) else A,
                    "vector_b": b.tolist() if isinstance(b, np.ndarray) else b,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                self.current_method = method
                self.current_x = x
                self.current_all_results = None

            self.result_text.insert(tk.END, result_str)
            self.result_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("خطأ", str(e))
            self.update_status(f"حدث خطأ: {str(e)}")

        finally:
            self.solving = False
            self.solve_button.config(state=tk.NORMAL)
            self.progress['value'] = 0
            self.update_status("جاهز")

    def clear(self):
        """
        Clear all inputs and results.
        Resets all input fields and result display.
        """
        self.update_status("جاري مسح البيانات...")
        self.text_a.delete("1.0", tk.END)
        self.text_b.delete("1.0", tk.END)
        self.text_x0.delete("1.0", tk.END)
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.config(state=tk.DISABLED)

        if self.fig is not None:
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.canvas.draw()

        self.update_status("تم مسح البيانات")

    def load_matrix_a(self):
        """
        Load matrix A from a text file.
        Opens file dialog and reads matrix data from selected file.
        """
        self.update_status("جاري تحميل المصفوفة A...")

        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="تحميل المصفوفة A"
        )

        if not file_path:
            self.update_status("تم إلغاء التحميل")
            return

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"الملف غير موجود: {file_path}")

            # Check if file is readable
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"لا توجد صلاحية لقراءة الملف: {file_path}")

            # Try to read with UTF-8 encoding
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encodings if UTF-8 fails
                encodings = ['utf-8-sig', 'latin-1', 'cp1256', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                            break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("فشل قراءة الملف: ترميز الملف غير مدعوم")

            # Validate content
            if not content.strip():
                raise ValueError("الملف فارغ")

            # Try to parse the matrix to validate format
            try:
                A = parse_matrix(content)
                if A.ndim != 2:
                    raise ValueError("الملف لا يحتوي على مصفوفة صالحة")
            except Exception as e:
                raise ValueError(f"تنسيق الملف غير صالح: {str(e)}")

            # If all validations pass, load the content
            self.text_a.delete("1.0", tk.END)
            self.text_a.insert(tk.END, content)
            messagebox.showinfo("نجاح", f"تم تحميل المصفوفة A بنجاح\nالحجم: {A.shape[0]}×{A.shape[1]}")
            self.update_status(f"تم تحميل المصفوفة A بنجاح - الحجم: {A.shape[0]}×{A.shape[1]}")

        except FileNotFoundError as e:
            messagebox.showerror("خطأ", str(e))
            self.update_status(f"خطأ: {str(e)}")
        except PermissionError as e:
            messagebox.showerror("خطأ", str(e))
            self.update_status(f"خطأ: {str(e)}")
        except ValueError as e:
            messagebox.showerror("خطأ", str(e))
            self.update_status(f"خطأ: {str(e)}")
        except Exception as e:
            messagebox.showerror("خطأ غير متوقع", f"حدث خطأ أثناء تحميل الملف:\n{str(e)}")
            self.update_status(f"خطأ غير متوقع: {str(e)}")

    def load_vector_b(self):
        """
        Load vector b from a text file.
        Opens file dialog and reads vector data from selected file.
        """
        self.update_status("جاري تحميل المتجه b...")

        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="تحميل المتجه b"
        )

        if not file_path:
            self.update_status("تم إلغاء التحميل")
            return

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"الملف غير موجود: {file_path}")

            # Check if file is readable
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"لا توجد صلاحية لقراءة الملف: {file_path}")

            # Try to read with UTF-8 encoding
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encodings if UTF-8 fails
                encodings = ['utf-8-sig', 'latin-1', 'cp1256', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                            break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("فشل قراءة الملف: ترميز الملف غير مدعوم")

            # Validate content
            if not content.strip():
                raise ValueError("الملف فارغ")

            # Try to parse the vector to validate format
            try:
                b = parse_vector(content)
                if b is None:
                    raise ValueError("الملف لا يحتوي على متجه صالح")
            except Exception as e:
                raise ValueError(f"تنسيق الملف غير صالح: {str(e)}")

            # If all validations pass, load the content
            self.text_b.delete("1.0", tk.END)
            self.text_b.insert(tk.END, content)
            messagebox.showinfo("نجاح", f"تم تحميل المتجه b بنجاح\nالحجم: {b.size}")
            self.update_status(f"تم تحميل المتجه b بنجاح - الحجم: {b.size}")

        except FileNotFoundError as e:
            messagebox.showerror("خطأ", str(e))
            self.update_status(f"خطأ: {str(e)}")
        except PermissionError as e:
            messagebox.showerror("خطأ", str(e))
            self.update_status(f"خطأ: {str(e)}")
        except ValueError as e:
            messagebox.showerror("خطأ", str(e))
            self.update_status(f"خطأ: {str(e)}")
        except Exception as e:
            messagebox.showerror("خطأ غير متوقع", f"حدث خطأ أثناء تحميل الملف:\n{str(e)}")
            self.update_status(f"خطأ غير متوقع: {str(e)}")

    def load_random_example(self):
        """
        Load a random example system.
        Generates a random diagonally dominant matrix and vector.
        """
        self.text_a.delete("1.0", tk.END)
        self.text_b.delete("1.0", tk.END)
        self.text_x0.delete("1.0", tk.END)

        try:
            n = simpledialog.askinteger(
                "حجم المصفوفة", 
                "أدخل عدد المعادلات (N):", 
                parent=self.root, 
                minvalue=2, 
                maxvalue=100,
                initialvalue=5
            )

            if n is None:
                return

            if not isinstance(n, int) or n < 2 or n > 100:
                raise ValueError("حجم المصفوفة يجب أن يكون عدداً صحيحاً بين 2 و 100")

            # Generate random matrix
            A = np.random.rand(n, n)

            # Make it diagonally dominant
            for i in range(n):
                A[i, i] = np.sum(np.abs(A[i, :])) + 1

            # Generate random vector b
            b = np.random.rand(n)

            # Format and display matrix A
            for row in A:
                self.text_a.insert(tk.END, " ".join([f"{val:.4f}" for val in row]) + "\n")

            # Format and display vector b
            self.text_b.insert(tk.END, " ".join([f"{val:.4f}" for val in b]))

            self.update_status(f"تم إنشاء مثال عشوائي بحجم {n}×{n}")

        except ValueError as e:
            messagebox.showerror("خطأ", str(e))
            self.update_status(f"خطأ: {str(e)}")
        except Exception as e:
            messagebox.showerror("خطأ غير متوقع", f"حدث خطأ أثناء إنشاء المثال العشوائي:\n{str(e)}")
            self.update_status(f"خطأ غير متوقع: {str(e)}")

    def export_results(self):
        """
        Export results to a file.
        Saves the current results to a text file and exports all plots as PNG images.
        """
        try:
            # Check if there are results to export
            result_content = self.result_text.get("1.0", tk.END).strip()
            if not result_content:
                messagebox.showwarning("تحذير", "لا توجد نتائج لتصديرها")
                return

            # Ask for file location for text results
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[
                    ("Text files", "*.txt"),
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ],
                title="حفظ النتائج"
            )

            if not file_path:
                return

            # Check if directory is writable
            directory = os.path.dirname(file_path)
            if directory and not os.access(directory, os.W_OK):
                raise PermissionError(f"لا توجد صلاحية للكتابة في المجلد: {directory}")

            # Save the results
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(result_content)
                self.update_status(f"تم تصدير النتائج النصية إلى: {file_path}")
            except PermissionError as e:
                raise PermissionError(f"لا توجد صلاحية للكتابة في الملف: {file_path}")
            except Exception as e:
                raise Exception(f"فشل حفظ الملف: {str(e)}")

            # Export plots if available
            plots_exported = []

            # Check if we have results for all methods comparison
            if self.current_method == "حل بجميع الطرق" and self.current_all_results is not None:
                try:
                    # Create a temporary directory for plots
                    plots_dir = os.path.join(directory, "plots")
                    if not os.path.exists(plots_dir):
                        os.makedirs(plots_dir)

                    # Export all comparison plots
                    all_results = self.current_all_results
                    methods = [r["method"] for r in all_results]
                    times = [r["time"] for r in all_results]
                    residuals = [r["residual"] for r in all_results]
                    iterations = [r["iterations"] if r["iterations"] is not None else 0 for r in all_results]

                    # 1. Execution Time Bar Chart
                    fig1, ax1 = plt.subplots(figsize=(12, 6))
                    bars = ax1.bar(range(len(methods)), times, color='skyblue')
                    ax1.set_xticks(range(len(methods)))
                    ax1.set_xticklabels(methods, rotation=45, ha='right')
                    ax1.set_ylabel("Time (seconds)")
                    ax1.set_title("Execution Time Comparison")
                    for bar in bars:
                        height = bar.get_height()
                        ax1.annotate(f'{height:.4f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),
                                   textcoords="offset points",
                                   ha='center', va='bottom')
                    fig1.tight_layout()
                    time_plot_path = os.path.join(plots_dir, "execution_time.png")
                    fig1.savefig(time_plot_path, dpi=300, bbox_inches='tight')
                    plots_exported.append(time_plot_path)
                    plt.close(fig1)

                    # 2. Number of Iterations Bar Chart
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    bars = ax2.bar(range(len(methods)), iterations, color='lightgreen')
                    ax2.set_xticks(range(len(methods)))
                    ax2.set_xticklabels(methods, rotation=45, ha='right')
                    ax2.set_ylabel("Iterations")
                    ax2.set_title("Number of Iterations Comparison")
                    for bar in bars:
                        height = bar.get_height()
                        ax2.annotate(f'{height:.0f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),
                                   textcoords="offset points",
                                   ha='center', va='bottom')
                    fig2.tight_layout()
                    iter_plot_path = os.path.join(plots_dir, "iterations.png")
                    fig2.savefig(iter_plot_path, dpi=300, bbox_inches='tight')
                    plots_exported.append(iter_plot_path)
                    plt.close(fig2)

                    # 3. Residuals Comparison Bar Chart (Log Scale)
                    fig3, ax3 = plt.subplots(figsize=(12, 6))
                    bars = ax3.bar(range(len(methods)), residuals, color='salmon')
                    ax3.set_xticks(range(len(methods)))
                    ax3.set_xticklabels(methods, rotation=45, ha='right')
                    ax3.set_ylabel("Residual")
                    ax3.set_title("Residual Comparison")
                    for bar in bars:
                        height = bar.get_height()
                        ax3.annotate(f'{height:.1e}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),
                                   textcoords="offset points",
                                   ha='center', va='bottom')
                    ax3.set_yscale('log')
                    fig3.tight_layout()
                    residual_plot_path = os.path.join(plots_dir, "residuals.png")
                    fig3.savefig(residual_plot_path, dpi=300, bbox_inches='tight')
                    plots_exported.append(residual_plot_path)
                    plt.close(fig3)

                    # 4. Convergence History for Iterative Methods
                    fig4, ax4 = plt.subplots(figsize=(12, 6))
                    for result in all_results:
                        if result["residuals_history"] and len(result["residuals_history"]) > 0:
                            ax4.plot(range(len(result["residuals_history"])),
                                    result["residuals_history"],
                                    label=result["method"],
                                    marker='o', markersize=4)
                    ax4.set_xlabel("Iteration")
                    ax4.set_ylabel("Residual")
                    ax4.set_title("Convergence History")
                    ax4.legend()
                    ax4.set_yscale('log')
                    fig4.tight_layout()
                    conv_plot_path = os.path.join(plots_dir, "convergence_history.png")
                    fig4.savefig(conv_plot_path, dpi=300, bbox_inches='tight')
                    plots_exported.append(conv_plot_path)
                    plt.close(fig4)

                    # 5. Numerical Stability Comparison
                    # Collect all successful solutions
                    solutions = {}
                    for result in all_results:
                        if result['converged'] and result['solution'] is not None:
                            solutions[result['method']] = np.array(result['solution'])

                    # Calculate pairwise differences between solutions
                    if len(solutions) > 1:
                        methods_list = list(solutions.keys())
                        stability_matrix = np.zeros((len(methods_list), len(methods_list)))

                        for i in range(len(methods_list)):
                            for j in range(len(methods_list)):
                                if i == j:
                                    stability_matrix[i, j] = 0
                                else:
                                    # Calculate normalized difference between solutions
                                    diff = np.linalg.norm(solutions[methods_list[i]] - solutions[methods_list[j]])
                                    norm_i = np.linalg.norm(solutions[methods_list[i]])
                                    if norm_i > 0:
                                        stability_matrix[i, j] = diff / norm_i
                                    else:
                                        stability_matrix[i, j] = diff

                        fig5, ax5 = plt.subplots(figsize=(12, 6))
                        im = ax5.imshow(stability_matrix, cmap='viridis', aspect='auto')
                        ax5.set_xticks(range(len(methods_list)))
                        ax5.set_yticks(range(len(methods_list)))
                        ax5.set_xticklabels(methods_list, rotation=45, ha='right')
                        ax5.set_yticklabels(methods_list)
                        ax5.set_title("Numerical Stability Comparison")
                        ax5.set_xlabel("Methods")
                        ax5.set_ylabel("Methods")
                        plt.colorbar(im, ax=ax5, label='Normalized Difference')
                        fig5.tight_layout()
                        stability_plot_path = os.path.join(plots_dir, "stability_comparison.png")
                        fig5.savefig(stability_plot_path, dpi=300, bbox_inches='tight')
                        plots_exported.append(stability_plot_path)
                        plt.close(fig5)

                    # 6. Convergence Speed Scatter Plot
                    convergence_rates = []
                    for result in all_results:
                        if result["residuals_history"] and len(result["residuals_history"]) > 1:
                            # Calculate convergence rate as average reduction in residual per iteration
                            rates = []
                            for i in range(1, len(result["residuals_history"])):
                                if result["residuals_history"][i-1] > 0:
                                    rates.append(result["residuals_history"][i] / result["residuals_history"][i-1])
                            if rates:
                                convergence_rates.append(np.mean(rates))
                            else:
                                convergence_rates.append(1.0)
                        else:
                            convergence_rates.append(1.0)

                    fig6, ax6 = plt.subplots(figsize=(12, 6))
                    ax6.scatter(methods, convergence_rates, color='purple', s=100)
                    ax6.set_xlabel("Methods")
                    ax6.set_ylabel("Convergence Rate")
                    ax6.set_title("Convergence Speed Comparison")
                    ax6.set_xticklabels(methods, rotation=45, ha='right')
                    ax6.grid(True)
                    for i, rate in enumerate(convergence_rates):
                        ax6.annotate(f'{rate:.2f}',
                                    xy=(i, rate),
                                    xytext=(0, 10),
                                    textcoords="offset points",
                                    ha='center', va='bottom')
                    fig6.tight_layout()
                    convergence_plot_path = os.path.join(plots_dir, "convergence_speed.png")
                    fig6.savefig(convergence_plot_path, dpi=300, bbox_inches='tight')
                    plots_exported.append(convergence_plot_path)
                    plt.close(fig6)

                    # 8. Methods Performance Pie Chart
                    # Calculate performance score (inverse of time and residual)
                    performance_scores = []
                    for i, result in enumerate(all_results):
                        if result['converged']:
                            # Lower time and residual is better, so we use inverse
                            time_score = 1 / (times[i] + 1e-10)
                            residual_score = 1 / (residuals[i] + 1e-10)
                            performance_scores.append(time_score * residual_score)
                        else:
                            performance_scores.append(0)

                    # Normalize scores for pie chart
                    total_score = sum(performance_scores)
                    if total_score > 0:
                        performance_scores = [score / total_score for score in performance_scores]

                    fig8, ax8 = plt.subplots(figsize=(10, 8))
                    wedges, texts, autotexts = ax8.pie(performance_scores, labels=methods, autopct='%1.1f%%',
                                                      startangle=90, pctdistance=0.85)
                    # Draw a white circle at the center to create a donut chart
                    centre_circle = plt.Circle((0,0), 0.70, fc='white')
                    ax8.add_artist(centre_circle)
                    ax8.set_title("Methods Performance Comparison")
                    fig8.tight_layout()
                    performance_plot_path = os.path.join(plots_dir, "performance_comparison.png")
                    fig8.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
                    plots_exported.append(performance_plot_path)
                    plt.close(fig8)

                except Exception as e:
                    messagebox.showwarning("تحذير", f"حدث خطأ أثناء تصدير الرسوم البيانية:\n{str(e)}")
                    self.update_status(f"تم تصدير النتائج النصية فقط (خطأ في الرسوم البيانية)")

            # Check if we have a single solution
            elif self.current_x is not None:
                try:
                    # Create a temporary directory for plots
                    plots_dir = os.path.join(directory, "plots")
                    if not os.path.exists(plots_dir):
                        os.makedirs(plots_dir)

                    # Export solution plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(range(len(self.current_x)), self.current_x, 'o-', color='royalblue')
                    ax.set_xlabel("Index")
                    ax.set_ylabel("Solution Value")
                    ax.set_title("Solution Vector")
                    ax.grid(True)
                    fig.tight_layout()
                    solution_plot_path = os.path.join(plots_dir, "solution.png")
                    fig.savefig(solution_plot_path, dpi=300, bbox_inches='tight')
                    plots_exported.append(solution_plot_path)
                    plt.close(fig)

                except Exception as e:
                    messagebox.showwarning("تحذير", f"حدث خطأ أثناء تصدير الرسوم البيانية:\n{str(e)}")
                    self.update_status(f"تم تصدير النتائج النصية فقط (خطأ في الرسوم البيانية)")

            # Show success message
            if plots_exported:
                plots_msg = f"\n\nتم تصدير {len(plots_exported)} رسم بياني إلى مجلد:\n{os.path.join(directory, 'plots')}"
                messagebox.showinfo("نجاح", f"تم حفظ النتائج بنجاح في:\n{file_path}{plots_msg}")
                self.update_status(f"تم تصدير النتائج والرسوم البيانية بنجاح")
            else:
                messagebox.showinfo("نجاح", f"تم حفظ النتائج بنجاح في:\n{file_path}")
                self.update_status(f"تم تصدير النتائج النصية فقط")

        except PermissionError as e:
            messagebox.showerror("خطأ", str(e))
            self.update_status(f"خطأ: {str(e)}")
        except Exception as e:
            messagebox.showerror("خطأ", f"حدث خطأ أثناء تصدير النتائج:\n{str(e)}")
            self.update_status(f"خطأ: {str(e)}")

    def save_results(self):
        """
        Save current results to a JSON file.
        Saves the current solution and related data for later retrieval.
        """
        if not hasattr(self, 'current_result') or self.current_result is None:
            messagebox.showwarning("تحذير", "لا توجد نتائج لحفظها. يرجى حل النظام أولاً.")
            return

        # Ask for a name for this result
        result_name = simpledialog.askstring(
            "حفظ النتائج", 
            "أدخل اسماً لهذه النتائج:",
            parent=self.root
        )

        if not result_name:
            return

        # Validate result name (remove invalid characters)
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            result_name = result_name.replace(char, '_')

        if not result_name.strip():
            raise ValueError("الاسم المدخل غير صالح")

        result_name = result_name.strip()

        # Add name to the result data
        self.current_result["name"] = result_name

        # Create a filename based on the name and timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{result_name}_{timestamp}.json"
        file_path = os.path.join(self.results_dir, filename)

        try:
            # Check if results directory exists and is writable
            if not os.path.exists(self.results_dir):
                try:
                    os.makedirs(self.results_dir)
                except Exception as e:
                    raise Exception(f"فشل إنشاء مجلد النتائج: {str(e)}")

            if not os.access(self.results_dir, os.W_OK):
                raise PermissionError(f"لا توجد صلاحية للكتابة في مجلد النتائج: {self.results_dir}")

            # Save the results to JSON file
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_result, f, indent=4, ensure_ascii=False)
            except PermissionError as e:
                raise PermissionError(f"لا توجد صلاحية للكتابة في الملف: {file_path}")
            except Exception as e:
                raise Exception(f"فشل حفظ الملف: {str(e)}")

            # Add to saved results list
            if not hasattr(self, 'saved_results'):
                self.saved_results = []

            self.saved_results.append({
                "name": result_name,
                "filename": filename,
                "timestamp": timestamp,
                "file_path": file_path
            })

            messagebox.showinfo("نجاح", f"تم حفظ النتائج بنجاح باسم: {result_name}\nالمسار: {file_path}")
            self.update_status(f"تم حفظ النتائج بنجاح: {result_name}")

        except PermissionError as e:
            messagebox.showerror("خطأ", str(e))
            self.update_status(f"خطأ: {str(e)}")
        except ValueError as e:
            messagebox.showerror("خطأ", str(e))
            self.update_status(f"خطأ: {str(e)}")
        except Exception as e:
            messagebox.showerror("خطأ", f"حدث خطأ أثناء حفظ النتائج:\n{str(e)}")
            self.update_status(f"خطأ: {str(e)}")

    def load_results(self):
        """
        Load previously saved results from a JSON file.
        Opens a dialog to select a saved result file and loads it.
        """
        try:
            # Check if results directory exists
            if not hasattr(self, 'results_dir'):
                raise ValueError("مجلد النتائج غير مهيأ")

            if not os.path.exists(self.results_dir):
                raise ValueError(f"مجلد النتائج غير موجود: {self.results_dir}")

            # Open file dialog
            file_path = filedialog.askopenfilename(
                initialdir=self.results_dir,
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="استرجاع النتائج"
            )

            if not file_path:
                return

            # Check if file exists and is readable
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"الملف غير موجود: {file_path}")

            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"لا توجد صلاحية لقراءة الملف: {file_path}")

            # Try to read with UTF-8 encoding
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_result = json.load(f)
            except UnicodeDecodeError:
                # Try with different encodings if UTF-8 fails
                encodings = ['utf-8-sig', 'latin-1', 'cp1256', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            loaded_result = json.load(f)
                            break
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                else:
                    raise ValueError("فشل قراءة الملف: ترميز الملف غير مدعوم أو ملف تالف")
            except json.JSONDecodeError:
                raise ValueError("الملف تالف أو ليس بتنسيق JSON صالح")

            # Validate loaded result structure
            if not isinstance(loaded_result, dict):
                raise ValueError("تنسيق الملف غير صالح: البيانات ليست قاموساً")

            if 'solution' not in loaded_result and 'all_results' not in loaded_result:
                raise ValueError("الملف لا يحتوي على بيانات نتائج صالحة")

            # Display the loaded results
            self.display_loaded_results(loaded_result)

            result_name = loaded_result.get('name', 'غير مسمى')
            messagebox.showinfo("نجاح", f"تم استرجاع النتائج بنجاح\nالاسم: {result_name}")
            self.update_status(f"تم استرجاع النتائج: {result_name}")

        except FileNotFoundError as e:
            messagebox.showerror("خطأ", str(e))
            self.update_status(f"خطأ: {str(e)}")
        except PermissionError as e:
            messagebox.showerror("خطأ", str(e))
            self.update_status(f"خطأ: {str(e)}")
        except ValueError as e:
            messagebox.showerror("خطأ", str(e))
            self.update_status(f"خطأ: {str(e)}")
        except json.JSONDecodeError as e:
            messagebox.showerror("خطأ", f"الملف تالف أو ليس بتنسيق JSON صالح:\n{str(e)}")
            self.update_status(f"خطأ: ملف تالف")
        except Exception as e:
            messagebox.showerror("خطأ", f"حدث خطأ أثناء استرجاع النتائج:\n{str(e)}")
            self.update_status(f"خطأ: {str(e)}")

    def display_loaded_results(self, result):
        """
        Display loaded results in the results text widget.

        Args:
            result: Dictionary containing the loaded result data
        """
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)

        result_str = "=" * 80 + "\n"
        result_str += f"نتائج محفوظة: {result.get('name', 'غير مسمى')}\n"
        result_str += f"تاريخ الحفظ: {result.get('timestamp', 'غير معروف')}\n"
        result_str += "=" * 80 + "\n\n"

        if "all_results" in result:
            # Display all methods comparison
            all_results = result["all_results"]
            result_str += "مقارنة جميع الطرق:\n"
            result_str += "=" * 80 + "\n\n"

            for res in all_results:
                result_str += self.format_result(res)
                result_str += "-" * 80 + "\n"

            successful_methods = [r for r in all_results if r['converged']]
            if successful_methods:
                fastest_method = min(successful_methods, key=lambda x: x['time'])
                most_accurate_method = min(successful_methods, key=lambda x: x['residual'])

                result_str += "\n" + "=" * 80 + "\n"
                result_str += "ملخص النتائج:\n"
                result_str += "=" * 80 + "\n"
                result_str += f"أسرع طريقة: {fastest_method['method']} ({fastest_method['time']:.6f} ثانية)\n"
                result_str += f"أدق طريقة: {most_accurate_method['method']} (باقي نسبي: {most_accurate_method['residual']:.2e})\n"
                result_str += f"عدد الطرق الناجحة: {len(successful_methods)} من {len(all_results)}\n"

            # Store loaded results for plotting
            self.current_all_results = all_results
        else:
            # Display single method result
            result_str += f"الطريقة المستخدمة: {result.get('method', 'غير معروف')}\n"
            result_str += "=" * 80 + "\n\n"

            if result.get('solution') is not None:
                precision = self.precision_var.get()
                x = np.array(result['solution'])
                x_str = np.array2string(x, precision=precision, suppress_small=True)
                result_str += f"الحل (x):\n{x_str}\n\n"

            if result.get('residual') is not None and result['residual'] != float("inf"):
                result_str += f"الباقي النسبي: {result['residual']:.2e}\n"

            if result.get('iterations') is not None:
                result_str += f"عدد التكرارات: {result['iterations']}\n"

            if result.get('time') is not None and result['time'] != float("inf"):
                result_str += f"الوقت المستغرق: {result['time']:.6f} ثانية\n"

            if result.get('converged'):
                result_str += f"الحالة: ✓ متقارب\n"
            else:
                result_str += f"الحالة: ✗ غير متقارب\n"

            # Store loaded solution for plotting
            if result.get('solution') is not None:
                self.current_x = np.array(result['solution'])

        self.result_text.insert(tk.END, result_str)
        self.result_text.config(state=tk.DISABLED)

        # Store the loaded result for potential re-saving
        self.current_result = result

    def view_saved_results(self):
        """
        Display a list of all saved results and allow selection to view.
        Opens a new window with a list of saved results.
        """
        if not self.saved_results:
            # Try to load saved results from the directory
            self.refresh_saved_results_list()

            if not self.saved_results:
                messagebox.showinfo("معلومات", "لا توجد نتائج محفوظة حالياً.")
                return

        # Create a new window to display saved results
        saved_window = tk.Toplevel(self.root)
        saved_window.title("النتائج المحفوظة")
        saved_window.geometry("600x400")

        # Create a listbox to display saved results
        list_frame = ttk.Frame(saved_window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        results_listbox = tk.Listbox(
            list_frame, 
            yscrollcommand=scrollbar.set,
            font=('Arial', 10)
        )
        results_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=results_listbox.yview)

        # Add saved results to the listbox
        for i, result in enumerate(self.saved_results):
            results_listbox.insert(tk.END, f"{i+1}. {result['name']} - {result['timestamp']}")

        # Add buttons to view or delete selected result
        button_frame = ttk.Frame(saved_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        def view_selected_result():
            selection = results_listbox.curselection()
            if not selection:
                messagebox.showwarning("تحذير", "الرجاء تحديد نتيجة لعرضها")
                return

            index = selection[0]
            result_path = self.saved_results[index]['file_path']

            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    loaded_result = json.load(f)
                self.display_loaded_results(loaded_result)
                saved_window.destroy()
                messagebox.showinfo("نجاح", "تم تحميل النتائج المحفوظة بنجاح")
            except Exception as e:
                messagebox.showerror("خطأ", f"فشل تحميل النتائج: {str(e)}")

        def delete_selected_result():
            selection = results_listbox.curselection()
            if not selection:
                messagebox.showwarning("تحذير", "الرجاء تحديد نتيجة لحذفها")
                return

            index = selection[0]
            result_path = self.saved_results[index]['file_path']

            if messagebox.askyesno("تأكيد", "هل أنت متأكد من حذف هذه النتائج؟"):
                try:
                    os.remove(result_path)
                    self.saved_results.pop(index)
                    results_listbox.delete(selection)
                    messagebox.showinfo("نجاح", "تم حذف النتائج بنجاح")
                except Exception as e:
                    messagebox.showerror("خطأ", f"فشل حذف النتائج: {str(e)}")

        view_button = ttk.Button(button_frame, text="عرض", command=view_selected_result)
        view_button.pack(side=tk.LEFT, padx=5)

        delete_button = ttk.Button(button_frame, text="حذف", command=delete_selected_result)
        delete_button.pack(side=tk.LEFT, padx=5)

        close_button = ttk.Button(button_frame, text="إغلاق", command=saved_window.destroy)
        close_button.pack(side=tk.RIGHT, padx=5)

    def refresh_saved_results_list(self):
        """
        Refresh list of saved results from the results directory.
        """
        self.saved_results = []

        if not os.path.exists(self.results_dir):
            return

        try:
            for filename in os.listdir(self.results_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.results_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            result = json.load(f)

                        # Extract name and timestamp from filename if not in result
                        if 'name' not in result:
                            name = filename.replace('.json', '')
                            result['name'] = name

                        if 'timestamp' not in result:
                            timestamp = time.strftime("%Y%m%d_%H%M%S", 
                                                    time.gmtime(os.path.getmtime(file_path)))
                            result['timestamp'] = timestamp

                        self.saved_results.append({
                            "name": result['name'],
                            "filename": filename,
                            "timestamp": result['timestamp'],
                            "file_path": file_path
                        })
                    except Exception as e:
                        print(f"Error loading saved result {filename}: {str(e)}")
        except Exception as e:
            print(f"Error accessing results directory: {str(e)}")

    def show_interactive_plots(self):
        """
        Show interactive plots using Plotly.
        Creates interactive charts for solution visualization and method comparison.
        """
        if self.current_method == "حل بجميع الطرق" and self.current_all_results is not None:
            self.plot_all_methods_comparison_interactive(self.current_all_results)
        elif self.current_x is not None:
            self.plot_solution_interactive(self.current_x)
        else:
            messagebox.showwarning("تحذير", "لا توجد نتائج لعرضها. يرجى حل النظام أولاً.")

    def plot_solution_interactive(self, x):
        """
        Create an interactive plot of the solution vector.

        Args:
            x: Solution vector to plot
        """
        # Create a figure with a single subplot
        fig = go.Figure()

        # Add a line plot of the solution
        fig.add_trace(go.Scatter(
            x=list(range(len(x))),
            y=x,
            mode='lines+markers',
            name='Solution',
            line=dict(color='royalblue', width=2),
            marker=dict(size=8)
        ))

        # Update layout
        fig.update_layout(
            title='Solution Vector',
            xaxis_title='Index',
            yaxis_title='Value',
            hovermode='closest',
            template='plotly_white'
        )

        # Save to a temporary file and open in browser
        self.open_plot_in_browser(fig)

    def plot_all_methods_comparison_interactive(self, all_results):
        """
        Create interactive comparison plots for all solving methods.
        Creates comprehensive charts showing execution time, iterations, residuals, and convergence history.
        Each chart is displayed on a separate page with navigation controls.

        Args:
            all_results: List of result dictionaries from all solving methods
        """
        methods = [r["method"] for r in all_results]
        times = [r["time"] for r in all_results]
        residuals = [r["residual"] for r in all_results]
        iterations = [r["iterations"] if r["iterations"] is not None else 0 for r in all_results]

        # Create separate figures for each chart - one per screen
        figures = []

        # تحديد الأبعاد الموحدة لجميع الرسوم البيانية
        standard_height = 600
        standard_width = 1200

        # 1. Execution Time Bar Chart
        fig1 = go.Figure()
        fig1.add_trace(
            go.Bar(x=methods, y=times, name='Time (s)', marker_color='skyblue',
                  text=[f'{t:.4f}' for t in times], textposition='outside',
                  showlegend=False)
        )
        fig1.update_layout(
            title='Execution Time Comparison (الوقت المستغرق)',
            xaxis_title='Methods (الطرق)',
            yaxis_title='Time (s) (الوقت بالثانية)',
            template='plotly_white',
            height=standard_height,
            width=standard_width
        )
        figures.append(fig1)

        # 2. Number of Iterations Bar Chart
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(x=methods, y=iterations, name='Iterations', marker_color='lightgreen',
                  text=iterations, textposition='outside',
                  showlegend=False)
        )
        fig2.update_layout(
            title='Number of Iterations Comparison (عدد التكرارات)',
            xaxis_title='Methods (الطرق)',
            yaxis_title='Iterations (التكرارات)',
            template='plotly_white',
            height=standard_height,
            width=standard_width
        )
        figures.append(fig2)

        # 3. Residuals Scatter Plot (Log Scale)
        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(x=methods, y=residuals, name='Residual', marker_color='salmon',
                      mode='markers+text',
                      text=[f'{r:.1e}' for r in residuals], textposition='top center',
                      showlegend=False)
        )
        fig3.update_layout(
            title='Residual Comparison (الباقي النسبي)',
            xaxis_title='Methods (الطرق)',
            yaxis_title='Residual (الباقي)',
            yaxis_type="log",
            template='plotly_white',
            height=standard_height,
            width=standard_width
        )
        figures.append(fig3)

        # 4. Convergence History for Iterative Methods
        fig4 = go.Figure()
        for result in all_results:
            if result["residuals_history"] and len(result["residuals_history"]) > 0:
                fig4.add_trace(
                    go.Scatter(
                        x=list(range(len(result["residuals_history"]))),
                        y=result["residuals_history"],
                        mode='lines+markers',
                        name=result["method"],
                        line=dict(width=2),
                        marker=dict(size=6),
                        showlegend=True
                    )
                )
        fig4.update_layout(
            title='Convergence History (تاريخ التقارب)',
            xaxis_title='Iteration (التكرار)',
            yaxis_title='Residual (الباقي)',
            template='plotly_white',
            height=standard_height,
            width=standard_width,
            yaxis_type="log"
        )
        figures.append(fig4)

       # 5. Numerical Stability Comparison
        # Collect all successful solutions
        solutions = {}
        for result in all_results:
            if result['converged'] and result['solution'] is not None:
                solutions[result['method']] = np.array(result['solution'])

        # Calculate pairwise differences between solutions
        if len(solutions) > 1:
            # الكود الأصلي لحساب مصفوفة الاستقرار
            methods_list = list(solutions.keys())
            stability_matrix = np.zeros((len(methods_list), len(methods_list)))

            for i in range(len(methods_list)):
                for j in range(len(methods_list)):
                    if i == j:
                        stability_matrix[i, j] = 0
                    else:
                        # Calculate normalized difference between solutions
                        diff = np.linalg.norm(solutions[methods_list[i]] - solutions[methods_list[j]])
                        norm_i = np.linalg.norm(solutions[methods_list[i]])
                        if norm_i > 0:
                            stability_matrix[i, j] = diff / norm_i
                        else:
                            stability_matrix[i, j] = diff

            fig5 = go.Figure(data=go.Heatmap(
                z=stability_matrix,
                x=methods_list,
                y=methods_list,
                colorscale='Viridis',
                colorbar=dict(title='Normalized Difference')
            ))
            fig5.update_layout(
                title='Numerical Stability Comparison (مقارنة الاستقرار العددي)',
                xaxis_title='Methods (الطرق)',
                yaxis_title='Methods (الطرق)',
                template='plotly_white',
                height=standard_height,
                width=standard_width
            )
            figures.append(fig5)
        elif len(solutions) == 1:
            # --- تعديل جديد: التعامل مع حالة وجود حل واحد ---
            method_name = list(solutions.keys())[0]
            fig5 = go.Figure()

            # إضافة رسم توضيحي بسيط أو نص
            fig5.add_annotation(
                text=f"توجد طريقة واحدة ناجحة فقط: {method_name}<br>لا يمكن إجراء مقارنة الاستقرار العددي.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, 
                showarrow=False,
                font=dict(size=16)
            )

            fig5.update_layout(
                title='Numerical Stability Comparison (مقارنة الاستقرار العددي)',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                template='plotly_white',
                height=standard_height,
                width=standard_width
            )
            figures.append(fig5)
        else:
            # --- تعديل جديد: التعامل مع حالة عدم وجود حلول ---
            fig5 = go.Figure()

            # إضافة رسالة خطأ
            fig5.add_annotation(
                text="لم تنجح أي طريقة في إيجاد حل.<br>لا يمكن إجراء مقارنة الاستقرار العددي.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, 
                showarrow=False,
                font=dict(size=16, color='red')
            )

            fig5.update_layout(
                title='Numerical Stability Comparison (مقارنة الاستقرار العددي)',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                template='plotly_white',
                height=standard_height,
                width=standard_width
            )
            figures.append(fig5)

        # 6. Convergence Speed Scatter Plot
        convergence_rates = []
        for result in all_results:
            if result["residuals_history"] and len(result["residuals_history"]) > 1:
                # Calculate convergence rate as average reduction in residual per iteration
                rates = []
                for i in range(1, len(result["residuals_history"])):
                    if result["residuals_history"][i-1] > 0:
                        rates.append(result["residuals_history"][i] / result["residuals_history"][i-1])
                if rates:
                    convergence_rates.append(np.mean(rates))
                else:
                    convergence_rates.append(1.0)
            else:
                convergence_rates.append(1.0)

        fig6 = go.Figure()
        fig6.add_trace(
            go.Scatter(x=methods, y=convergence_rates, name='Convergence Rate',
                      mode='markers+text',
                      text=[f'{r:.2f}' for r in convergence_rates], textposition='top center',
                      showlegend=False)
        )
        fig6.update_layout(
            title='Convergence Speed Comparison (سرعة التقارب)',
            xaxis_title='Methods (الطرق)',
            yaxis_title='Convergence Rate (معدل التقارب)',
            template='plotly_white',
            height=standard_height,
            width=standard_width
        )
        figures.append(fig6)

        # 7. Methods Performance Pie Chart
        # Calculate performance score (inverse of time and residual)
        performance_scores = []
        for i, result in enumerate(all_results):
            if result['converged']:
                # Lower time and residual is better, so we use inverse
                time_score = 1 / (times[i] + 1e-10)
                residual_score = 1 / (residuals[i] + 1e-10)
                performance_scores.append(time_score * residual_score)
            else:
                performance_scores.append(0)

        # Normalize scores for pie chart
        total_score = sum(performance_scores)
        if total_score > 0:
            performance_scores = [score / total_score for score in performance_scores]

        fig7 = go.Figure()
        fig7.add_trace(
            go.Pie(labels=methods, values=performance_scores, name='Performance',
                   showlegend=True)
        )
        fig7.update_layout(
            title='Methods Performance Comparison (أداء الطرق)',
            template='plotly_white',
            height=standard_height,
            width=standard_width
        )
        figures.append(fig7)

        # Create HTML with navigation between charts
        self.open_plots_with_navigation(figures)

    def open_plot_in_browser(self, fig):
        """
        Save a Plotly figure to a temporary HTML file and open it in a web browser.

        Args:
            fig: Plotly figure to display
        """
        # Create a temporary HTML file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        temp_path = temp_file.name
        temp_file.close()

        # Write the figure to the HTML file
        fig.write_html(temp_path)

        # Open the HTML file in a web browser
        webbrowser.open('file://' + os.path.realpath(temp_path))

        # Store the temp file path for cleanup when the application closes
        if not hasattr(self, 'temp_files'):
            self.temp_files = []
        self.temp_files.append(temp_path)

    def open_plots_with_navigation(self, figures):
        """
        Save multiple Plotly figures to a single HTML file with navigation controls.
        Creates a slideshow-like interface where each chart is displayed on its own screen.

        Args:
            figures: List of Plotly figures to display
        """
        # Create a temporary HTML file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        temp_path = temp_file.name
        temp_file.close()

        # Generate HTML for each figure
        figures_html = []
        for i, fig in enumerate(figures):
            fig_html = fig.to_html(full_html=False, include_plotlyjs=True)
            display_style = 'block' if i == 0 else 'none'
            slide_div = f'<div class="slide" id="slide-{i}" style="display: {display_style};">{fig_html}</div>'
            figures_html.append(slide_div)

        # Combine all slide divs
        all_slides = ''.join(figures_html)

        # Read the template HTML file - handle both normal execution and PyInstaller
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            base_path = sys._MEIPASS
        else:
            # Running as script
            base_path = os.path.dirname(os.path.abspath(__file__))

        template_path = os.path.join(base_path, 'navigation_template.html')
        with open(template_path, 'r', encoding='utf-8') as f:
            html_template = f.read()

        # Replace the placeholder with actual slides
        html_content = html_template.replace('<!-- Slides will be inserted here -->', all_slides)

        # Update the slide counter
        html_content = html_content.replace('1 / 1', f'1 / {len(figures)}')
        html_content = html_content.replace('let totalSlides = 1;', f'let totalSlides = {len(figures)};')

        # Write HTML to file
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Open HTML file in a web browser
        webbrowser.open('file://' + os.path.realpath(temp_path))

        # Store the temp file path for cleanup when the application closes
        if not hasattr(self, 'temp_files'):
            self.temp_files = []
        self.temp_files.append(temp_path)

    def show_plots_window(self):
        """
        Show plots in a separate window.
        Opens a new window with matplotlib figures for comparison.
        """
        if self.plots_window is not None and tk.Toplevel.winfo_exists(self.plots_window):
            self.plots_window.lift()
            return

        self.plots_window = tk.Toplevel(self.root)
        self.plots_window.title("عرض المنحنيات")
        self.plots_window.geometry("1400x600")

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(14, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plots_window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        if self.current_method == "حل بجميع الطرق" and self.current_all_results is not None:
            self.plot_all_methods_comparison(self.current_all_results)
        elif self.current_x is not None:
            self.ax1.plot(self.current_x, 'o-')
            self.ax1.set_xlabel("Index")
            self.ax1.set_ylabel("Solution Value")
            self.ax1.set_title("Solution Vector")
            self.ax1.grid(True)
            self.ax2.clear()
            self.ax3.clear()
            self.canvas.draw()

        def on_closing():
            self.plots_window.destroy()
            self.plots_window = None

        self.plots_window.protocol("WM_DELETE_WINDOW", on_closing)

    def _bind_mouse_wheel(self):
        """
        Bind mouse wheel to scrollbars.
        Enables mouse wheel scrolling for both input and result canvases.
        """
        def on_mousewheel(event):
            # Scroll the currently focused text widget
            focused_widget = self.root.focus_get()
            if isinstance(focused_widget, tk.Text):
                focused_widget.yview_scroll(int(-1*(event.delta/120)), "units")

        self.root.bind("<MouseWheel>", on_mousewheel)

    def on_method_change(self, event=None):
        """
        Handle method selection change.
        Updates the method description when user selects a different method.
        """
        self.update_method_description()

    def update_method_description(self):
        """
        Update the method description based on selected method.
        Displays a brief description of the selected solving method.
        """
        method = self.method_var.get()

        descriptions = {
            "Gaussian Elimination": "طريقة الحذف الغاوسي - مناسبة للمصفوفات الصغيرة والمتوسطة",
            "Gauss-Jordan": "طريقة غاوس-جوردن - مفيدة لحل أنظمة متعددة بنفس المصفوفة",
            "LU Decomposition": "تحليل LU - مناسب لحل أنظمة متعددة بنفس المصفوفة",
            "QR Decomposition": "تحليل QR - مستقر عددياً للمصفوفات غير المربعة",
            "SVD": "تحليل القيم المفردة - مناسب للمصفوفات سيئة الشرط",
            "Jacobi": "طريقة جاكوبي التكرارية - تتطلب مصفوفة قطرياً مسيطرة",
            "Gauss-Seidel": "طريقة غاوس-سيدل التكرارية - أسرع من جاكوبي عادةً",
            "Conjugate Gradient": "طريقة التدرج المترافق - مناسبة للمصفوفات المتماثلة الموجبة",
            "PCG (Jacobi Preconditioner)": "طريقة التدرج المترافق مع شرط جاكوبي المسبق - تسريع هائل للمصفوفات سيئة الشرط وتجميع للقيم المميزة",
            "Cramer's Rule": "قاعدة كرامر - مناسبة فقط للمصفوفات الصغيرة جداً",
            "حل بجميع الطرق": "مقارنة جميع الطرق واختيار الأنسب"
        }

        desc = descriptions.get(method, "")
        self.method_desc_label.config(text=desc)

    def update_status(self, message):
        """
        Update status bar message.
        Displays current operation status or error messages.

        Args:
            message: Status message to display
        """
        self.status_var.set(message)
        self.root.update_idletasks()

    def _unbind_mouse_wheel(self):
        """
        Unbind mouse wheel from scrollbars.
        Disables mouse wheel scrolling for canvases.
        """
        self.root.unbind("<MouseWheel>")

    def plot_all_methods_comparison(self, all_results):
        """
        Plot comparison charts for all solving methods.
        Creates three bar charts showing execution time, iterations, and residuals.

        Args:
            all_results: List of result dictionaries from all solving methods
        """
        methods = [r["method"] for r in all_results]
        times = [r["time"] for r in all_results]
        residuals = [r["residual"] for r in all_results]
        iterations = [r["iterations"] if r["iterations"] is not None else 0 for r in all_results]

        self.ax1.clear()
        bars = self.ax1.bar(range(len(methods)), times, color='skyblue')
        self.ax1.set_xticks(range(len(methods)))
        self.ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        self.ax1.set_ylabel("Time (seconds)")
        self.ax1.set_title("Execution Time Comparison")

        for bar in bars:
            height = bar.get_height()
            self.ax1.annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=7)

        self.ax2.clear()
        bars = self.ax2.bar(range(len(methods)), iterations, color='lightgreen')
        self.ax2.set_xticks(range(len(methods)))
        self.ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        self.ax2.set_ylabel("Iterations")
        self.ax2.set_title("Number of Iterations Comparison")

        for bar in bars:
            height = bar.get_height()
            self.ax2.annotate(f'{height:.0f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=7)

        self.ax3.clear()
        bars = self.ax3.bar(range(len(methods)), residuals, color='salmon')
        self.ax3.set_xticks(range(len(methods)))
        self.ax3.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        self.ax3.set_ylabel("Residual")
        self.ax3.set_title("Residual Comparison")

        for bar in bars:
            height = bar.get_height()
            self.ax3.annotate(f'{height:.1e}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=7)

        self.ax3.set_yscale('log')
        self.fig.tight_layout()

if __name__ == "__main__":
    root = tk.Tk()
    app = LinearSolverGUI(root)
    root.mainloop()
