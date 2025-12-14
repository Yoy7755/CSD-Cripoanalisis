from math import ceil, floor, gcd, sqrt, log, exp, isqrt
from random import randrange
from numpy import mod
import math
import random
import time
import os 
import concurrent.futures
from typing import List, Tuple, Optional
import tracemalloc
import argparse

def babyStepGiantStep(p: int, alfa: int, beta: int, order: int ,timeout = 34600) -> Optional[int]:
    n = isqrt(order) + 1
    T= {}

    current = 1
    for i in range(n):
        T[current] = i
        current = (current * alfa) % p
    alfa_minus = pow(alfa, -n, p)
    gamma = beta
    for i in range (0, n):
        if gamma in T: return i*n + T[gamma]
        gamma = (gamma*alfa_minus) % p

    return

def pollardRho(n: int, alpha: int, beta: int, order: int, timeout: float = 345600.0) -> Optional[int]:
    """
    Algoritmo Pollard's Rho para Logaritmo Discreto corregido.
    n: módulo p (donde ocurre la operación principal)
    alpha: generador
    beta: valor objetivo (alpha^x = beta mod n)
    order: orden del subgrupo (importante para las operaciones en los exponentes)
    """
    start_time = time.time()

    def f(x, a, b):
        """Función de iteración estándar dividida en 3 conjuntos."""
        subset = x % 3
        if subset == 0:
            new_x = (x * x) % n
            new_a = (a * 2) % order
            new_b = (b * 2) % order
        elif subset == 1:
            new_x = (x * beta) % n
            new_a = a
            new_b = (b + 1) % order
        else:
            new_x = (x * alpha) % n
            new_a = (a + 1) % order
            new_b = b
        return new_x, new_a, new_b

    while time.time() - start_time < timeout:
        a = randrange(0, order)
        b = randrange(0, order)
        x = (pow(alpha, a, n) * pow(beta, b, n)) % n
        
        X, A, B = x, a, b
        XX, AA, BB = x, a, b

        for _ in range(order):
            if time.time() - start_time > timeout: return None
            
            X, A, B = f(X, A, B)
            
            XX, AA, BB = f(XX, AA, BB)
            XX, AA, BB = f(XX, AA, BB)

            if X == XX:
                
                delta_b = (B - BB) % order
                delta_a = (AA - A) % order
                
                g = gcd(delta_b, order)
                
                if g == 1:
                    try:
                        inv = pow(delta_b, -1, order)
                        return (delta_a * inv) % order
                    except ValueError:
                        break 
                else:
                    if delta_a % g == 0:
                        try:
                            inv = pow(delta_b // g, -1, order // g)
                            res = (delta_a // g * inv) % (order // g)
                            
                            for k in range(g):
                                candidate = res + k * (order // g)
                                if pow(alpha, candidate, n) == beta:
                                    return candidate
                        except ValueError:
                            pass 
                break 
                
    return None

def _get_primes(limit: int) -> List[int]:
    """Genera una lista de primos hasta el límite."""
    primes = []
    is_prime = [True] * (limit + 1)
    for p in range(2, limit + 1):
        if is_prime[p]:
            primes.append(p)
            for i in range(p * p, limit + 1, p):
                is_prime[i] = False
    return primes

def _is_smooth_and_factor(num: int, base: List[int]) -> Tuple[bool, List[int]]:
    """
    Verifica si num es S-smooth sobre la base.
    Devuelve (True, lista_de_exponentes) o (False, []).
    """
    exponents = [0] * len(base)
    temp = num
    
    for i, p in enumerate(base):
        while temp % p == 0:
            exponents[i] += 1
            temp //= p
            if temp == 1:
                return True, exponents
    
    return temp == 1, exponents

def _solve_linear_system_mod(matrix: List[List[int]], target: List[int], modulus: int) -> Optional[List[int]]:
    """
    Resuelve Ax = B (mod m) usando Eliminación Gaussiana básica.
    matrix: A (lista de filas), target: B.
    Devuelve la solución x (lista de valores) o None si falla.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    
    M = [row[:] for row in matrix]
    res = target[:]
    
    pivot_row = 0
    col = 0
    
    while pivot_row < rows and col < cols:
        sel = -1
        for i in range(pivot_row, rows):
            if gcd(M[i][col], modulus) == 1: 
                sel = i
                break
        
        if sel == -1:
            col += 1
            continue
            
        M[pivot_row], M[sel] = M[sel], M[pivot_row]
        res[pivot_row], res[sel] = res[sel], res[pivot_row]
        
        try:
            inv = pow(M[pivot_row][col], -1, modulus)
        except ValueError:
            col += 1; continue 
        
        for j in range(col, cols):
            M[pivot_row][j] = (M[pivot_row][j] * inv) % modulus
        res[pivot_row] = (res[pivot_row] * inv) % modulus
        
        for i in range(rows):
            if i != pivot_row and M[i][col] != 0:
                factor = M[i][col]
                for j in range(col, cols):
                    M[i][j] = (M[i][j] - factor * M[pivot_row][j]) % modulus
                res[i] = (res[i] - factor * res[pivot_row]) % modulus
                
        pivot_row += 1
        col += 1
    
    solution = [0] * cols
    for i in range(min(rows, cols)):
        for j in range(cols):
            if M[i][j] == 1:
                solution[j] = res[i]
                break
                
    return solution

def indexCalculus(n: int, alpha: int, beta: int, order: int, timeout: float = 345600.0) -> Optional[int]:
    """
    Algoritmo Index Calculus para Logaritmo Discreto.
    n: módulo p
    alpha: generador
    beta: valor objetivo
    order: orden del grupo
    """
    start_time = time.time()
    
    B_bound = 1000
    S = _get_primes(B_bound)
    k = len(S)
    
    relations_matrix = [] 
    relations_rhs = []
    
    required_relations = k + 10 
    
    while len(relations_matrix) < required_relations:
        if time.time() - start_time > timeout: return None
        
        r = randrange(1, order)
        val = pow(alpha, r, n) 
        
        is_smooth, exponents = _is_smooth_and_factor(val, S)
        
        if is_smooth:
            relations_matrix.append(exponents)
            relations_rhs.append(r)
    
    logs_S = None
    while logs_S is None:
        if time.time() - start_time > timeout: return None
        try:
            logs_S = _solve_linear_system_mod(relations_matrix, relations_rhs, order)
        except Exception:
            pass 
            
        if logs_S is None:
            found_new = 0
            while found_new < 5:
                if time.time() - start_time > timeout: return None
                r = randrange(1, order)
                val = pow(alpha, r, n)
                is_smooth, exponents = _is_smooth_and_factor(val, S)
                if is_smooth:
                    relations_matrix.append(exponents)
                    relations_rhs.append(r)
                    found_new += 1
    
    while time.time() - start_time < timeout:
        r = randrange(1, order)
        val = (beta * pow(alpha, r, n)) % n
        
        is_smooth, exponents = _is_smooth_and_factor(val, S)
        
        if is_smooth:
            
            sum_logs = 0
            for i, e_i in enumerate(exponents):
                sum_logs = (sum_logs + e_i * logs_S[i]) % order
            
            result = (sum_logs - r) % order
            return result
            
    return None



ALGORITMOS = {
    'baby' : babyStepGiantStep,
    'pollard_rho': pollardRho,
    'index_calculus': indexCalculus
}

def leer_fichero(filename: str) -> List[Tuple[int, int, int, int, int]]:
    """
    Lee el fichero de retos de Logaritmo Discreto.
    
    Args:
        filename (str): path al fichero
    
    Returns:
        Lista de tuplas (n_bits, p, alpha, beta, orden)
    """
    retos = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().startswith('#') or not line.strip():
                continue
            
            try:
                parts = line.strip().split(',')
                
                if len(parts) != 5:
                    raise ValueError("Número incorrecto de elementos en la línea")

                n_bits = int(parts[0].strip())
                p = int(parts[1].strip())
                alpha = int(parts[2].strip())
                beta = int(parts[3].strip())
                orden = int(parts[4].strip())
                
                retos.append((n_bits, p, alpha, beta, orden))
                
            except (ValueError, IndexError) as e:
                print(f"Saltándose linea invalida ({e}): {line.strip()}")
    
    return retos


def run_method(metodo_name, p, alpha, beta, order, timeout):
    """
    Trabajador que ejecuta en paralelo un algoritmo de Logaritmo Discreto.
    Busca x tal que: alpha^x = beta (mod p)
    """
    method = ALGORITMOS.get(metodo_name)
    
    if not method:
        return None, 0, 0

    tracemalloc.start()
    start_time = time.time()
    
    result = None
    try:
        result = method(p, alpha, beta, order, timeout=timeout)
    except Exception as e:
        print(f"Error interno en algoritmo {metodo_name}: {e}")
        result = None
        
    end_time = time.time()
    current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return result, end_time - start_time, peak_memory


def resolver_dlp(filename: str, outfile: str, algorithms: list, timeout: float = 84000.0):
    """
    Orquestador para resolver retos de Logaritmo Discreto.
    """
    challenges = leer_fichero(filename)
    
    with open(outfile, "a") as f:
        
        for n_bits, p, alpha, beta, order in challenges:
            print(f"\nProcesando DLP {n_bits} bits: {alpha}^x = {beta} (mod {p})")
            
            result_row = {
                'bit_size': n_bits, 
                'p': p,
                'alpha': alpha,
                'beta': beta,
                'order': order
            }

            with concurrent.futures.ProcessPoolExecutor() as executor:
                future_to_algo = {}
                
                for algo_name in algorithms:
                    if algo_name == 'baby' and n_bits > 52:
                        print(f"  [Protección] Saltando BSGS para {n_bits} bits (Riesgo de colapso de RAM).")
                        result_row[f'{algo_name}_result'] = None
                        result_row[f'{algo_name}_time'] = 0
                        result_row[f'{algo_name}_mem_mb'] = 0
                        continue
                    if algo_name in ALGORITMOS:
                        future = executor.submit(run_method, algo_name, p, alpha, beta, order, timeout)
                        future_to_algo[future] = algo_name
                    else:
                        print(f"Advertencia: El algoritmo '{algo_name}' no está en el diccionario ALGORITMOS.")
                
                for future in concurrent.futures.as_completed(future_to_algo):
                    algo_name = future_to_algo[future]
                    try:
                        dlp_solution, duration, peak_mem = future.result()
                        
                        result_row[f'{algo_name}_result'] = dlp_solution
                        result_row[f'{algo_name}_time'] = duration
                        result_row[f'{algo_name}_mem_mb'] = peak_mem / (1024 * 1024)
                        
                        if dlp_solution is not None:
                            print(f"  > {algo_name} encontró solución: {dlp_solution} en {duration:.4f}s")
                        else:
                            print(f"  > {algo_name} no encontró solución o expiró.")

                    except Exception as exc:
                        print(f'{algo_name} generó una excepción crítica: {exc}')

            f.write(str(result_row) + "\n")
            f.flush()

def main():
    parser = argparse.ArgumentParser(description="Ejecutar algoritmos de logaritmos discretos midiendo su complejidad temporal y espacial.")
    
    parser.add_argument("--file", type=str, required=True, help="Archivo de Entrada")
    parser.add_argument("--out", type=str, required=True, help="Archivo de Salida")
    parser.add_argument("--algos", nargs='+', choices=ALGORITMOS.keys(), required=True,
                        help=f"Lista de algoritmos. Posibles algoritmos: {list(ALGORITMOS.keys())}")
    parser.add_argument("--timeout", type=float, default=84000.0, help="Timeout en segundos a definir")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found!")
        return

    print(f"Atacando el logaritmo discreto {args.file}")
    print(f"Algoritmos: {args.algos}")
    print(f"Salida: {args.out}")

    resolver_dlp(args.file, args.out, args.algos, args.timeout)

if __name__ == "__main__":
    main()