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

def pollardRho(n:int, alpha:int, beta:int, o:int, timeout: float = 345600.0) -> Optional[int]:
    start_time = time.time()
    A = B = AA = BB = 0
    X = XX = 1

    def f(x, a, b, n_mod, order):
        partition = x % 3
        
        if partition == 1: 
            
            x_new = (beta * x) % n_mod
            a_new = a
            b_new = (b + 1) % order
            
        elif partition == 0: 
            
            x_new = pow(x, 2, n_mod)
            a_new = (2 * a) % order
            b_new = (2 * b) % order
            
        else: 
            
            x_new = (alpha * x) % n_mod
            a_new = (a + 1) % order
            b_new = b 
            
        return x_new, a_new, b_new

    while time.time() - start_time < timeout:
        X, A, B = f(X, A, B, n, o)
        XX = f(XX, AA, BB, n, o)
        XX = f(XX, AA, BB, n, o)
        p = gcd(B-BB,o)

        if X == XX: 
            if p != 1: return None 
            try:
                return (AA - A)*pow(B - BB, -1, o) % o
            except ValueError:
                return None
    return None



ALGORITMOS = {
    'baby' : babyStepGiantStep,
    'pollard_rho': pollardRho
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
            # Ignorar comentarios y líneas vacías
            if line.strip().startswith('#') or not line.strip():
                continue
            
            try:
                # Separar por comas y limpiar espacios
                parts = line.strip().split(',')
                
                # Asegurar que tenemos los 5 elementos
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
    parser = argparse.ArgumentParser(description="Ejecutar algoritmos de factorización midiendo su complejidad temporal y espacial.")
    
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