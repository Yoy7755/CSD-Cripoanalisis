from math import floor, gcd, sqrt
from random import randrange
from numpy import mod
import time
import threading
import os 
import concurrent.futures
from typing import List, Tuple, Optional
from sympy.ntheory.factor_ import pollard_pm1, pollard_rho
from sympy.ntheory.qs import qs


ALGORITMOS = {
    'qs': qs

}
def leer_fichero(filename: str) -> List[Tuple[int, int]]:
    """
    Leer el fichero y procesar cada par bits/numero
    
    Args:
        filename (str): path al fichero
    
    Returns:
        Lista de tuplas (bit_size, number)
    """
    numeros = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().startswith('#'):
                continue
            
            try:
                bit_size, number = line.strip().split(',')
                numeros.append((int(bit_size.strip()), int(number.strip())))
            except (ValueError, IndexError):
                print(f"Saltandose linea invalida: {line.strip()}")
    
    return numeros

def run_method(metodo, number, timeout):
    """
    Trabajador que ejecuta en paralelo las pruebas de un algoritmo.
    """
    method = ALGORITMOS[metodo]
    
    tracemalloc.start()
    start_time = time.time()
    
    try:
        factor = method(number, timeout=timeout)
    except Exception as e:
        factor = None
        
    end_time = time.time()
    current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return factor, end_time - start_time, peak_memory

def factorizar(filename: str, outfile: str, algorithms: list, timeout: float = 84000.0):
    challenges = leer_fichero(filename)
    
    with open(outfile, "a") as f:
        
        for bit_size, number in challenges:
            print(f"\nProcesando: {bit_size} bits, n = {number}")
            
            result_row = {
                'bit_size': bit_size, 
                'number': number
            }

            with concurrent.futures.ProcessPoolExecutor() as executor:
                future_to_algo = {}
                
                for algo_name in algorithms:
                    if algo_name in ALGORITMOS:
                        future = executor.submit(run_method, algo_name, number, timeout)
                        future_to_algo[future] = algo_name
                
                for future in concurrent.futures.as_completed(future_to_algo):
                    algo_name = future_to_algo[future]
                    try:
                        factor, duration, peak_mem = future.result()
                        
                        result_row[f'{algo_name}_factor'] = factor
                        result_row[f'{algo_name}_time'] = duration
                        result_row[f'{algo_name}_mem_mb'] = peak_mem / (1024 * 1024)
                    except Exception as exc:
                        print(f'{algo_name} generó una excepción: {exc}')

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

    print(f"Factorizando {args.file}")
    print(f"Algoritmos: {args.algos}")
    print(f"Salida: {args.out}")

    factorizar(args.file, args.out, args.algos, args.timeout)
