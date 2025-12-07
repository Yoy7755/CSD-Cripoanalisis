from math import floor, gcd, sqrt
from random import randrange
from numpy import mod
import time
import threading
import os 
import concurrent.futures
from typing import List, Tuple, Optional
from sympy.ntheory.factor_ import pollard_pm1, pollard_rho
from sympy.ntheory.ecm import ecm
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

def factorizar(filename: str, outfile: str, timeout: float = 604800.0):
    """
    Correr en paralelo los diferentes métodos
    
    Args:
        filename (str): Path del fichero
        timeout (float): Timeout para cada método
    """
    challenges = leer_fichero(filename)
    
    def run_method(method, number):
        start_time = time.time()
        factor = method(number)
        end_time = time.time()
        return factor, end_time - start_time
    
    with open(outfile, "w") as f:
        for bit_size, number in challenges:
            print(f"\nProcessing: {bit_size} bits, n = {number}")
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_pollardp_1 = executor.submit(run_method, pollard_pm1, number)
                future_pollard = executor.submit(run_method, pollard_rho, number)
                future_lenstra = executor.submit(run_method, ecm, number)
                
                pollardp_1_factor, pollardp_1_time = future_pollardp_1.result()
                pollard_factor, pollard_time = future_pollard.result()
                lenstra_factor, lenstra_time = future_lenstra.result()
                
            result = {
                'bit_size': bit_size,
                'number': number,
                'pollardP_1_factor': pollardp_1_factor,
                'pollardP_1_time': pollardp_1_time,
                'pollard_factor': pollard_factor,
                'pollard_time': pollard_time,
                'lenstra_factor': lenstra_factor,
                'lenstra_time': lenstra_time,
            }
            
            f.write(str(result) + "\n")
            f.flush()
    
    return

def main():
    file = 'big_numbers.txt'
    
    # Ensure file exists
    if not os.path.exists(file):
        print(f"Error: Fichero '{file}' no encontrado!")
        return
    
    # Run challenges
    factorizar('big_numbers.txt', 'JESUCRISTO.txt')

if __name__ == "__main__":
    main()
