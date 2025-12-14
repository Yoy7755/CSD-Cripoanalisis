from math import floor, gcd, sqrt
from random import randrange
from numpy import mod
import time
import threading
import tracemalloc
import os 
import concurrent.futures
from typing import List, Tuple, Optional
from sympy.ntheory.factor_ import pollard_pm1, pollard_rho
from sympy.ntheory.ecm import ecm
from sympy.ntheory.qs import qs
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

def factorizar(filename: str, outfile: str, timeout: float = 3600.0):
    """
    Correr en paralelo los diferentes métodos y rastrear memoria.
    
    Args:
        filename (str): Path del fichero
        outfile (str): Path del fichero de salida
        timeout (float): Timeout para cada método
    """
    # Assuming leer_fichero and qs are defined elsewhere in your scope
    challenges = leer_fichero(filename)
    
    class RunWithTimeout:
        def __init__(self, function, args):
            self.function = function
            self.args = args
            self.answer = None
            self.exception = None
            self.completed = False
            self.time = None
            self.peak_memory = 0  # Initialize memory tracker

        def worker(self):
            try:
                # Start tracing memory allocations
                tracemalloc.start()
                
                start_time = time.time()
                self.answer = self.function(*self.args)
                self.time = time.time() - start_time
                
                # Get memory usage: (current, peak)
                _, peak = tracemalloc.get_traced_memory()
                self.peak_memory = peak
                
                # Stop tracing
                tracemalloc.stop()
                
                self.completed = True
            except Exception as e:
                self.exception = e
                self.completed = True
                # Ensure we stop tracing even if there is an error
                tracemalloc.stop()

        def run(self, timeout):
            thread = threading.Thread(target=self.worker)
            thread.daemon = True
            thread.start()
            thread.join(timeout)
            
            if not self.completed:
                return None, None, None, "Timeout"
            elif self.exception:
                return None, None, None, f"Error: {self.exception}"
            else:
                return self.answer, self.time, self.peak_memory, "Success"
    
    with open(outfile, "w") as f:
        # B = 2**32 - 1  # Unused variable in this scope, but kept if you need it later
        
        for bit_size, number in challenges:
            print(f"\nProcessing: {bit_size} bits, n = {number}")

            # Prepare arguments. Note: prime_bound and 10000 passed as args to qs
            runner = RunWithTimeout(qs, (number, 4000, 10000))
            
            # Unpack the 4 return values now
            result, exec_time, peak_mem, status = runner.run(timeout=timeout)
            
            print(f"Status: {status} | Time: {exec_time}s | Peak Mem: {peak_mem} bytes")
            
            output_data = {
                'bit_size': bit_size,
                'number': number,
                'criba_cuadratica_factor': result,
                'criba_cuadratica_time': exec_time,
                'criba_cuadratica_memory_bytes': peak_mem, # Added memory field
                'status': status
            }
                
            f.write(str(output_data) + "\n")
            f.flush()
    
    return

def main():
    file = 'ExtensionRetosFactorizacion.txt'
    
    # Ensure file exists
    if not os.path.exists(file):
        print(f"Error: Fichero '{file}' no encontrado!")
        return
    
    # Run challenges
    factorizar(file, 'Resultados_qs.txt')

if __name__ == "__main__":
    main()
