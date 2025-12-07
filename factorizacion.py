from math import floor, gcd, sqrt, log, exp, isqrt
from random import randrange
from numpy import mod
import random
import time
import signal
import os 
import concurrent.futures
from typing import List, Tuple, Optional

class EllipticCurve:
    def __init__(self, a: int, b: int, p: int):
        """Parámetros a, b sobre grupo p"""
        self.a = a
        self.b = b
        self.p = p
        
        # Si 4a^3 + 27b^2 ≠ 0 (mod p) 
        disc = (4 * (a ** 3) + 27 * (b ** 2)) % p
        if disc == 0:
            raise ValueError("La curva es singular")

class Point:
    def __init__(self, curve: EllipticCurve, x: int = None, y: int = None):
        """Inicializar un punto con sus coordenadas y al curva a la que pertenece.
           None representa el infinito"""
        self.curve = curve
        self.x = x
        self.y = y  

        if x is None and y is None:
            return
            
        # Comprobar si el punto esta en la curva
        if not self._is_on_curve():
            raise ValueError("El punto no está en la curva")

    
    def _is_on_curve(self) -> bool:
        """Comprobar si el punto esta en la curva"""
        if self.x is None or self.y is None:  # El punto en el infinito siempre pertenece
            return True
            
        #modulo p
        x = self.x % self.curve.p
        y = self.y % self.curve.p
        
        # Comprobar si cumple: y^2 = x^3 + ax + b (mod p)
        left = (y * y) % self.curve.p
        right = (pow(x, 3, self.curve.p) + 
                (self.curve.a * x) % self.curve.p + 
                self.curve.b) % self.curve.p
        return left == right
    
    def __eq__(self, other: 'Point') -> bool:
        """Comprobar si dos puntos son iguales"""
        if not isinstance(other, Point):
            return False
        if self.x is None or other.x is None:
            return self.x is None and other.x is None
        return (self.curve.a == other.curve.a and 
                self.curve.b == other.curve.b and 
                self.curve.p == other.curve.p and 
                self.x % self.curve.p == other.x % self.curve.p and 
                self.y % self.curve.p == other.y % self.curve.p)
    
    def __neg__(self) -> 'Point':
        """Devolver el punto negado"""
        if self.x is None:  # El punto en el infinito es su propio inverso
            return self
        return Point(self.curve, self.x, (-self.y) % self.curve.p)
    
    def den_inv(self, other: 'Point') -> int:
        if self.x is None or other.x is None:
            raise TypeError("No se puede calcular de un punto en el infinito")
        if self == other:
            den = (2 * self.y) % self.curve.p
        else:
            den = (other.x - self.x) % self.curve.p

        return pow(den, -1, self.curve.p)
    
    def __add__(self, other: 'Point') -> 'Point':
        if not isinstance(other, Point) or self.curve.p != other.curve.p:
            raise TypeError("Los puntos deben pertenecer a la misma curva")
        
        if self.x is None: return other
        if other.x is None: return self
        if self == -other: return Point(self.curve)
            
        try:
            if self == other:
                if self.y == 0:
                    return Point(self.curve)
                num = (3 * pow(self.x, 2, self.curve.p) + self.curve.a) % self.curve.p
                den = (2 * self.y) % self.curve.p
                if den == 0:
                    raise ZeroDivisionError(den)
                try:
                    slope = (num * pow(den, -1, self.curve.p)) % self.curve.p
                except ValueError:
                    raise ZeroDivisionError(den)
            else:
                num = (other.y - self.y) % self.curve.p
                den = (other.x - self.x) % self.curve.p
                if den == 0:
                    raise ZeroDivisionError(den)
                try:
                    slope = (num * pow(den, -1, self.curve.p)) % self.curve.p
                except ValueError:
                    raise ZeroDivisionError(den)
                
            x3 = (pow(slope, 2, self.curve.p) - self.x - other.x) % self.curve.p
            y3 = (-self.y + slope * (self.x - x3)) % self.curve.p
            
            return Point(self.curve, x3, y3)
            
        except ZeroDivisionError as e:
            raise ZeroDivisionError(e.args[0])
    
    def __sub__(self, other: 'Point') -> 'Point':
        """Resta de puntos"""
        return self + (-other)

    def __mul__(self, scalar: int) -> 'Point':
        """Multiplicacion punto por escalar usando el double-and-add"""
        if not isinstance(scalar, int):
            raise TypeError("El escalar tiene que ser un entero")
            
        # Casos especiales
        if scalar < 0:
            return (-self) * (-scalar) # El punto negado por escalar en positivo
        if scalar == 0:
            return Point(self.curve)  # Multiplicar por 0 devuelve el punto en el infinito
        if self.x is None:
            return self  # Multiplicar el punto en el infinito da él mismo
            
        # Algoritmo Double-and-add 
        result = Point(self.curve)  # Empezamos con el punto en el infinito
        current = self
        while scalar:
            if scalar & 1:  # Si el bit es 1, se suma al resultado
                result += current
            current += current  # Se duplica el punto
            scalar >>= 1  # Siguiente bit
            
        return result
    
    def __rmul__(self, scalar: int) -> 'Point':
        """Multiplicación por la derecha"""
        return self * scalar

    def __str__(self) -> str:
        """Punto a string"""
        if self.x is None:
            return "Point(∞)"
        return f"Point({self.x}, {self.y})"

# Método pollard p-1 para la descomposición de números en factores
def pollardP_1(n:int, timeout: float = 345600.0) -> Optional[int]:
    start_time = time.time()
    A = randrange(start=2, stop= n - 1, step=1)
    p = gcd(A,n)
    if 1 < p < n: return p
    k = 2
    while time.time() - start_time < timeout:
        A = pow(A,k,n)
        d = gcd(A-1,n)
        if 1 < d < n: return d
        if d == n: return False
        k += 1

    return None


# Método pollardRho de descomposición de números en factores
def pollardRho(n:int, timeout: float = 345600.0) -> Optional[int]:
    start_time = time.time()
    A = B = randrange(start=2, stop= n - 1, step=1)
    while time.time() - start_time < timeout:
        A = (A*A + 1) % n
        B = (B*B + 1) % n
        B = (B*B + 1) % n
        p = gcd(A-B,n)
        if 1 < p < n: return p
        if p == n: return n
    
    return None

# Método de factorización de curva elíptica de Lenstra
def Lenstra(n: int, B: int = 1000, timeout: float = 345600.0) -> Optional[int]:

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                x = randrange(n)
                y = randrange(n)
                a = randrange(n)
            
                b = (pow(y, 2, n) - (pow(x, 3, n) + (a * x) % n)) % n
                c = EllipticCurve(a, b, n)
                p = Point(c, x, y)
                
                
                current = p
                for i in range(2, B + 1):
                    current = current * i  
                    
            except ZeroDivisionError as e:
                
                den = int(str(e).split()[0])
                factor = gcd(den, n)
                if 1 < factor < n:
                    return factor
        # print(f"Lenstra superó el timeout de {timeout} segundos")
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

def _is_smooth(num: int, base: List[int]) -> Tuple[bool, List[int]]:
    """Verifica si num es B-smooth y devuelve el vector de exponentes."""
    exponents = [0] * (len(base) + 1) # +1 para el factor -1
    
    if num < 0:
        exponents[0] = 1
        num = -num
    
    for i, p in enumerate(base):
        while num % p == 0:
            exponents[i + 1] += 1
            num //= p
            
    return (num == 1, exponents)

def _solve_gaussian(matrix: List[List[int]]) -> List[int]:
    """Resuelve el sistema lineal mod 2 (Eliminación Gaussiana)."""
    rows = len(matrix)
    if rows == 0: return []
    cols = len(matrix[0])
    
    aug_matrix = [row[:] + [1 if i == r else 0 for i in range(rows)] for r, row in enumerate(matrix)]
    
    pivot_row = 0
    col = 0
    while pivot_row < rows and col < cols:
        if aug_matrix[pivot_row][col] % 2 == 0:
            for i in range(pivot_row + 1, rows):
                if aug_matrix[i][col] % 2 == 1:
                    aug_matrix[pivot_row], aug_matrix[i] = aug_matrix[i], aug_matrix[pivot_row]
                    break
            else:
                col += 1
                continue
        
        for i in range(rows):
            if i != pivot_row and aug_matrix[i][col] % 2 == 1:
                for j in range(col, len(aug_matrix[i])):
                    aug_matrix[i][j] = (aug_matrix[i][j] + aug_matrix[pivot_row][j])
        
        pivot_row += 1
        col += 1

    for row in aug_matrix:
        if all(x % 2 == 0 for x in row[:cols]):
            indices = [i for i, x in enumerate(row[cols:]) if x % 2 == 1]
            if indices: return indices
    return []

def quadraticSieve(n: int, timeout: float = 345600.0) -> Optional[int]:
    """
    Algoritmo de Factorización por Criba Cuadrática.
    Fase 1: Búsqueda de relaciones (Sieving).
    Fase 2: Álgebra lineal sobre GF(2).
    """
    start_time = time.time()
    
    if n % 2 == 0: return 2
    root_n = isqrt(n)
    if root_n * root_n == n: return root_n

    # Configuración de la base de factores B [cite: 46]
    # Heurística simple para tamaño de base
    try:
        B_limit = int(exp(0.5 * sqrt(log(n) * log(log(n))))) + 50
    except ValueError:
        B_limit = 100 # Fallback para números muy pequeños

    primes = _get_primes(B_limit) 
    base = primes 
    relations = [] 
    
    # Secuencia de desplazamiento: 0, 1, -1, 2, -2... 
    count = 0
    
    while time.time() - start_time < timeout:
        if count % 2 == 0:
            current_shift = count // 2
        else:
            current_shift = -(count // 2 + 1)
        count += 1
        
        a = root_n + current_shift
        b2 = (a * a) - n # b2 = a^2 - n [cite: 20]
        
        is_smooth, exponents = _is_smooth(b2, base) # Verificar B-smooth [cite: 30]
        
        if is_smooth:
            relations.append({'a': a, 'exponents': exponents, 'b2': b2})
            
            # Se necesitan suficientes relaciones (k + c) 
            if len(relations) > len(base) + 5:
                matrix_mod2 = [[e % 2 for e in r['exponents']] for r in relations]
                indices = _solve_gaussian(matrix_mod2) # Resolver sistema lineal [cite: 124]
                
                if indices:
                    x = 1
                    y_exponents = [0] * (len(base) + 1)
                    
                    for idx in indices:
                        rel = relations[idx]
                        x = (x * rel['a']) % n
                        for i, exp_val in enumerate(rel['exponents']):
                            y_exponents[i] += exp_val
                    
                    y = 1
                    # Calcular y = sqrt(producto b2_i) [cite: 96]
                    for i, p in enumerate(base):
                        exp_val = y_exponents[i + 1] // 2
                        if exp_val > 0:
                            y = (y * pow(p, exp_val, n)) % n
                            
                    # Verificar x ≡ ±y (mod n) [cite: 96]
                    if x != y and x != (n - y):
                        p_factor = gcd(x - y, n) # mcd(x-y, n) [cite: 102]
                        if 1 < p_factor < n:
                            return p_factor
                    
                    # Si falla, eliminamos la primera relación para probar otra combinación
                    if indices:
                         relations.pop(indices[0])
    return None

# híbrido entre Lenstra y PollardRho
def hibrido(n: int, B: int = 3000, timeout: float = 345600.0) -> Optional[int]:
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                x1 = randrange(n)
                y1 = randrange(n)
                x2 = randrange(n)
                y2 = randrange(n)
                
                while x1 == x2:
                    x2 = randrange(n)

                num = (pow(y1,2,n) - pow(y2,2,n) - (pow(x1,3,n) - pow(x2,3,n))) % n
                
                try:
                    den = pow(x1 - x2,-1, n)
                except Exception:
                    raise ZeroDivisionError(den)
                
                a = (num * den) % n
    
                b = (pow(y1, 2, n) - pow(x1, 3, n) - (a * x1) % n) % n
                c = EllipticCurve(a, b, n)

                P = Q = Point(c, x1, y1)
                R = Point(c, x2, y2)
                i = 0
                while i < B:
                    i+=1
                    P = P + (2 * R)
                    Q = Q + R
                    p_x = gcd(Q.x - P.x, n)
                    if 1 < p_x < n: return p_x
                    p_y = gcd(Q.y - P.y, n)
                    if 1 < p_y < n: return p_y

                    
            except ZeroDivisionError as e:  
                den = int(str(e).split()[0])
                factor = gcd(den, n)
                if 1 < factor < n:
                    return factor
        
        print(f"Hibrido superó el timeout de {timeout} segundos")
        return None

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

def factorizar(filename: str, outfile: str, timeout: float = 84000.0):
    """
    Correr en paralelo los diferentes métodos
    
    Args:
        filename (str): Path del fichero
        timeout (float): Timeout para cada método
    """
    challenges = leer_fichero(filename)
    
    def run_method(method, number):
        start_time = time.time()
        factor = method(number, timeout=timeout)
        end_time = time.time()
        return factor, end_time - start_time
    
    with open(outfile, "w") as f:
        for bit_size, number in challenges:
            print(f"\nProcessing: {bit_size} bits, n = {number}")
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_pollardp_1 = executor.submit(run_method, pollardP_1, number)
                future_pollard = executor.submit(run_method, pollardRho, number)
                future_lenstra = executor.submit(run_method, Lenstra, number)
                
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
    file = 'ProblemasFactorizacion.txt'
    
    # Ensure file exists
    if not os.path.exists(file):
        print(f"Error: Fichero '{file}' no encontrado!")
        return
    
    # Run challenges
    factorizar(file, 'resultados_sinhibridocontinuacion.txt')

if __name__ == "__main__":
    main()
