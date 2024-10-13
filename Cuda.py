from numba import cuda
import numpy as np
from timeit import default_timer as timer

# Função normal para rodar na CPU
def func(a):                                 
    for i in range(10000000): 
        a[i] += 1      

# Função otimizada para rodar na GPU
@cuda.jit
def func2(a): 
    i = cuda.grid(1)  # Obter o índice do thread
    if i < a.size:  # Verificar se o índice está dentro do limite
        a[i] += 1

if __name__ == "__main__": 
    n = 10000000                            
    a = np.ones(n, dtype=np.float64)

    # Medir tempo de execução na CPU
    start = timer() 
    func(a) 
    print("Sem GPU:", timer() - start)     
  
    # Transferir dados para a GPU
    d_a = cuda.to_device(a)

    # Definir grid e blocos
    threads_per_block = 256
    blocks_per_grid = (d_a.size + (threads_per_block - 1)) // threads_per_block

    # Medir tempo de execução na GPU
    start = timer() 
    func2[blocks_per_grid, threads_per_block](d_a)  # Executar o kernel
    cuda.synchronize()  # Sincronizar para garantir que a execução terminou
    print("Com GPU:", timer() - start)

    # Transferir resultados de volta para a CPU
    d_a.copy_to_host(a)
