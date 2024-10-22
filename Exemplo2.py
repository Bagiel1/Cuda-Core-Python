from numba import cuda
import numpy as np
import time

@cuda.jit
def kernel(matriz1, matriz2, result):
    # Obter os índices da thread na grade 2D
    row, col = cuda.grid(2)  # Obtendo índice de linha e coluna
    # Verificar se os índices estão dentro dos limites da matriz resultante
    if row < result.shape[0] and col < result.shape[1]:
        sum = 0
        for k in range(matriz1.shape[1]):  # Ou len(matriz2) - as duas devem ser iguais
            sum += matriz1[row, k] * matriz2[k, col]
        result[row, col] = sum  # Atribui o resultado

def GPU(matriz1, matriz2):
    host_array1 = np.array(matriz1)
    host_array2 = np.array(matriz2)

    # Transferir as matrizes para a GPU
    device_array1 = cuda.to_device(host_array1)
    device_array2 = cuda.to_device(host_array2)

    # Alocar a matriz de resultado na GPU
    result = cuda.device_array((host_array1.shape[0], host_array2.shape[1]), dtype=np.int32)

    # Configuração de threads e blocos
    threads_per_block = (16, 16)  # Usando um bloco 2D
    blocks_per_grid = ((host_array1.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
                        (host_array2.shape[1] + threads_per_block[1] - 1) // threads_per_block[1])

    # Executar o kernel
    kernel[blocks_per_grid, threads_per_block](device_array1, device_array2, result)

    # Copiar o resultado de volta para o host
    result_host = result.copy_to_host()

    return result_host



def NoGPU(matriz1,matriz2):
    if len(matriz1[0]) == len(matriz2):
        result= [[0]*len(matriz2[0]) for i in range(len(matriz1))]
        for i in range(len(matriz1)):
            for j in range(len(matriz2[0])):
                sum=0
                for k in range(len(matriz2)):
                    sum += matriz1[i][k] * matriz2[k][j]
                result[i][j] = sum
    return result


c= 1000
n= 4000
d= 3000
a= np.random.randint(0,1000, size=(c,n))
b= np.random.randint(0,1000, size=(n,d))

start= time.time()  
GPU(a,b)
end= time.time() - start
print(end)

start= time.time()
NoGPU(a,b)
end= time.time() - start
print(end)