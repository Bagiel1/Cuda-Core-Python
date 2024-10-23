import numpy as np
from matplotlib import pyplot as plt
import random
import numba
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numba import cuda
import math
import time



#Kernel:
@cuda.jit

def kernel(matriz, tamanho, posicoes_iniciais, rng, dist_limite, caminhos, xs_gpu, ys_gpu, zs_gpu):
    i= cuda.grid(1)

    if i >= posicoes_iniciais.shape[0]:
        return
    

    xs_t= posicoes_iniciais[i,0]
    ys_t= posicoes_iniciais[i,1]
    zs_t= posicoes_iniciais[i,2]
    
    if math.sqrt((xs_t - tamanho // 2) ** 2 + (ys_t - tamanho // 2) ** 2 + (zs_t - tamanho // 2)**2)>= dist_limite:
        condicao= True

        while condicao and 0 <= xs_t <= tamanho and 0 <= ys_t <= tamanho and 0 <= zs_t <= tamanho:
            direcao = int(xoroshiro128p_uniform_float32(rng, i) * 1000) % 6          
            
            vizinho_encontrado = False
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        if 0 <= xs_t + dx < tamanho and 0 <= ys_t + dy < tamanho and 0 <= zs_t + dz <= tamanho:
                            if (dx != 0 or dy != 0 or dz != 0) and matriz[xs_t + dx, ys_t + dy, zs_t + dz] == 1:
                                vizinho_encontrado = True
                                break
        
            if vizinho_encontrado:
                matriz[xs_t,ys_t,zs_t] = 1
                xs_gpu[i]= xs_t
                ys_gpu[i]= ys_t
                zs_gpu[i]= zs_t
                condicao= False
            
            else:
                xs_t += caminhos[direcao][0]
                ys_t += caminhos[direcao][1]
                zs_t += caminhos[direcao][2]
        

def programa():
    tamanho= 400
    dist_limite= 180
    p= 1100000

    matriz= np.zeros((tamanho+1,tamanho+1,tamanho+1), dtype=np.int32)
    matriz[tamanho//2, tamanho//2, tamanho//2] = 1

    angulos= np.random.uniform(0,2*np.pi,p)
    angulosz= np.random.uniform(-np.pi/2, np.pi/2, p)
    distancias= np.random.uniform(dist_limite, tamanho//2, p)
    px= (np.cos(angulos) * distancias).astype(int) + tamanho//2
    py= (np.sin(angulos)*distancias).astype(int) + tamanho//2
    pz= (np.sin(angulosz) * distancias).astype(int) + tamanho//2

    posicoes_iniciais= np.stack((px,py,pz), axis=-1)

    caminhos= np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])

    caminhos_gpu = cuda.to_device(caminhos)
    matriz_gpu = cuda.to_device(matriz)
    posicoes_gpu= cuda.to_device(posicoes_iniciais)

    rng = create_xoroshiro128p_states(p, seed=1)

    xs_gpu= cuda.device_array(p, dtype=np.int32)
    ys_gpu= cuda.device_array(p, dtype=np.int32)
    zs_gpu= cuda.device_array(p, dtype=np.int32)


    threads_por_block = 256
    blocks_por_grid = (p + threads_por_block-1) // threads_por_block

    inicio= time.time()
    kernel[blocks_por_grid, threads_por_block](matriz_gpu, tamanho, posicoes_gpu, rng, dist_limite, caminhos_gpu, xs_gpu, ys_gpu, zs_gpu)
    fim= time.time() - inicio

    matriz= matriz_gpu.copy_to_host()
    xs_cpu = xs_gpu.copy_to_host()
    ys_cpu = ys_gpu.copy_to_host()
    zs_cpu = zs_gpu.copy_to_host()

    print(fim)

    mask = (xs_cpu != 0) | (ys_cpu != 0) | (zs_cpu != 0)
    xs_cpu = xs_cpu[mask]
    ys_cpu = ys_cpu[mask]
    zs_cpu = zs_cpu[mask]

    fig = plt.figure(figsize=(10,7))
    ax= fig.add_subplot(111, projection='3d')
    
    ax.scatter(xs_cpu,ys_cpu,zs_cpu)
    ax.set_xlim(0,400)
    ax.set_ylim(0,400)
    ax.set_zlim(0,400)
    plt.show()

programa()
        
    