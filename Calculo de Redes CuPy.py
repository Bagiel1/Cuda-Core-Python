import cupy as cp

def criarmatrizC(num):
    a = cp.random.randint(1, 4, num)
    return a

def criarConecGrade(num_linhas, num_colunas):
    conec = []
    canos = (num_linhas * (num_colunas - 1)) + ((num_linhas - 1) * num_colunas)
    for i in range(num_linhas):
        for j in range(num_colunas):
            no_atual = i * num_colunas + j + 1

            if j < num_colunas - 1:
                no_direita = no_atual + 1
                conec.append([no_atual, no_direita])

            if i < num_linhas - 1:
                no_baixo = no_atual + num_colunas
                conec.append([no_atual, no_baixo])

    return cp.array(conec), canos

def Assembly(conec, C):
    n_nos = int(cp.max(conec))
    n_canos = len(C)
    A = cp.zeros((n_nos, n_nos))
    D = cp.zeros((n_canos, n_nos))

    if n_canos != len(conec):
        raise ValueError("Valor Diferente")

    for k in range(n_canos):
        entrada = conec[k][0] - 1
        saida = conec[k][1] - 1
        D[k, entrada] = 1
        D[k, saida] = -1

    for k in range(n_canos):
        n1, n2 = conec[k]
        Ck = C[k]

        A[n1 - 1, n1 - 1] += Ck
        A[n1 - 1, n2 - 1] -= Ck
        A[n2 - 1, n1 - 1] -= Ck
        A[n2 - 1, n2 - 1] += Ck

    return A, D

def Solve(conec, C, natm, nb, qb):
    Atilde = Assembly(conec, C)[0]
    
    Atilde[natm, :] = cp.where(cp.arange(len(Atilde)) == natm, 1, 0)

    b = cp.zeros((len(Atilde), 1))
    b[nb] = qb

    pressure = cp.linalg.solve(Atilde, b)

    pressure_rounded = cp.around(pressure, decimals=4)

    return pressure_rounded

num, num2 = 20, 20

C = criarmatrizC(criarConecGrade(num, num2)[1])
conec = criarConecGrade(num, num2)[0]

solu = Solve(conec, C, num * num2 - 1, 0, 3)
print(solu)
