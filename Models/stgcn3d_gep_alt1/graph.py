import torch

class Graph:
    # batch_templates size: (N, V, 4), 4 dims are: x, y, vx, vy
    def __init__(self, batch_templates):
        N, V, C = batch_templates.size()

        self.s_kernel = 2
        self.A = torch.zeros(N, self.s_kernel, V, V)
        self.A.requires_grad = False

        self.edges(batch_templates)


    def edges(self, batch_templates):
        N = self.A.size(0)
        V = self.A.size(-1)

        # s_kernel == 0
        self.A[:, 0] = torch.eye(V).repeat(N, 1, 1)

        # s_kernel == 1
        for num in range(N):
            for i in range(V):
                for j in range(i+1, V):
                    xi, yi, vxi, vyi = batch_templates[num, i]
                    xj, yj, vxj, vyj = batch_templates[num, j]
                    a, b, c, d = (xi-xj), (yi-yj), (vxi-vxj), (vyi-vyj)
                    tmin = -(a*c+b*d)/(c**2+d**2).ceil().item()
                    self.A[num, 1, i, j] = 1.0 / tmin if tmin > 0.0 else 0.0
                    self.A[num, 1, j, i] = self.A[num, 1, i, j]
    

    def normalize_undigraph(self, alpha=1e-3):
        DADs = torch.zeros(self.A.size())
        DADs.requires_grad = False

        N = self.A.size(0)
        V = self.A.size(-1)

        for num in range(N):
            for k in range(self.s_kernel):
                A = self.A[num, k]
                D1 = torch.sum(A, 0)+alpha
                Dn = torch.zeros(V, V)
                for i in range(V):
                    Dn[i, i] = D1[i]**(-0.5)
                DAD = Dn.mm(A).mm(Dn)
                DADs[num, k] = DAD
        
        return DADs
