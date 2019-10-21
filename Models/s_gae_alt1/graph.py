import torch

class Graph:
    # templates size: (V, 4), 4 dims are: x, y, vx, vy
    def __init__(self, templates, vmax):
        V, C = templates.size()

        self.num_nodes = V
        self.s_kernel = 2
        self.A = torch.zeros(self.s_kernel, vmax, vmax)
        self.A.requires_grad = False

        self.edges(templates)

    
    def edges(self, templates):
        # s_kernel == 0
        self.A[0, :self.num_nodes, :self.num_nodes] = torce.eye(self.num_nodes)

        # s_kernel == 1
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                xi, yi, vxi, vyi = templates[i]
                xj, yj, vxj, vyj = templates[j]
                a, b, c, d = (xi-xj), (yi-yj), (vxi-vxj), (vyi-vyj)
                tmin = -(a*c+b*d)/(c**2+d**2).ceil().item()
                self.A[1, i, j] = 1.0 / tmin if tmin > 0.0 else 0.0
                self.A[1, j, i] = self.A[1, i, j]

        
    def normalize_undigraph(self, alpha=1e-3):
        DAD = torch.zeros(self.A.size())
        DAD.requires_grad = False

        for k in range(self.s_kernel):
            A_ = self.A[k, :self.num_nodes, :self.num_nodes]
            D1 = torch.sum(A_, 0)+alpha
            Dn = torch.zeros(self.num_nodes, self.num_nodes)
            for i in range(self.num_nodes):
                Dn[i, i] = D1[i]**(-0.5)
            DAD_ = Dn.mm(A_).mm(Dn)
            DAD[k] = DAD_
        
        return DAD


    def graph_pos_weight(self, alpha=1e-3):
        A_sum = self.A.sum()-1.0*self.num_nodes+alpha
        pos_weight = (self.num_nodes**2-A_sum)/A_sum

        return pos_weight


    def graph_norm(self, alpha=1e-3):
        A_sum = self.A.sum()-1.0*self.num_nodes+alpha
        norm = (self.num_nodes**2)/(2*(self.num_nodes**2-A_sum))

        return norm


    def graph_A(self):
        return self.A