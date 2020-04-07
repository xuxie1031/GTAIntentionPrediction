import torch
import numpy as np

class Graph:
	def __init__(self, templates):
		V, C = templates.size()

		self.s_kernel = 2
		self.A = torch.zeros(self.s_kernel, V, V)
		self.A.requires_grad = False

		self.edges(templates)


	def edges(self):
		V = self.A.size(-1)

		# s_kernel == 0
		self.A[:, 0] = torch.eye(V)

		# s_kernel == 1
		for i in range(V):
			for j in range(i+1, V):
				xi, yi, vxi, vyi = templates[i]
				xj, yj, vxj, vyj = templates[j]
				a, b, c, d = (xi-xj), (yi-yj), (vxi-vxj), (vyi-vyj)
				tmin = -(a*c+b*d)/(c**2+d**2).ceil().item()
				self.A[1, i, j] = 1.0 / tmin if tmin > 0.0 else 0.0
				self.A[1, j, i] = self.A[1, i, j]


	def normalize_undigraph(self, alpha=1e-3):
		DADs = torch.zeros(self.A.size())
		DADs.requires_grad = False

		V = self.A.size(-1)

		for k in range(self.s_kernel):
			A = self.A[k]
			D1 = torch.sum(A, 0)+alpha
			Dn = torch.zeros(V, V)
			for i in range(V):
				D[i, i] = D1[i]**(-0.5)
			DAD = Dn.mm(A).mm(Dn)

		return DAD