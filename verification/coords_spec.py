"""Coordinate order of the control model. One source for verification/ and sim/.

x = (p, q) in R^15 with x_a = Tr(P_a rho) and rho = (I + sum_a x_a P_a) / 4.
A label names one Pauli per qubit, qubit 1 first: "XY" = sigma_x (x) sigma_y.
The p labels commute with ZZ. The q labels anticommute with ZZ.
"""
P_LAB = ("ZZ", "IZ", "ZI", "XX", "YY", "XY", "YX")
Q_LAB = ("XI", "YZ", "XZ", "YI", "IX", "ZY", "ZX", "IY")
LAB = P_LAB + Q_LAB

# Input Hamiltonians H_1 .. H_4 (local drives).
H_LAB = ("XI", "YI", "IX", "IY")

# Target Phi+ and the bad equilibrium Phi- in x coordinates.
XBAR = (1, 0, 0, 1, -1, 0, 0) + (0,) * 8
XBAD = (1, 0, 0, -1, 1, 0, 0) + (0,) * 8

# Constant offset of the offset law, u0 = (1, 0, -1, 0).
U0 = (1, 0, -1, 0)
