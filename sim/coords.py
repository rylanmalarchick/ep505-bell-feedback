"""Map between the density matrix rho (4 x 4) and the control state x in R^15.

The coordinate order comes from verification/coords_spec.py, the same file
that verification/check_control_model.py uses. The model matrices A0 and B_j
are computed here from their definition in the SME, not typed in:

    A0[a, b]  = Tr(P_a D(P_b)) / 4,   D(r) = -i omega [ZZ, r] + M (ZZ r ZZ - r)
    B_j[a, b] = Tr(P_a (-i [H_j, P_b])) / 4

All functions take a batch: rho has shape (N, 4, 4), x has shape (N, 15).
"""
import numpy as np

from verification.coords_spec import H_LAB, LAB, U0, XBAD, XBAR

from .params import Plant

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}


def pauli(label: str) -> np.ndarray:
    """Two-qubit Pauli product, qubit 1 first: pauli("XY") = X (x) Y."""
    return np.kron(_PAULI[label[0]], _PAULI[label[1]])


BASIS = np.stack([pauli(lab) for lab in LAB])  # (15, 4, 4)
H_J = np.stack([pauli(lab) for lab in H_LAB])  # (4, 4, 4)
ZZ = pauli("ZZ")
HALF = 0.5 * (pauli("ZI") + pauli("IZ"))  # diag(1, 0, 0, -1)
I4 = np.eye(4, dtype=complex)

X_BAR = np.array(XBAR, dtype=float)
X_BAD = np.array(XBAD, dtype=float)
U_0 = np.array(U0, dtype=float)

PHI_PLUS = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
PHI_MINUS = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
PSI_PLUS = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
PSI_MINUS = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)


def to_x(rho: np.ndarray) -> np.ndarray:
    """x_a = Tr(P_a rho). Shape (N, 4, 4) -> (N, 15)."""
    return np.einsum("aij,nji->na", BASIS, rho).real


def to_rho(x: np.ndarray) -> np.ndarray:
    """rho = (I + sum_a x_a P_a) / 4. Shape (N, 15) -> (N, 4, 4)."""
    return (I4 + np.einsum("na,aij->nij", x, BASIS)) / 4


def fidelity(rho: np.ndarray, psi: np.ndarray = PHI_PLUS) -> np.ndarray:
    """F = <psi| rho |psi>, Phi+ by default. Shape (N, 4, 4) -> (N,)."""
    return np.einsum("i,nij,j->n", psi.conj(), rho, psi).real


def xbar_of(psi: np.ndarray) -> np.ndarray:
    """Coordinates x of the pure state |psi><psi|."""
    return to_x(np.outer(psi, psi.conj())[None])[0]


def p1(rho: np.ndarray) -> np.ndarray:
    """p1 = Tr(ZZ rho), the parity. ZZ = diag(1, -1, -1, 1)."""
    d = np.einsum("nii->ni", rho).real
    return d[:, 0] - d[:, 1] - d[:, 2] + d[:, 3]


def _coord_matrix(op) -> np.ndarray:
    """Matrix of the linear map rho -> op(rho) in x coordinates."""
    return np.array([[np.trace(Pa @ op(Pb)).real / 4 for Pb in BASIS] for Pa in BASIS])


def a0_matrix(prm: Plant) -> np.ndarray:
    """Unforced drift matrix A0 (15 x 15) from the SME drift."""
    def drift(r):
        return -1j * prm.omega * (ZZ @ r - r @ ZZ) + prm.M * (ZZ @ r @ ZZ - r)
    return _coord_matrix(drift)


B_MATS = np.stack([_coord_matrix(lambda r, H=H: -1j * (H @ r - r @ H)) for H in H_J])  # (4, 15, 15)

def h_rows(xbar: np.ndarray) -> np.ndarray:
    """Row j is xbar^T B_j, so h(x) = x @ h_rows(xbar).T. Shape (4, 15)."""
    return np.einsum("a,jab->jb", xbar, B_MATS)


H_ROWS = h_rows(X_BAR)  # target Phi+


def h(x: np.ndarray, rows: np.ndarray = H_ROWS) -> np.ndarray:
    """Damping functions h_j(x) = xbar^T B_j x. Shape (N, 15) -> (N, 4)."""
    return x @ rows.T
