import numpy as np

class HammingCode:
    def __init__(self, m):
        self.m = m
        self.n = 2**m - 1
        self.k = 2**m - m - 1

    def encode(self, message):
        G = self._generate_matrix()
        return np.dot(message, G) % 2

    def decode(self, received):
        H = self._parity_check_matrix()
        syndrome = np.dot(received, H.T) % 2
        if np.sum(syndrome) == 0:
            return received[self.m:]
        error_pos = np.sum(syndrome * (2**np.arange(self.m)))
        corrected = np.copy(received)
        corrected[error_pos-1] = 1 - corrected[error_pos-1]
        return corrected[self.m:]

    def _generate_matrix(self):
        I = np.eye(self.k, dtype=int)
        P = np.zeros((self.k, self.m), dtype=int)
        for i in range(self.k):
            for j in range(self.m):
                P[i, j] = (i+1) & (1 << j) != 0
        return np.hstack((P, I))

    def _parity_check_matrix(self):
        G = self._generate_matrix()
        P = G[:, :self.m]
        I = np.eye(self.m, dtype=int)
        return np.hstack((I, P.T))

def repetition_code_encode(message, n):
    return np.repeat(message, n)

def repetition_code_decode(received, n):
    return np.round(np.mean(received.reshape(-1, n), axis=1)).astype(int)

# Example usage
if __name__ == "__main__":
    # Hamming code example
    hamming = HammingCode(3)
    message = np.array([1, 0, 1, 1])
    encoded = hamming.encode(message)
    print("Encoded message:", encoded)
    received = encoded.copy()
    received[2] = 1 - received[2]  # Introduce an error
    decoded = hamming.decode(received)
    print("Decoded message:", decoded)

    # Repetition code example
    message = np.array([1, 0, 1, 1])
    encoded = repetition_code_encode(message, 3)
    print("Repetition encoded:", encoded)
    received = encoded.copy()
    received[1] = 1 - received[1]  # Introduce an error
    decoded = repetition_code_decode(received, 3)
    print("Repetition decoded:", decoded)
