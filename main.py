from typing import Optional, Tuple, Any
import numpy as np
import pickle


def generate_matrix(n: int,
                    str_range: int) -> np.ndarray:
    assert 0 <= n <= 3
    while True:
        matrix = np.random.randint(0, 500, size=(n, n))
        det = determinant(n, matrix)
        if det and modular_multi_inv(det, str_range):
            print('=' * 10, '생성된 key matrix', '=' * 10)
            print(matrix)
            return matrix


def determinant(n: int,
                matrix: np.ndarray) -> int:
    if n == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    elif n == 3:
        return (matrix[0, 0] * matrix[1, 1] * matrix[2, 2]
                + matrix[0, 1] * matrix[1, 2] * matrix[2, 0]
                + matrix[0, 2] * matrix[1, 0] * matrix[2, 1]) \
               - (matrix[0, 2] * matrix[1, 1] * matrix[2, 0]
                  + matrix[0, 0] * matrix[1, 2] * matrix[2, 1]
                  + matrix[0, 1] * matrix[1, 0] * matrix[2, 2])
    else:
        return matrix[0,0]


def input_str() -> str:
    return input('암호화 할 문자를 입력해주세요:')


def str_2_int_translate(target: str,
                        translater: dict) -> np.ndarray:
    res = np.zeros(len(target))
    for idx, s in enumerate(target):
        res[idx] = translater[s]
    return res


def int_2_str_translate(target: np.ndarray,
                        translater: dict) -> str:
    str_matrix = []
    for m in target:
        for s in m:
            str_matrix.append(translater[s])
    return ''.join(str_matrix)


def divide_str(target: np.ndarray,
               k: int) -> np.ndarray:
    str_len = len(target)
    if not str_len % k:
        return np.array(np.array_split(target, str_len // k))
    else:
        end = str_len // k * k
        res = np.array_split(target[:end], str_len // k)
        end_add = np.zeros(k)
        end_add[:str_len % k] = target[end:]
        return np.concatenate([res, end_add.reshape((-1, k))], axis=0)


def modular_multi_inv(d: int,
                      m: int) -> Optional[int]:
    if d > m:
        d = d % m
    for i in range(1, m):
        if (d % m) * (i % m) % m == 1:
            return i
    return None


def encrypto_hill(plaintext_matrix: np.ndarray,
                key_matrix: np.ndarray,
                str_range: int) -> np.ndarray:
    ciphertext_matrix = np.zeros(plaintext_matrix.shape)
    for idx, m in enumerate(plaintext_matrix):
        ciphertext_matrix[idx] = key_matrix @ m % str_range
    return ciphertext_matrix


def decrypto_hill(key_matrix: np.ndarray,
                  ciphertext_matrix: np.ndarray,
                  str_range: int,
                  k: int):
    inv_key = inverse_key_matrix(key_matrix, k, str_range)
    plaintext_matrix = np.zeros(ciphertext_matrix.shape)
    for idx, m in enumerate(ciphertext_matrix):
        plaintext_matrix[idx] = inv_key @ m % str_range
    return plaintext_matrix


def adjoint(matrix: np.ndarray,
            k: int) -> np.ndarray:
    adj_matrix = np.zeros(matrix.shape)
    for x in range(k):
        for y in range(k):
            det = np.delete(np.delete(matrix.copy(), x, 0), y, 1)
            adj_matrix[x, y] = (1 if (x + y) % 2 else -1) * determinant(k - 1, det)
    return adj_matrix.T


def inverse_key_matrix(key: np.ndarray,
                       k: int,
                       str_range: int) -> np.ndarray:
    return modular_multi_inv(determinant(k, key), str_range) * adjoint(key, k) % str_range


def load_translater() -> Tuple[dict, dict]:
    with open('str_to_int.pickle', 'rb') as f:
        str_to_int = pickle.load(f)
    with open('int_to_str.pickle', 'rb') as f:
        int_to_str = pickle.load(f)
    return str_to_int, int_to_str


if __name__ == '__main__':
    str_to_int, int_to_str = load_translater()
    str_range = len(str_to_int)
    while True:
        try:
            k = int(input('암오화에 사용할 matrix의 크기를 입력해 주세요.(2, 3만 가능): '))
            key_matrix = generate_matrix(k, str_range)
            break
        except ValueError as e:
            print("숫자를 입력해주세요!")
        except:
            print("2와 3만 가능합니다!")
    target_str = input_str()
    target_int = str_2_int_translate(target_str, str_to_int)
    plaintext_matrix = divide_str(target_int, k)
    ciphertext_matrix = encrypto_hill(plaintext_matrix, key_matrix, str_range)

    print(int_2_str_translate(ciphertext_matrix, int_to_str))
    print(int_2_str_translate(decrypto_hill(key_matrix, ciphertext_matrix, str_range, k), int_to_str))
