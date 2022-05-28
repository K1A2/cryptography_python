from typing import Optional, Tuple
import numpy as np
import pickle
import re

np.set_printoptions(threshold=5)


def load_translater() -> Tuple[dict, dict]:
    '''
    str_int_translate.py에서 생성된 딕셔너리 로드

    :return: 숫자와 문자간 변환을 위한 딕셔너리 리턴
    '''
    with open('str_to_int.pickle', 'rb') as f:
        str_to_int = pickle.load(f)
    with open('int_to_str.pickle', 'rb') as f:
        int_to_str = pickle.load(f)
    return str_to_int, int_to_str


def generate_matrix(n: int,
                    str_range: int) -> np.ndarray:
    '''
    Key Matrix 생성

    :param n: Key Matrix의 크기
    :param str_range: 변환 가능한 모든 문자의 개수
    :return: Key Matrix 리턴
    '''
    assert 2 <= n <= 3
    while True:
        matrix = np.random.randint(0, 500, size=(n, n))
        det = determinant(n, matrix)
        if det and modular_multi_inv(det, str_range):
            print('=' * 10, '생성된 key matrix', '=' * 10)
            print(matrix)
            return matrix


def determinant(n: int,
                matrix: np.ndarray) -> int:
    '''
    행렬의 determinant를 구하는 함수

    :param n: matrix의 크기
    :param matrix: determinant를 구할 matrix
    :return: determinant 값
    '''
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


def modular_multi_inv(d: int,
                      m: int) -> Optional[int]:
    '''
    Modular Multiplicative Inverse 연산하는 함수

    :param d: determinant  값
    :param m: 문자의 범위
    :return: Modular Multiplicative Inverse 연산 결과 값
    '''
    if d > m:
        d %= m
    for i in range(1, m):
        if (d % m) * (i % m) % m == 1:
            return i
    return None


def input_str() -> str:
    '''
    암호화 할 문자를 입력 받아 가능한 문자만 있는지 검사하고 리턴하는 함수

    :return: 암호화 할 문자열
    '''
    print('='*10, '암호화 할 문자를 입력해주세요.', '='*10)
    print('가능한 문자: 알파벳 대소문자, 모든 한글, 숫자, 공백')
    o = re.compile('[^a-zA-Z0-9가-힣ㄱ-ㆎ ]')
    while True:
        strs = input('문자 입력: ')
        if o.search(strs):
            print('가능한 문자만 입력해 주세요!')
        else:
            return strs


def str_2_int_translate(target: str,
                        translater: dict) -> np.ndarray:
    '''
    문자를 숫자로 변환하는 함수

    :param target: 변환할 문자열
    :param translater: 문자와 숫자를 1:1 매칭한 딕셔너리
    :return: 변환된 숫자 Matrix
    '''
    print('='*10, '입력된 문자열 숫자로 변환','='*10)
    res = np.zeros(len(target))
    for idx, s in enumerate(target):
        res[idx] = translater[s]
    print('입력된 문자열:', target)
    print('변환된 문자열:', res)
    return res


def int_2_str_translate(target: np.ndarray,
                        translater: dict):
    '''
        숫자를 문자로 변환하는 함수

        :param target: 변환할 숫자 Matrix
        :param translater: 문자와 숫자를 1:1 매칭한 딕셔너리
    '''
    print('='*10, '암호화 된 matrix 문자열로 변환','='*10)
    str_matrix = []
    for m in target:
        for s in m:
            str_matrix.append(translater[s])
    print('변환된 문자열:', ''.join(str_matrix))


def divide_str(target: np.ndarray,
               k: int) -> np.ndarray:
    '''
    문자열을 k개로 나누어 주는 함수

    :param target: 나눌 문자열
    :param k: 나눌 개수
    :return: 나눈 Matrix
    '''
    str_len = len(target)
    print('='*10, f'문자열을 {str_len // k + (1 if str_len % k else 0)}x{k} matrix로 만들기','='*10)
    res = None
    if not str_len % k:
        res = np.array(np.array_split(target, str_len // k))
    else:
        end = str_len // k * k
        if end:
            res = np.array_split(target[:end], str_len // k)
            end_add = np.zeros(k)
            end_add[:str_len % k] = target[end:]
            res = np.concatenate([res, end_add.reshape((-1, k))], axis=0)
        else:
            res = np.zeros(k)
            res[:str_len % k] = target
            res = res.reshape((-1, k))
    print('변환된 matrix')
    print(res)
    return res


def encrypto_hill(plaintext_matrix: np.ndarray,
                key_matrix: np.ndarray,
                str_range: int) -> np.ndarray:
    '''
    문자 Matrix를 Hill 암호화 하는 함수

    :param plaintext_matrix: 암호화 할 문자 Matrix
    :param key_matrix: Key Natrix
    :param str_range: 변환 가능한 문자의 개수
    :return: 암호화 된 숫자 Matrix
    '''
    print('='*10,'암호화 된 matrix','='*10)
    ciphertext_matrix = np.zeros(plaintext_matrix.shape)
    for idx, m in enumerate(plaintext_matrix):
        ciphertext_matrix[idx] = key_matrix @ m % str_range
    print(ciphertext_matrix)
    return ciphertext_matrix


def decrypto_hill(key_matrix: np.ndarray,
                  ciphertext_matrix: np.ndarray,
                  str_range: int,
                  k: int):
    '''
    암호화 된 Matrix를 복호화 하는 함수

    :param key_matrix: Key Matrix
    :param ciphertext_matrix: 암호화 된 숫자 Matrix
    :param str_range: 변환 가능한 문자의 개수
    :param k: 한 덩어리의 개수
    :return: 복호화 된 Matrix
    '''
    print('=' * 10, '복호화 하기', '=' * 10)
    inv_key = inverse_key_matrix(key_matrix, k, str_range)
    plaintext_matrix = np.zeros(ciphertext_matrix.shape)
    for idx, m in enumerate(ciphertext_matrix):
        plaintext_matrix[idx] = inv_key @ m % str_range
    print('복호화 된 matrix:')
    print(plaintext_matrix)
    return plaintext_matrix


def adjoint(matrix: np.ndarray,
            k: int) -> np.ndarray:
    '''
    Matrix의 Adjoint를 구하는 함수

    :param matrix: Adjoint를 구할 Matrix
    :param k: Matrix의 크기
    :return: Matrix의 Adjoint
    '''
    adj_matrix = np.zeros(matrix.shape)
    for x in range(k):
        for y in range(k):
            det = np.delete(np.delete(matrix.copy(), x, 0), y, 1)
            adj_matrix[x, y] = (-1 if (x + y) % 2 else 1) * determinant(k - 1, det)
    return adj_matrix.T


def inverse_key_matrix(key_matrix: np.ndarray,
                       k: int,
                       str_range: int) -> np.ndarray:
    '''
    Inverse Key Matrix를 구하는 함수

    :param key_matrix: Key Matrix
    :param k: Key Matrix의 크기
    :param str_range: 변환 가능한 문자의 개수
    :return: Inverse Key Matrix
    '''
    return modular_multi_inv(determinant(k, key_matrix), str_range) * adjoint(key_matrix, k) % str_range


if __name__ == '__main__':
    # 문자 숫자 변환을 위한 딕셔너리 로딩
    str_to_int, int_to_str = load_translater()
    str_range = len(str_to_int)
    # Key Matrix의 크기 입력 후 랜덤으로 생성
    while True:
        try:
            k = int(input('암오화에 사용할 matrix의 크기를 입력해 주세요.(2, 3만 가능): '))
            key_matrix = generate_matrix(k, str_range)
            break
        # 2~3 이외의 값이 입력되거나 숫자가 아니면 다시 입력
        except ValueError as e:
            print("숫자를 입력해주세요!")
        except:
            print("2와 3만 가능합니다!")
    # 암호화 할 문자열 입력
    target_str = input_str()
    # 입력받은 문자열 숫자로 변환
    target_int = str_2_int_translate(target_str, str_to_int)
    # 숫자로 변환한 행렬을 k개로 나눔
    plaintext_matrix = divide_str(target_int, k)
    # 나눈 행렬을 암호화
    ciphertext_matrix = encrypto_hill(plaintext_matrix, key_matrix, str_range)
    # 암호화 된 행렬을 문자로 출력
    int_2_str_translate(ciphertext_matrix, int_to_str)

    # 암호화 된 행렬을 다시 복호화
    plaintext_matrix = decrypto_hill(key_matrix, ciphertext_matrix, str_range, k)
    # 복호화 된 행렬을 문자로 출력
    int_2_str_translate(plaintext_matrix, int_to_str)
