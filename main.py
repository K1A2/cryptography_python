import numpy as np

str_eng_c_start = ord('A')
str_eng_c_end = ord('Z')

str_eng_s_start = ord('a')
str_eng_s_end = ord('z')

str_kr_jamo_start = ord('ㄱ')
str_kr_jamo_end = 12686

str_kr_start = ord('가')
str_kr_end = ord('힣')

str_num_start = ord('0')
str_num_end = ord('9')

str_eng_c_range = str_eng_c_end - str_eng_c_start
str_eng_s_range = str_eng_s_end - str_eng_s_start
str_kr_jamo_range = str_kr_jamo_end - str_kr_jamo_start
str_kr_range = str_kr_end - str_kr_start
str_num_range = str_num_end - str_num_start

str_range = str_eng_c_range + str_eng_s_range + str_kr_jamo_range + str_kr_range + str_num_range

def generate_matrix(n=2) -> np.ndarray:
    assert 0 <= n <= 3
    while True:
        matrix = np.random.randint(0, str_range, size=(n, n))
        det = determinant(n, matrix)
        if det:
            print('='*10, '생성된 key matrix', '='*10)
            print(matrix)
            return matrix

def determinant(n: int,
                matrix: np.ndarray) -> int:
    if n == 2:
        return matrix[0,0] * matrix[1,1] - matrix[0,1] * matrix[1,0]
    elif n == 3:
        return (matrix[0,0] * matrix[1,1] * matrix[2,2]
                + matrix[0,1] * matrix[1,2] * matrix[2,0]
                + matrix[0,2] * matrix[1,0] * matrix[2,1])\
               - (matrix[0,2] * matrix[1,1] * matrix[2,0]
                  + matrix[0,0] * matrix[1,2] * matrix[2,1]
                  + matrix[0,1] * matrix[1,0] * matrix[2,2])

if __name__ == '__main__':
    while True:
        try:
            k = int(input('암오화에 사용할 matrix의 크기를 입력해 주세요.(2, 3만 가능): '))
            generate_matrix(k)
            break
        except ValueError as e:
            print("숫자를 입력해주세요!")
        except:
            print("2와 3만 가능합니다!")
