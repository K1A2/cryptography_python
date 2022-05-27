import numpy as np

str_eng_start = ord('A')
str_kr_start = ord('가')

def generate_matrix(n=2):
    assert 0 <= n <= 3
    while True:
        matrix = np.random.randint(0, 100, size=(n, n))
        det = determinant(n, matrix)
        if det:
            print('='*10, '생성된 key matrix', '='*10)
            print(matrix)
            break

def determinant(n: int,
                matrix: np.ndarray):
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
