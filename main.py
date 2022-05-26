import numpy

str_eng_start = ord('A')
str_kr_start = ord('가')

def generate_matrix(n=2):
    assert 0 <= n <= 3
    while True:
        matrix = numpy.random.randint(0, 100, size=(n, n))
        print(numpy.linalg.det(matrix))

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
