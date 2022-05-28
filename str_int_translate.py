import pickle

strs = [' ']

str_eng_c_start = ord('A')
str_eng_c_end = ord('Z')
for i in range(str_eng_c_start, str_eng_c_end + 1):
    strs.append(chr(i))

str_eng_s_start = ord('a')
str_eng_s_end = ord('z')
for i in range(str_eng_s_start, str_eng_s_end + 1):
    strs.append(chr(i))

str_kr_jamo_start = ord('ㄱ')
str_kr_jamo_end = 12686
for i in range(str_kr_jamo_start, str_kr_jamo_end + 1):
    strs.append(chr(i))

str_kr_start = ord('가')
str_kr_end = ord('힣')
for i in range(str_kr_start, str_kr_end + 1):
    strs.append(chr(i))

str_num_start = ord('0')
str_num_end = ord('9')
for i in range(str_num_start, str_num_end + 1):
    strs.append(chr(i))

str_to_int = dict()
int_to_str = dict()
for idx, s in enumerate(strs):
    str_to_int[s] = idx
    int_to_str[idx] = s

with open('str_to_int.pickle', 'wb') as f:
    pickle.dump(str_to_int, f, pickle.HIGHEST_PROTOCOL)
with open('int_to_str.pickle', 'wb') as f:
    pickle.dump(int_to_str, f, pickle.HIGHEST_PROTOCOL)