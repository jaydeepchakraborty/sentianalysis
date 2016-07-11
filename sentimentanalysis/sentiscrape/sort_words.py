file_name = "pos.txt"
# file_name = "neg.txt"

arrays = []
with open(file_name, "r", encoding='utf-8') as f:
    for line in f:
        arrays.append(line)
f.close()

arrays.sort()


file = open(file_name, "w", encoding='utf-8') 

for arr in arrays:
    file.write(arr)
    

file.close()
