import time

st = time.time()

sum = 0
for i in range(20000):
    sum += i

en = time.time()

print(st)
print(en)
print(en-st)