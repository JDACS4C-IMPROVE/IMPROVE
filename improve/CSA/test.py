import os
import numpy as np
import math
import sys

print('Enter an integer')
n=int(sys.stdin.readline())
if n<=0:
    raise Exception("Error! Enter integers greater than or equal to 1")
print ('Enter a line')
line=sys.stdin.readline()
for i in range(n):
    print(line)
