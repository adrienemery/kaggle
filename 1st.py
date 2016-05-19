import random
import sys
import os
import matplotlib.pyplot as plt
import math
import time

print('Karatsuba multiplication')
x=input("first_number=")
y=input("second_number=")


def karatsuba(x,y):

  len_x=len(str(x))
  len_y=len(str(y))

  if (int(len_x) == 1 or int(len_y) == 1):
      return int(x) * int(y)
  else:
      n1 = int(math.ceil(len_x / 2.0))
      n2 = int(math.ceil(len_y / 2.0))

      if (n1 < n2):
          n = n1
      else:
          n = n2

      m1 = len_x - n
      m2 = len_y - n
      a = karatsuba(int(x[0:m1]), int(y[0:m2]))
      c = karatsuba(int(x[m1:len_x]), int(y[m2:len_y]))
      b = karatsuba(int(x[0:m1]) + int(x[m1:len_x]), int(y[0:m2]) + int(y[m2:len_y])) - a - c
      results = a * math.pow(10, 2 * n) + b * math.pow(10, n) + c
      return int(results)
