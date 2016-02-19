import matplotlib.pyplot as plt
import sys
import re

_input = map(float, re.sub('[\s+]', '', sys.stdin.readline()).split(':')[::-1])
print _input

plt.plot(_input)
plt.show()

