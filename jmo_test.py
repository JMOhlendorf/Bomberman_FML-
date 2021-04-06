import numpy as np
from scipy.special import softmax

a = (2, 2)
a = np.array(a)
b = (4, 2)
b = np.array(b)
print( type(b) == np.ndarray)
dist = np.linalg.norm(a - b)
print(dist)

my_arr = np.array([45])
"""
field = np.ones((5,5))
coins_pos = np.array([[0,0], [0,2], [3,4], [1,2]])
print(coins_pos.shape)
coins_x = coins_pos[:, 0]
coins_y = coins_pos[:, 1]
print(coins_x)
coin_field = np.zeros_like(field)
coin_field[coins_x, coins_y] = 2
print(coin_field)
"""

"""
length = 8
scan_length = 1
field = np.zeros((length,length))
field[1,1] = 3
added_tiles = scan_length - 1
new_shape = length + 2 * added_tiles
new_field = -1 * np.ones((new_shape, new_shape))
# filling the new_field with original tile values
new_field[added_tiles: length + added_tiles, added_tiles: length + added_tiles] = field
print(new_field)

x = 1 + added_tiles
y = 1 + added_tiles
surrounding = []
for j in range(y - scan_length, y + (scan_length + 1)):
    for i in range(x - scan_length , x + (scan_length + 1)):
        surrounding.append(new_field[i, j])
print(surrounding)
"""
"""
a = np.array([2, 4, 7, 1, 8])
prob = softmax(a)
print(np.sum(prob))
rho = 2
prob1 = softmax(a/rho)
print(np.sum(prob1))
"""
"""
arr = np.ones((3,3))
patch = 2
add = int(patch / 2)
add_total = add * 2
newshape = arr.shape[0] + add_total
a = np.zeros((newshape, newshape))
a[add: 3 + add, add: 3 + add] = arr
print(a)
"""