def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	import numpy as np
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	x=np.array(a)
	y=np.array(b)
	if x.shape[1]!=y.shape[0]:
		return -1
	return np.dot(a,b)
	