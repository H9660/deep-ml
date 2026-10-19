import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	import numpy as np
	x=np.array(a).shape[0] * np.array(a).shape[1]
	y=new_shape[1] * new_shape[0]
	if not x==y:
		return []

	return np.array(a).reshape(new_shape)
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	