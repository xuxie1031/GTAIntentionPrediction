import numpy as np
import datetime
import os

def within_time_range(h, m, hmin=7, mmin=50, hmax=8, mmax=35):
	if h >= hmin and h <= hmax:
		if h == hmin:
			if m >= mmin:
				return True
			else:
				return False
		elif h == hmax:
			if m < mmax:
				return True
			else:
				return False
		else:
			return True
	return False


file_path = os.path.join('TrajDset', 'NGSIM')

for name in os.listdir(file_path):
	file_name = os.path.join(file_path, name)
	if os.path.isfile(file_name):
		break

raw_data = np.loadtxt(file_name, delimiter=',', skiprows=1)

converted_data = []
for line in raw_data:
	time_stamp = int(line[3])
	h = datetime.datetime.fromtimestamp(time_stamp/1000).hour
	m = datetime.datetime.fromtimestamp(time_stamp/1000).minute
	if within_time_range(h, m):
		converted_data.append(line)

converted_name = 'NGSIM_simple.txt'
np.savetxt(converted_name, converted_data, delimiter=',')