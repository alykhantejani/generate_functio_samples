import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import numexpr as ne
import argparse
import json
import math
import random

def plot_2d(variable_ranges, f, num_samples, jitter = 0, draw_true_function = False, alpha = 0.35):
	fig = plt.figure()
	subplot = fig.add_subplot(111)

	samples_ranges = {}
	key = variable_ranges.keys()[0]
	
	range_to_sample = np.linspace(variable_ranges[key][0], variable_ranges[key][1], num_samples * 5)
	samples_ranges[key] = random.sample(set(range_to_sample), num_samples)
	#plot initial data
	samples = ne.evaluate(f, local_dict = samples_ranges)
	
	if jitter != 0:
		for i in xrange(0, len(samples)):
			samples[i] = samples[i] + random.uniform(-jitter, jitter) 

	subplot.set_xlim([min(samples_ranges[key]), max(samples_ranges[key])])
	subplot.set_ylim([min(samples), max(samples)])

	subplot.set_xlabel(key)
	subplot.plot(samples_ranges[key], samples, 'bo', clip_on = False)

	if draw_true_function:
		true_eval_ranges = {key: np.linspace(variable_ranges[key][0], variable_ranges[key][1], 100)}
		true_values = ne.evaluate(f, local_dict = true_eval_ranges)
		subplot.set_xlim([min(min(true_eval_ranges[key]), subplot.get_xlim()[0]), max(max(true_eval_ranges[key]), subplot.get_xlim()[1])])
		subplot.set_ylim([min(min(true_values), subplot.get_ylim()[0]) ,max(max(true_values), subplot.get_ylim()[1])])
		subplot.plot(true_eval_ranges[true_eval_ranges.keys()[0]], true_values, 'r-', alpha = alpha)

	return subplot, samples


def plot_3d(variable_ranges, f, num_samples, jitter = 0, draw_true_function = False, alpha = 0.35, cmap = 'gist_rainbow_r'):
	assert (math.sqrt(num_samples)).is_integer(), "Please ensure sqrt(num_samples) is an integer"

	fig = plt.figure()
	subplot = fig.add_subplot(111, projection = '3d')
	
	dimension_step = int(math.sqrt(num_samples))
	
	x_key = variable_ranges.keys()[0]
	x1_key = variable_ranges.keys()[1]
	x_ranges = random.sample(set(np.linspace(variable_ranges[x_key][0], variable_ranges[x_key][1], dimension_step * 3)), dimension_step)
	x1_ranges = random.sample(set(np.linspace(variable_ranges[x1_key][0], variable_ranges[x1_key][1], dimension_step * 3)), dimension_step)

	samples_ranges = {x_key : [], x1_key: []}

	for i in range(0, dimension_step):
		for j in range(0, dimension_step):
			samples_ranges[x_key].append(x_ranges[i])
			samples_ranges[x1_key].append(x1_ranges[j])


	samples = ne.evaluate(f, local_dict = samples_ranges)

	if jitter != 0 :
		for i in xrange(0, len(samples)):
			samples[i] = samples[i] + random.uniform(-jitter, jitter) 
	
	subplot.set_xlabel(x_key)
	subplot.set_ylabel(x1_key)

	subplot.set_xlim([min(variable_ranges[x_key]), max(variable_ranges[x_key])])
	subplot.set_ylim([min(variable_ranges[x1_key]), max(variable_ranges[x1_key])])
	subplot.set_zlim([min(samples), max(samples)])

	subplot.plot(samples_ranges[x_key], samples_ranges[x1_key], samples, 'bo', clip_on = False)

	if draw_true_function:
		x = x1 = None

		low = min(min(variable_ranges[x_key]), min(variable_ranges[x1_key]))
		high = max(max(variable_ranges[x_key]), max(variable_ranges[x1_key]))
		x = x1 = np.arange(low, high, 0.05)
		X, X1 = np.meshgrid(x, x1)

		error = np.array([ne.evaluate(f, local_dict = {x_key : a, x1_key: b}) for a, b in zip(np.ravel(X), np.ravel(X1))])
		Error = error.reshape(X.shape)

		subplot.plot_surface(X, X1, Error, cmap = cmap, alpha = alpha)


	return subplot, samples



parser = argparse.ArgumentParser()
parser.add_argument('--function', type = str, required = True, help = 'function to plot expressed in terms of variables in variable map')
parser.add_argument('--variable_ranges', type = json.loads, required = True, help = 'range of variables in function i.e. \'{ "x1" : [-1, 1], "x2" : [-2, 2] }\'')
parser.add_argument('--function_output_name', type = str, default = 'y', help = 'name to display for output of function')
parser.add_argument('--plot_out_file', type = str, default = None, help = 'File to save plot to')
parser.add_argument('--samples_out_file', type = str, default = None, help = 'File to save samples (y,x) to (.npy)')
parser.add_argument('--jitter', type = float, default = 0.5, help = 'jitter range +/- [jitter]')
parser.add_argument('--num_samples', type = int, default = 15, help = 'number of samples to generate')
parser.add_argument('--draw_true_function', action='store_true')
parser.add_argument('--alpha', type = float, default = 0.35, help = 'Alpha for plotting true graph (0 = transparent, 1 = opaque)')
parser.add_argument('--cmap', type = str, default = 'gist_rainbow_r', help = 'cmap to use for 3d surfaces')
parser.add_argument('--title', type = str, default = '')
parser.add_argument('--display', action = 'store_true', help = 'Display live interactive visualization')


args = parser.parse_args()

jitter = args.jitter
plot_out_file = args.plot_out_file
samples_out_file = args.samples_out_file
num_samples = args.num_samples
variable_ranges = args.variable_ranges

assert len(variable_ranges) > 0 and len(variable_ranges) <= 2, 'Can only plot 2D or 3D graphs'

f = args.function

subplot, samples = None, []

if len(variable_ranges) == 1:
	subplot, samples = plot_2d(variable_ranges, f, num_samples, jitter = jitter, draw_true_function = args.draw_true_function, alpha = args.alpha)
	subplot.set_ylabel(args.function_output_name)
else:
	subplot, samples = plot_3d(variable_ranges, f, num_samples, jitter = jitter, draw_true_function = args.draw_true_function, alpha = args.alpha)
	subplot.set_zlabel(args.function_output_name)

subplot.set_title(args.title)
subplot.grid()

if samples_out_file is not None:
	print('saving samples to ' + samples_out_file)
	np.save(samples_out_file, samples)

if plot_out_file is not None:
	print('saving plot to ' + plot_out_file)
	plt.savefig(plot_out_file)

if args.display:
	plt.show()