import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import numexpr as ne
import argparse
import json

def plot_2d(samples_ranges, sample_values, true_values, true_values_ranges):
	fig = plt.figure()
	
	keys = samples_ranges.keys()

	subplot = fig.add_subplot(111)

	subplot.set_xlabel(keys[0])

	subplot.set_xlim([min(samples_ranges[keys[0]]), max(samples_ranges[keys[0]])])
	subplot.set_ylim([min(samples), max(samples)])

	subplot.plot(samples_ranges[keys[0]], samples, 'bo', clip_on = False)

	if true_values is  not None:
			subplot.set_xlim([min(min(true_values_ranges[keys[0]]), subplot.get_xlim()[0]), max(max(true_values_ranges[keys[0]]), subplot.get_xlim()[1])])
			subplot.set_ylim([min(min(true_values), subplot.get_ylim()[0]) ,max(max(true_values), subplot.get_ylim()[1])])
			subplot.plot(true_values_ranges[true_values_ranges.keys()[0]], true_values, 'r-', alpha = 0.2)

	return subplot

def plot_3d(samples_ranges, sample_values, f):
	fig = plt.figure()
	subplot = fig.add_subplot(111, projection = '3d')
	
	keys = samples_ranges.keys()

	subplot.set_xlabel(keys[0])
	subplot.set_ylabel(keys[1])

	subplot.set_xlim([min(samples_ranges[keys[0]]), max(samples_ranges[keys[0]])])
	subplot.set_ylim([min(samples_ranges[keys[0]]), max(samples_ranges[keys[0]])])
	subplot.set_zlim([min(samples), max(samples)])

	subplot.plot(samples_ranges[keys[0]], samples_ranges[keys[1]], samples, 'bo', clip_on = False)

	if f is not None:
		x = x1 = None

		low = min(min(samples_ranges[keys[0]]), min(samples_ranges[keys[1]]))
		high = max(max(samples_ranges[keys[0]]), max(samples_ranges[keys[1]]))
		x = x1 = np.arange(low, high, 0.05)

		X, X1 = np.meshgrid(x, x1)

		error = np.array([ne.evaluate(f, local_dict = {keys[0] : a, keys[1]: b}) for a, b in zip(np.ravel(X), np.ravel(X1))])
		Error = error.reshape(X.shape)

		subplot.plot_surface(X, X1, Error, cmap = 'gist_rainbow_r', alpha = 0.2)

	return subplot

parser = argparse.ArgumentParser()
parser.add_argument('--function', type = str, required = True, help = 'function to plot expressed in terms of variables in variable map')
parser.add_argument('--variable_ranges', type = json.loads, required = True, help = 'range of variables in function i.e. \'{ "x1" : [-1, 1], "x2" : [-2, 2] }\'')
parser.add_argument('--function_output_name', type = str, default = 'y', help = 'name to display for output of function')
parser.add_argument('--plot_out_file', type = str, default = None, help = 'File to save plot to')
parser.add_argument('--samples_out_file', type = str, default = None, help = 'File to save samples (y,x) to (.npy)')
parser.add_argument('--jitter', type = float, default = 0.5, help = 'jitter range +/- [jitter]')
parser.add_argument('--num_samples', type = int, default = 15, help = 'number of samples to generate')
parser.add_argument('--draw_true_function', action='store_true')
parser.add_argument('--title', type = str, default = '')
parser.add_argument('--display', action = 'store_true', help = 'Display live interactive visualization')

args = parser.parse_args()

jitter = args.jitter
plot_out_file = args.plot_out_file
samples_out_file = args.samples_out_file
num_samples = args.num_samples
variable_ranges = args.variable_ranges

assert len(variable_ranges) > 0 and len(variable_ranges) <= 2, 'Can only plot 2D or 3D graphs'
plot2d = len(variable_ranges) == 1

samples_eval_ranges = {}
for key in variable_ranges:
	samples_eval_ranges[key] = np.linspace(variable_ranges[key][0], variable_ranges[key][1], num_samples)

f = args.function
#plot initial data
samples = ne.evaluate(f, local_dict = samples_eval_ranges)

for i in xrange(0, len(samples)):
	samples[i] = samples[i] + random.uniform(-jitter, jitter) 

subplot = None
if plot2d:
	true_values = None
	true_eval_ranges = None
	if args.draw_true_function:
		true_eval_ranges = {key: np.linspace(variable_ranges[key][0], variable_ranges[key][1], 100)}
		true_values = ne.evaluate(f, local_dict = true_eval_ranges)

	subplot = plot_2d(samples_eval_ranges, samples, true_values, true_eval_ranges)
	subplot.set_ylabel(args.function_output_name)
else:
	subplot = plot_3d(samples_eval_ranges, samples, f)
	subplot.set_zlabel(args.function_output_name)

subplot.set_title(args.title)

subplot.grid()

if samples_out_file:
	print('saving samples to ' + samples_out_file)
	out = {'out_values': samples}
	for k in samples_eval_ranges:
		out[k] = samples_eval_ranges[k]

	np.save(samples_out_file, out)

if plot_out_file:
	print('saving plot to ' + plot_out_file)
	plt.savefig(plot_out_file)

if args.display:
	plt.show()