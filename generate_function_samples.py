import matplotlib.pyplot as plt
import numpy as np
import random
import numexpr as ne
import argparse

def enable_smart_axes(subplot):
	# set axis lines to be smart
	subplot.spines['left'].set_position('zero')
	subplot.spines['right'].set_color('none')
	subplot.spines['bottom'].set_position('zero')
	subplot.spines['top'].set_color('none')
	subplot.spines['left'].set_smart_bounds(True)
	subplot.spines['bottom'].set_smart_bounds(True)
	subplot.xaxis.set_ticks_position('bottom')
	subplot.yaxis.set_ticks_position('left')


parser = argparse.ArgumentParser()
parser.add_argument('--function', type = str, required = True, help = 'function to plot expressed in terms of variable x')
parser.add_argument('--plot_out_file', type = str, default = None, help = 'File to save plot to')
parser.add_argument('--samples_out_file', type = str, default = None, help = 'File to save samples (x,y) to (.npy)')
parser.add_argument('--jitter', type = float, default = 0.5, help = 'jitter range +/- [jitter]')
parser.add_argument('--num_samples', type = int, default = 15, help = 'number of samples to generate')
parser.add_argument('--xrange', nargs = 2, type = float, default = [-1, 1], help = 'x-axis range in format min max')
parser.add_argument('--draw_true_function', action='store_true')
parser.add_argument('--x_label', type = str, default = 'x')
parser.add_argument('--y_label', type = str, default = 'y')
parser.add_argument('--title', type = str, default = '')
parser.add_argument('--enable_smart_axes', action = 'store_true', help = 'Make axes lines cross (0,0)')

args = parser.parse_args()

jitter = args.jitter
plot_out_file = args.plot_out_file
samples_out_file = args.samples_out_file
num_samples = args.num_samples

x_min = float(args.xrange[0])
x_max = float(args.xrange[1])

f = args.function

#plot initial data
x = np.linspace(x_min, x_max, num_samples)
samples = ne.evaluate(f, local_dict = {'x': x})

for i in xrange(0, len(samples)):
	samples[i] = samples[i] + random.uniform(-jitter, jitter) 

margin = 0.25

fig = plt.figure()
subplot = fig.add_subplot(111)

subplot.set_xlabel(args.x_label)
subplot.set_ylabel(args.y_label)
subplot.set_title(args.title)

subplot.set_xlim([x_min, x_max])
subplot.set_ylim([min(samples), max(samples)])

if args.enable_smart_axes:
	enable_smart_axes(subplot)

subplot.grid()

subplot.plot(x, samples, 'bo', clip_on = False)

if args.draw_true_function:
	x_all = np.linspace(x_min, x_max, 100)
	y = ne.evaluate(f, local_dict = {'x' : x_all})
	subplot.plot(x_all, y, 'r-', alpha = 0.2)

if samples_out_file:
	print('saving samples to ' + samples_out_file)
	paired_samples = zip(x,samples)
	np.save(samples_out_file, paired_samples)

if plot_out_file:
	print('saving plot to ' + plot_out_file)
	plt.savefig(plot_out_file)
else:
	plt.show()
