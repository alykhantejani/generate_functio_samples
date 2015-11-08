# generate_function_samples
A python script to generate samples from a given function. For example, for the function (The Maclaurin expansion `sin(x)`), 
```
y = (2.5*x - (2.5*x^3)/6 + (2.5*x^5)/120)
``` 
We can plot 15 samples using this equation, with added jitter of `+/- 0.5`. We can also plot the function across the whole range without jitter on the same plot with the command:

```
python generate_function_samples.py --function '(2.5*x - (2.5*x**3)/6 + (2.5*x**5)/120)' --jitter 0.5 --samples 15 --draw_true_function

```
which will produce the following plot:
<center><img src ="https://github.com/alykhantejani/generate_function_samples/blob/master/example_output.png"/></center>

We can also plot 3D samples. For example for the function 
```
y = sin((x^2))/2 - (x1^2)/4 + 3)cos(2x + 1 - e^x1)
```
we can plot 15 samples using the command below:
```
python generate_function_samples.py --function 'sin(0.5*x**2 - 0.25*x1**2 + 3) * cos(2*x + 1 - exp(x1))' --jitter 0.5 --num_samples 15 --draw_true_function --plot_out_file example_out2323put.png --variable_ranges '{"x": [-2.5, 2.5], "x1": [-3, 3]}'
```
which will produce the following plot:
<center><img src ="https://github.com/alykhantejani/generate_function_samples/blob/master/example_output_3d.png"/></center>
