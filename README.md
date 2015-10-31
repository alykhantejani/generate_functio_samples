# generate_function_samples
A python script to generate samples from a given function. For example, for the function (The Maclaurin expansion `sin(x)`), 
```
y = (2.5*x - (2.5*x^3)/6 + (2.5*x^5)/120)
``` 
We can plot 15 samples using this equation, with added jitter of `+/- 0.5`. We can also plot the function across the whole range without jitter on the same plot with the command:

```
python generate_function_samples.py --function '(2.5*x - (2.5*x**3)/6 + (2.5*x**5)/120)' --jitter 0.5 --samples 15 --draw_true_function

```

Which will produce the following plot:

![](https://github.com/alykhantejani/generate_function_samples/raw/master/src/example_output.png "Output")
