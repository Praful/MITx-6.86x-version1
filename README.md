# MITx-6.86x-version1
Shareable files for the MIT course [Machine Learning with Python - From Linear Models to Deep Learning](https://courses.edx.org/courses/course-v1:MITx+6.86x+1T2020/course/). The course took place from Feb to May 2020.

## Project 4: Collaborative Filtering via Gaussian Mixtures

### 3. Expectation–maximization algorithm

The tests in `test_3_em_algorithm.py` use input and output from my submissions on the course for this section. These are used to check your results. Tests exist for the original toy dataset, and the `estep`, `mstep`, and `run` functions.

I've also added tests for the BIC function, which is in part 5.

I tested and developed these on Windows 10. Obviously, I can't guarantee that these 
will work on your computer.

Instructions

1. Put all files in the same folder (`netflix`) as `naive_em.py`. 
2. To run, enter in a console:
```
python test_3_em_algorithm.py
```

The other files (`*test_input*.py`) contain the input and output test data. 

If you don't want to run a particular test, you can either comment out the whole test 
or add the following above the function definition:
```
@unittest.skip('Skip this test')
```
