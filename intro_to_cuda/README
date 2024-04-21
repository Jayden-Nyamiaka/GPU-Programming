CS 179: GPU Computing
Lab 1: Introduction to CUDA
Name: Jayden Nyamiaka

================================================================================
Question 1: Common Errors (20 points)
================================================================================

--------------------------------------------------------------------------------
1.1
--------------------------------------------------------------------------------
Issue: 
int *a = 3;
It attempts to assign an int to an int pointer when it should be assigning an 
int address.

Fix: 
Replace 
    int *a = 3;
with
    int three = 3;
    int *a = &three;

--------------------------------------------------------------------------------
1.2
--------------------------------------------------------------------------------
Issue: 
int *a, b;
Variable a is an int pointer while variable b is an int, but it's treating b 
like an int pointer. It seems it thought the * would apply to both a and b.

Fix:
Replace 
    int *a, b;
with 
    int *a, *b;

--------------------------------------------------------------------------------
1.3
--------------------------------------------------------------------------------
Issue: 
int i, *a = (int *) malloc(1000);
For int pointer a, 1000 bytes are allocated when space is needed for 1000 ints.

Fix:
Replace
    int i, *a = (int *) malloc(1000);
with
    int i, *a = (int *) malloc(1000 * sizeof(int));

--------------------------------------------------------------------------------
1.4
--------------------------------------------------------------------------------
Issue: 
int **a = (int **) malloc(3 * sizeof (int *));
The issue isn't really in the above line but with the 2D array creation. To 
declare a 2D array using a double pointer, the memory for each row also must be 
allocated.

Fix:
After
    int **a = (int **) malloc(3 * sizeof (int *));
add lines
    for (int i = 0; i < 3; i++) {
      a[i] = (int *) malloc(100 * sizeof(int));
    }
    
--------------------------------------------------------------------------------
1.5
--------------------------------------------------------------------------------
Issue:
if (!a)
The if statement condition above should evaluate to true if *a == 0, but a is 
not being dereferenced to get the value stored at *a. With !a, the condition 
evaliates to true as long as the pointer is not NULL, which is every run 
because of scanf.

Fix:
Replace
    if (!a)
with 
    if (!*a)
    

================================================================================
Question 2: Parallelization (30 points)
================================================================================

--------------------------------------------------------------------------------
2.1
--------------------------------------------------------------------------------
Given an input signal x[n], suppose we have two output signals y_1[n] and
y_2[n], given by the difference equations: 
		y_1[n] = x[n - 1] + x[n] + x[n + 1]
		y_2[n] = y_2[n - 2] + y_2[n - 1] + x[n]

Which calculation do you expect will have an easier and faster implementation on
the GPU, and why?

The calculation for y_1[n] is easier and faster to implement on the GPU because 
it is more parallelizable. y_1[n] only relies on accessing the input signals
x[n], so the calculation for y_1[n] can done completely in parallel. However, 
y_2[n] requires the previous calculations of y_2[n - 2] and y_2[n - 1], so 
these needs to be calculated serially, and this takes longer.
--------------------------------------------------------------------------------
2.2
--------------------------------------------------------------------------------
In class, we discussed how the exponential moving average (EMA), in comparison
to the simple moving average (SMA), is much less suited for parallelization on
the GPU. 

Recall that the EMA is given by:
	y[n] = c * x[n] + (1 - c) * y[n - 1]

Suppose that c is close to 1, and we only require an approximation to y[n]. How
can we get this approximation in a way that is parallelizable? (Explain in
words, optionally along with pseudocode or equations.)

The key here is recognizing that the effect of the previous terms of y[n-k] on 
y[n] becomes trivial as k increases. y[n] can be rewritten as 
    y[n] = c * ∑_(i=0 to k-1)((1-c)^i * x[n-i]) + (1-c)^k * y[n-k].
Notice that y[n-k] is being multiplied by the factor (1-c)^k and since c is 
close to 1, this term will vanish quickly as k increases. Thus, depending on 
how accurate an approximation is desired, one can choose a sufficiently large 
k and approximate y[n] by ignoring y[n-k] and instead using the values from 
x[n] to x[n-k+1] and c. This computation is parallelizable because values of 
y[n] no longer rely on its previous terms.


================================================================================
Question 3: Small-Kernel Convolution (50 points)
================================================================================
Implemented in blur.cu
To run and test code,
    1) Port files over to remote GPU machine (or alternatively your own 
       properly equipped GPU)
        - Can be done via rsync -aP . username@machine.domain:lab1
    2) In the machine environment, run make all
    3) Then, run ls and confirm audio-blur and noaudio-blur files exists
    4) Test noaudio-blur via ./noaudio-blur 512 200
        - Make sure no "incorrect" statements print
    5) Test audio-blur via 
        - Run ./audio-blur 512 200 resources/example_test.wav output.wav
        - Make sure no "incorrect" statements print
        - Port output.wav to local machine via 
            rsync -aP username@machine.domain:lab1/output.wav .
        - Play output audio file to hear the effect of the transformation
        