CS 179
Assignment 2: Matrix Transposition
Name: Jayden Nyamiaka

Due: Wednesday, April 17, 2024 - 3:00 PM.

========================================
Submission:
Put all answers to conceptual questions in a file called README.md.
Implement all code in the .cu files as directed in the instructions below.
After answering all of the questions, list how long part 1 and part 2 took
in the README.md Feel free to leave any other feedback.

Port files over to remote GPU machine at username@machine.domain.
Finally, put a zip file of your solution in your home directory on the
remote GPU machine, with the name lab[N]_2024_submission.zip
Your submission should be a single archive file (.zip)
with your README file and all code.

========================================


PART 1

Question 1.1: Latency Hiding (5 points)
---------------------------------------

Approximately how many arithmetic instructions does it take to hide the latency of a single arithmetic instruction on a GK110?

Assume all of the arithmetic instructions are independent (ie have no
instruction dependencies).

You do not need to consider the number of execution cores on the chip.

Hint: What is the latency of an arithmetic instruction? How many instructions
can a GK110 begin issuing in 1 clock cycle (assuming no dependencies)?

Background/Specs:
GK110 has 4 warp schedulers in each SM and 2 dispatchers in each scheduler.
GPU clock = 1 GHz (1 clock/ns)
GPU Arithmetic Instruction Latency: ~ 10 ns
Context switching between warps if 1 ns on 1 GHz GPU
GK110 can start instructions in up to 4 warps each clock
with up to 2 subsequent, independent instructions in each warp.

Answer:
The latency of an arithmetic instruction on a GPU is about 10 ns.
For the GK110, there's about 1 clock per ns, and it can start instructions
in up to 4 warps each clock with up to 2 subsequent, independent
instructions in each warp. This sums to beginning 8 instructions each ns, so
it takes the GK110 beginning 80 arithmetic instructions to hide the latency
of a single arithmetic instruction.

Question 1.2: Thread Divergence (6 points)
------------------------------------------

Let the block shape be (32, 32, 1).

(a)
int idx = threadIdx.y + blockSize.y * threadIdx.x;
if (idx % 32 < 16)
    foo();
else
    bar();

Does this code diverge? Why or why not?

Answer:
No. The block will be split into warps of 32 threads, so each row in the block
will become its own warp. We know that both threadIdx.x and threadIdx.y will
run from 0 to 31 (from the block shape), and we see the if statement will
return true for all threadIdx.y values from 0 to 15 and be false for
threadIdx.y values from 16 to 31. Since the value for threadIdx.y is
constant for each row, then each value for threadIdx.y within a single warp
is also constant. Thus, for ever warp, the evaluation of the if statement
and resulting behavior within the warp (foo() or bar()) will be the same
such that there's no warp divergence.

(b)
const float pi = 3.14;
float result = 1.0;
for (int i = 0; i < threadIdx.x; i++)
    result *= pi;

Does this code diverge? Why or why not? (This is a bit of a trick question,
either "yes" or "no can be a correct answer with appropriate explanation.)

Answer:
Yes. The variable threadIdx.x will run from 0 to 31, inclusive (from the
block shape). Each row becomes its own warp such that there will be various
different values for threadIdx.x within a single warp. Thus, the loop will
iterate a different number of times for each thread within a warp and
have different final values for result, leading to warp divergence.

Question 1.3: Coalesced Memory Access (9 points)
------------------------------------------------

Let the block shape be (32, 32, 1). Let data be a (float *) pointing to global
memory and let data be 128 byte aligned (so data % 128 == 0).

Consider each of the following access patterns.

(a)
data[threadIdx.x + blockSize.x * threadIdx.y] = 1.0;

Is this write coalesced? How many 128 byte cache lines does this write to?

Answer:
Floats in C++ take up 4 bytes, so with each additional x, the index of data
increases by 4 bytes. For our block, each row is its own warp, and addresses
are stored per warp, using less cache lines if the memory is coalesced. All
threads in the same warp, access data directly adjacent to each other
(according to the indexing above), so for a single warp, there are 32
adjacent memory accesses each 4 bytes long, adding up to a continuos block
of 128 bytes of global memory per warp. Since data is 128-byte aligned and
cache lines are 128 bytes, the 128-byte memory block for each warp is also
128-byte aligned and each warp will need a single 128-byte cache line. Thus,
this write is coalesced and will write to 1 cache line per warp and 32 cache
lines in total.

(b)
data[threadIdx.y + blockSize.y * threadIdx.x] = 1.0;

Is this write coalesced? How many 128 byte cache lines does this write to?

Answer:
Along the same reasoning as question 1a, this write is not
coalesced and does not utilize cache lines efficiently. This indexing will
make threads in the same column access adjacent data. However, since each
row is its own warp, these column-adjacent threads are in different warps
and makes it so threads in the same warp are accessing data
(32 floats)*(4 bytes/float) = 128 bytes away from each other. Thus, each
warp needs a cache line for every thread such that this write is not
coalesced and will write to 32 cache lines per warp and 32*32 = 1024 cache
lines in total.

(c)
data[1 + threadIdx.x + blockSize.x * threadIdx.y] = 1.0;

Is this write coalesced? How many 128 byte cache lines does this write to?

Answer:
Along the same reasoning as question 1a, this write is not coalesced
because the data is not aligned. Even though all adjacent threads in the
same warp are accessing adjacent data and a continuos 128-byte block of
memory, the misalignment means we need 2 cache lines per warp since cache
lines must be 128-byte aligned. Since 1 is being added to the indexing, this
will shift all the accesses by 4 bytes compared to the similar indexing in
1a. This makes the block unaligned such that the first 31 threads of a warp
are accessing the last 124 bytes of one aligned 128-byte block while the
last thread in the warp is accessing the first 4 bytes of the following
aligned 128-byte block. Thus, this write is not coalesced and will write
to 2 cache lines per warp and 32*2= 64 cache lines in total.

Question 1.4: Bank Conflicts and Instruction Dependencies (15 points)
---------------------------------------------------------------------

Let's consider multiplying a 32 x 128 matrix with a 128 x 32 element matrix.
This outputs a 32 x 32 matrix. We'll use 32 ** 2 = 1024 threads and each thread
will compute 1 output element. Although its not optimal, for the sake of
simplicity let's use a single block, so grid shape = (1, 1, 1),
block shape = (32, 32, 1).

For the sake of this problem, let's assume both the left and right matrices have
already been stored in shared memory are in column major format. This means the
element in the ith row and jth column is accessible at lhs[i + 32 * j] for the
left hand side and rhs[i + 128 * j] for the right hand side.

This kernel will write to a variable called output stored in shared memory.

Consider the following kernel code:

int i = threadIdx.x;
int j = threadIdx.y;
for (int k = 0; k < 128; k += 2) {
    output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
    output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
}

(a)
Are there bank conflicts in this code? If so, how many ways is the bank conflict
(2-way, 4-way, etc)?

Answer:
Note that each row comprises its own thread, so using a block shape of
(32,32,1), all threads within a single warp will have the same constant value
for j = threadIdx.y with only i = threadIdx.x iterating from 0 to 31. Now,
let's analyze the data we're accessing to see if there are any bank conflicts.
Accessing:
    - output[i + 32 * j]
    - lhs[i + 32 * k]
    - rhs[k + 128 * j]
    - lhs[i + 32 * (k + 1)]
    - rhs[(k + 1) + 128 * j]
Accesses for output[i + 32 * j], lhs[i + 32 * k], and lhs[i + 32 * (k + 1)]
don't cause bank conflicts because multiples of 32 do not affect the bank,
making i the only variable that affects which bank these addresses are in.
Then, since i varies from 0 to 31 for different threads in the same warp, all
threads within the same warp will access different banks for these addresses.
For rhs[k + 128 * j] and rhs[(k + 1) + 128 * j], we can again note that the
multiples of 128 will not affect the bank these addresses are in. Then, since
all these threads run the same code in parallel (at the same exact time), the
values for k will be exactly the same for all threads within the same warp
such that all threads will request a shared access of the same data in 2
adjacent banks, and the data will be broadcasted to all threads simultaneously
without violating the parallelism. Thus, there are no bank conflicts.

(b)
Expand the inner part of the loop (below)

output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];

into "psuedo-assembly" as was done in the coordinate addition example in lecture 4.

There's no need to expand the indexing math, only to expand the loads, stores,
and math. Notably, the operation a += b * c can be computed by a single
instruction called a fused multiply add (FMA), so this can be a single
instruction in your "psuedo-assembly".

Hint: Each line should expand to 5 instructions.

Answer:

1. z = output[i + 32 * j];
2. l = lhs[i + 32 * k];
3. r = rhs[k + 128 * j];
4. FFMA(z, l, r);
5. output[i + 32 * j] = z;
6. z = output[i + 32 * j];
7. l = lhs[i + 32 * (k + 1)];
8. r = rhs[(k + 1) + 128 * j];
9. FFMA(z, l, r);
10. output[i + 32 * j] = z;

(c)
Identify pairs of dependent instructions in your answer to part b.

Answer:

- 4 is dependent on 1,2,3
- 5 is dependent on 4
- 6 is dependent on 5
- 9 is dependent on 6,7,8
- 10 is dependent on 9

(d)
Rewrite the code given at the beginning of this problem to minimize instruction
dependencies. You can add or delete instructions (deleting an instruction is a
valid way to get rid of a dependency!) but each iteration of the loop must still
process 2 values of k.

Answer:

Assembly Optimization of inner part of the loop:

1. z = output[i + 32 * j];
2. l0 = lhs[i + 32 * k];
3. r0 = rhs[k + 128 * j];
4. l1 = lhs[i + 32 * (k + 1)];
5. r2 = rhs[(k + 1) + 128 * j];
6. FFMA(z, l1, r1);
7. FFMA(z, l2, r2);
8. output[i + 32 * j] = z;

CUDA Optimization of the complete kernel code:

int i = threadIdx.x;
int j = threadIdx.y;
float temp = 0; 
for (int k = 0; k < 128; k += 2) {
    temp += lhs[i+32*k] * rhs[k+128*j] + lhs[i+32*(k+1)] * rhs[(k+1)+128*j];
}
output[i + 32 * j] += temp

Note: This way, we're not constantly writing to global memory and can save
all intermediate results in a register, instead of loading and storing back
in forth from global memory to registers.

(e)
Can you think of any other anything else you can do that might make this code
run faster?

Answer:
There are a couple of ways we could further speed up this code. One is using
a single stack variable to store all intermediate operations and then only
adding all the results to global memory once, at the end. This would make it
so the code is only using registers without the need to load and unload from
shared memory (since stack variables are usually stored in registers), which
could significantly cut down the time it takes the code to run. We could also
further unroll the loop dependent on k. The range for k is [0,128) such that
for rhs (via indexing k+128*j), every bank has 4 addresses in rhs. These can
be grouped together and make it so we have one warp per k loop (with each
thread doing those 4 operations in the same bank) such that the computations
in the loop are also parallelized. This would save a lot of time as well.


PART 2 - Matrix transpose optimization (65 points)
--------------------------------------------------

Optimize the CUDA matrix transpose implementations in transpose_device.cu.
Read ALL of the TODO comments. Matrix transpose is a common exercise in GPU
optimization, so do not search for existing GPU matrix transpose code on the
internet.

Your transpose code only need to be able to transpose square matrices where the
side length is a multiple of 64.

The initial implementation has each block of 1024 threads handle a 64x64 block
of the matrix, but you can change anything about the kernel if it helps obtain
better performance.

The main method of transpose.cc already checks for correctness for all transpose
results, so there should be an assertion failure if your kernel produces incorrect
output.

The purpose of the shmemTransposeKernel is to demonstrate proper usage of global
and shared memory. The optimalTransposeKernel should be built on top of
shmemTransposeKernel and should incorporate any "tricks" such as ILP, loop
unrolling, vectorized IO, etc that have been discussed in class.

You can compile and run the code by running

make transpose
./transpose

and the build process was tested on minuteman. If this does not work on haru for
you, be sure to add the lines

export PATH=/usr/local/cuda-6.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:$LD_LIBRARY_PATH

to your ~/.profile file (and then exit and ssh back in to restart your shell).

On OS X, you may have to run or add to your .bash_profile the command

export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/usr/local/cuda/lib/

in order to get dynamic library linkage to work correctly.

The transpose program takes 2 optional arguments: input size and method. Input
size must be one of -1, 512, 1024, 2048, 4096, and method must be one all,
cpu, gpu_memcpy, naive, shmem, optimal. Input size is the first argument and
defaults to -1. Method is the second argument and defaults to all. You can pass
input size without passing method, but you cannot pass method without passing an
input size.

Examples:
./transpose
./transpose 512
./transpose 4096 naive
./transpose -1 optimal

Copy paste the output of ./transpose into README.md once you are done.
Describe the strategies used for performance in either block comments over the
kernel (as done for naiveTransposeKernel) or in README.md.

Index of the GPU with the lowest temperature: 1 (0 C)
Time limit for this program set to 10 seconds
Size 512 naive CPU: 1.096128 ms
Size 512 GPU memcpy: 0.024128 ms
Size 512 naive GPU: 0.047552 ms
Size 512 shmem GPU: 0.010624 ms
Size 512 optimal GPU: 0.009696 ms

Size 1024 naive CPU: 5.420576 ms
Size 1024 GPU memcpy: 0.048032 ms
Size 1024 naive GPU: 0.110656 ms
Size 1024 shmem GPU: 0.035424 ms
Size 1024 optimal GPU: 0.033984 ms

Size 2048 naive CPU: 49.950848 ms
Size 2048 GPU memcpy: 0.157376 ms
Size 2048 naive GPU: 0.386112 ms
Size 2048 shmem GPU: 0.136256 ms
Size 2048 optimal GPU: 0.135776 ms

Size 4096 naive CPU: 176.075745 ms
Size 4096 GPU memcpy: 0.537120 ms
Size 4096 naive GPU: 1.655136 ms
Size 4096 shmem GPU: 0.550752 ms
Size 4096 optimal GPU: 0.554176 ms

BONUS (+5 points, maximum set score is 100 even with bonus)
--------------------------------------------------------------------------------

Mathematical scripting environments such as Matlab or Python + Numpy often
encourage expressing algorithms in terms of vector operations because they offer
a convenient and performant interface. For instance, one can add 2 n-component
vectors (a and b) in Numpy with c = a + b.

This is often implemented with something like the following code:

void vec_add(float *left, float *right, float *out, int size) {
    for (int i = 0; i < size; i++)
        out[i] = left[i] + right[i];
}

Consider the code

a = x + y + z

where x, y, z are n-component vectors.

One way this could be computed would be

vec_add(x, y, a, n);
vec_add(a, z, a, n);

In what ways is this code (2 calls to vec_add) worse than the following?

for (int i = 0; i < n; i++)
    a[i] = x[i] + y[i] + z[i];

List at least 2 ways (you don't need more than a sentence or two for each way).

Answer:

- The first way requires more register loading and unloading. In the first
way, you must load data into registers for x[i] and y[i], then do the
computation storing the result in a register for a[i], then store the register
from a[i] into the data for a[i] itself. Then, it has to do this again to add
a and z. However, for the second way, this register loading is mitigated
because all loads can be done in the beginning and then only one store is
required for a[i] per index.
- The first way requires 2 loops whereas the second way only requires one. The
set up and operations necessary for the loop requires additional time and
compute, so having 2 loops when one is sufficient is already much worse.
- The second way has better instruction level parallelism (ILP). The second
way compiles such that all the loads are done in the beginning. This makes
the code less sequential, have higher ILP, and is overall better.
