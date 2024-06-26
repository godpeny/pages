# Language
## Interpreted vs Compiled Language
 - Interpreted Language : interpreted language is a programming language that executes instructions directly and freely, without previously compiling a program into machine-language instructions.
 - Compiled Language : compiled languages are converted directly into machine code that the processor can execute. As a result, they tend to be faster and more efficient to execute than interpreted languages. They also give the developer more control over hardware aspects, like memory management and CPU usage.
compiled languages need a “build” step – they need to be manually compiled first. You need to “rebuild” the program every time you need to make a change. In our hummus example, the entire translation is written before it gets to you.  
 - Interpreter vs Compiler : a compiler translates the entire source code into machine code before execution, resulting in faster execution since no translation is needed during runtime. On the other hand, an interpreter translates code line by line during execution, making it easier to detect errors but potentially slowing down the program.
### Interpreter
### Compiler
### Global Interpreer Lock (GIL)
#### Basics
- a global interpreter lock is a mechanism used in computer-language interpreters to synchronize the execution of threads so that only one native thread (per process) can execute basic operations (such as memory allocation and reference counting) at a time.
as a general rule, an interpreter that uses GIL will see only one thread to execute at a time, even if runs on a multi-core processor, although someimplementations provide for CPU intensive code to release the GIL, allowing multiple threads to use multiple cores. 
- a global interpreter lock is a mutual-exclusion lock held by a programming language interpreter thread to avoid sharing code that is not thread-safe with other threads.
#### Advantages
- increased speed of single-threaded programs (no necessity to acquire or release locks on all data structures separately)
- easy integration of C libraries that usually are not thread-safe,
- ease of implementation (having a single GIL is much simpler to implement than a lock-free interpreter or one using fine-grained locks).
#### Disadvantages
- if the process is almost purely made up of interpreted code and does not make calls outside of the interpreter which block for long periods of time (allowing the GIL to be released by that thread while they process), there is likely to be very little increase in speed when running the process on a multiprocessor machine.
- https://www.dabeaz.com/python/GIL.pdf

## Python
### Multi-Processing vs Multi-Threading
### Asyncio
#### Event Loop
#### Coroutine

### '*' vs '@' in vectors
- '*' : * operator performs element-wise multiplication on arrays or vectors. This means each element in one array is multiplied by the corresponding element in the other array. The two arrays must have the same shape, or they must be broadcastable to a common shape.
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = a * b  # Element-wise multiplication
print(result)  # Outputs: [4, 10, 18]
```

- '@' : @ operator is used for matrix multiplication (also known as the dot product of matrices). For 1-D arrays, it acts as a dot product; for 2-D arrays (matrices), it performs standard matrix multiplication; and for arrays with more than two dimensions, it performs a sum product over the last axis of the first array and the second-to-last axis of the second array.
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = a @ b  # Dot product
print(result)  # Outputs: 32 (1*4 + 2*5 + 3*6)
```

### '|' In Python
#### 1. Union
when used with data structure such as dictionary or set, it performs a union operation and returns a set/dictionary containingitems from both initial data structures.
```python
# 1
x =  {"a":1,"b":2} 
y = {"c":3,"d":4}

x|y # {‘a’: 1, ‘b’: 2, ‘c’: 3, ‘d’: 4}

# 2
# If a key appears in both dictionary the key from the second dictionary is used
{"a":1,"b":2} | {"a":2,"c":3} # {‘a’: 2, ‘b’: 2, ‘c’: 3}

# 3
x = {1,2,3}
y = {1,2,4}

x|y # {1, 2, 3, 4}

# 4
{1,2,3}|{4,5,6}|{7,8,9}  # {1, 2, 3, 4, 5, 6, 7, 8, 9}
{'a':1} | {'b':2} | {'c':3} # {‘a’: 1, ‘b’: 2, ‘c’: 3}
```
#### 2. Logical OR
when used on boolean values(i.e. True and False), it performs a logical OR operation.
```python
True | True # = True or True = True
False | True # = False or True = True
False | False # = False or False = False
```
### 3. Bitwise OR
When used on integer values, is performs a bitwise OR operation. The integers are converted to binary format and operations are performed on each bit and the results are then returned in decimal format. Bitwise OR returns 1 if either of the bit is 1 else 0.
```python
12 | 7 # 15

"""
12 = 1100 (Binary format)
7 = 0111 (Binary format)

12|7 = 1111 (Binary) = 15
"""
```

## Golang
### Concurrency
Goroutine, Channel
https://go.dev/doc/effective_go#concurrency
### Generic
### Interface and Type Assertion
