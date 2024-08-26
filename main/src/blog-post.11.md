# Language
## Interpreted vs Compiled Language
 - Interpreted Language : interpreted language is a programming language that executes instructions directly and freely, without previously compiling a program into machine-language instructions.
 - Compiled Language : compiled languages are converted directly into machine code that the processor can execute. As a result, they tend to be faster and more efficient to execute than interpreted languages. They also give the developer more control over hardware aspects, like memory management and CPU usage.
compiled languages need a “build” step – they need to be manually compiled first. You need to “rebuild” the program every time you need to make a change. In our hummus example, the entire translation is written before it gets to you.  
## Interprete vs Compiler
An interpreter is a software tool that directly executes high-level programming code without prior translation into machine code. It reads and executes the code line by line, translating each line into machine instructions on the fly, making it easier to identify errors and debug the code.
A compiler operates in several phases. It first analyzes the source code's structure, ensuring it adheres to the programming language's rules. Then, it converts the code into an intermediate representation, optimizing it for performance. Afterwards, it generates target machine code, utilizing various optimization techniques to improve efficiency. This gives the final output as an executable program.
In result, a compiler translates the entire source code into machine code before execution, resulting in faster execution since no translation is needed during runtime. On the other hand, an interpreter translates code line by line during execution, making it easier to detect errors but potentially slowing down the program.

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
Multithreading is a technique where multiple threads are spawned by a process to do different tasks, at about the same time, just one after the other. This gives you the illusion that the threads are running in parallel, but they are actually run in a concurrent manner. In Python, the Global Interpreter Lock (GIL) prevents the threads from running simultaneously.
Multiprocessing is a technique where parallelism in its truest form is achieved. Multiple processes are run across multiple CPU cores, which do not share the resources among them. Each process can have many threads running in its own memory space. In Python, each process has its own instance of Python interpreter doing the job of executing the instructions.
https://www.geeksforgeeks.org/difference-between-multithreading-vs-multiprocessing-in-python/

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

## Function Literals
A function literal is just an expression that defines an unnamed function.
```golang
f := func() {
        fmt.Println("I am a function literal!")
    }
f() // I am a function literal!
```

## Closure
 A closure is a function value that references variables from outside its body
 ```golang
 func main() {
    a := incrementor()
    fmt.Println(a()) // 1
    fmt.Println(a()) // 2

    b := incrementor()
    fmt.Println(b()) // 1
}

func incrementor() func() int {
    var x int
    return func() int {
        x++
        return x
    }
}
 ```

## Golang
### Goroot, GoPath, GoProxy, GoSumDB

### Naked Return
https://go.dev/tour/basics/7

### Concurrency
#### Goroutines
Goroutine is a function executing concurrently with other goroutines in the same address space.  
It is lightweight, costing little more than the allocation of stack space. And the stacks start small, so they are cheap, and grow by allocating (and freeing) heap storage as required.  
Goroutines are multiplexed onto multiple OS threads so if one should block, such as while waiting for I/O, others continue to run. This hides many of the complexities of thread creation and management.
When the call completes, the goroutine exits, silently. Since the goroutines have no way of signaling completion by themselves, Channels are needed.

#### Channel
Channels are a typed conduit through which you can send and receive values with the channel operator, ' <- '.  
```golang
ch <- v    // Send v to channel ch.
v := <-ch  // Receive from ch, and assign value to v.
```
Channels are allocated with ``make`` and the resulting value acts as a reference to an underlying data structure. 
If an optional integer parameter is provided, it sets the buffer size for the channel. The default is zero, for an unbuffered or synchronous channel.
```golang
ci := make(chan int)            // unbuffered channel of integers
cj := make(chan int, 0)         // unbuffered channel of integers
cs := make(chan *os.File, 100)  // buffered channel of pointers to Files
```
Unbuffered channels combine communication—the exchange of a value—with synchronization—guaranteeing that two calculations (goroutines) are in a known state. That is, if the channel is unbuffered, the sender blocks until the receiver has received the value.  
Similarly, if the channel has a buffer, the sender blocks only until the value has been copied to the buffer; if the buffer is full, this means waiting until some receiver has retrieved a value.
A buffered channel can be used like a semaphore as the capacity of the channel buffer limits the number of simultaneous calls to process.  
Once ``MaxOutstanding`` handlers are executing process, any more will block trying to send into the filled channel buffer, until one of the existing handlers finishes and receives from the buffer.
```golang
var sem = make(chan int, MaxOutstanding)

func Serve(queue chan *Request) {
    for req := range queue {
        sem <- 1
        go func(req *Request) {
            process(req)
            <-sem
        }(req)
    }
}
```

https://go.dev/doc/effective_go#concurrency
### Generic
https://go.dev/blog/intro-generics
https://go.dev/doc/tutorial/generics
### Interface and Type Assertion

### Range over Function
https://go.dev/wiki/RangefuncExperiment#how-are-more-complicated-loops-implemented
https://github.com/golang/go/discussions/56413
https://go.dev/blog/range-functions
