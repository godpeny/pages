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
 - Goroot : GOROOT is for compiler and tools that come from go installation and is used to find the standard libraries. It should always be set to the installation directory. 
 - GoPath : GOPATH, also called the workspace directory, is the directory where the Go code belongs. It is implemented by and documented in the go/build package and is used to resolve import statements.
 ```shell
GOPATH="/Users/godpeny/Code"
GOROOT="/opt/homebrew/Cellar/go@1.20/1.20.14/libexec"
 ```
- GOPROXY: Defines proxy servers that must be used to download dependencies. These proxy servers are used when you trigger the go command. Read more about GOPROXY in Module downloading and verification at golang.org.
- GOSUMDB: Identifies the name of the checksum database. The checksum database verifies that your packages from the go.sum file are trusted. Read more about GOSUMDB in Module authentication failures at golang.org.
- GOPRIVATE: Lists packages that are considered private. The go command does not use the GOPRIVATE or checksum database when downloading and validating these packages. Read more about GOPRIVATE in Module configuration for non-public modules at golang.org.
- GONOPROXY: Lists packages that are considered private. The go command does not use the proxy when downloading these packages. GONOPROXY overrides GOPRIVATE.
- GONOSUMDB: Lists packages that are considered private. The go command does not use the checksum database during validation of these packages. Overrides GOPRIVATE.


### Naked Return
https://go.dev/tour/basics/7

### Context

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

### Push
Provide a Set method that takes a function, and to call that function with every element in the Set. We’ll call this Push, because the Set pushes every value to the function. Here if the function returns false, we stop calling it.

ex 1)
```golang
// push.go
package push

func Backward[E any](s []E) func(func(int, E) bool) {
	return func(yield func(int, E) bool) {
		for i := len(s) - 1; i >= 0; i-- {
			if !yield(i, s[i]) { // "push" set value to the function
				return
			}
		}
	}
}
...
// main.go
package main

import (
	"fmt"
	pullFunc "github.com/godpeny/golang_range_over_function/pull"
	pushFunc "github.com/godpeny/golang_range_over_function/push"
)

func push() {
	s := []string{"hello", "world"}

    // (1)
	f := pushFunc.Backward(s)
	f(func(i int, x string) bool {
		fmt.Println(i, x)
		return true // "true" means continue running, "false" means stop
	})
    // When there are no more values to pass to yield, or if yield returns false, the push function returns.

    // (2)
	pushFunc.Backward(s)(func(i int, x string) bool {
		fmt.Println(i, x)
		return true
	})

    // new feature (3)
	for i, x := range pushFunc.Backward(s) {
		fmt.Println(i, x)
	}
}
```
(1) or (2) are the ways of earlier ways of iterating over sequences. ((1) and (2) are actually same)  
But with newly added feature, you can use as (3), which is by supporting range syntex. 

ex 2)
```golang
func (s *Set[E]) Push(f func(E) bool) {
    for v := range s.m {
        if !f(v) { // "push" set value to the function
            return
        }
    }
}

func PrintAllElementsPush[E comparable](s *Set[E]) {
    s.Push(func(v E) bool {
        fmt.Println(v)
        return true
    })
}
```

### Pull
Another approach is to return a function. Pull returns a next function that returns each
element of s with a bool for whether the value is valid. Each time the function is called, it will return a value from the Set, along with a boolean that reports whether the value is valid. The boolean result will be false when the loop has gone through all the elements. In this case we also need a stop function that can be called when no more values are needed.

ex 1)
```golang
// pull.go
func (l *List[V]) Iter() func() (V, bool) {
	cur := l
	return func() (v V, ok bool) {
		if cur == nil {
			return v, false
		}
		v, ok = cur.value, true
		cur = cur.next
		return
	}
}

...
// main.go
// (1)
next := l.Iter()
for v, ok := next(); ok; v, ok = next() { // check the value and bool from next function
	fmt.Println(v)
}

// possible range func for pull function but not supported in 1.23
(2)
for v := range l.Iter() {
	fmt.Println(v)
}
```
(1) is the way of iterating over sequences in pull function. 
(2) is the possible way of iterating using range but not supported in 1.23  

check, https://github.com/golang/go/discussions/56413


ex 2)
```golang
func (s *Set[E]) Pull() (func() (E, bool), func()) {
    ch := make(chan E)
    stopCh := make(chan bool)

    go func() {
        defer close(ch)
        for v := range s.m {
            select {
            case ch <- v:
            case <-stopCh:
                return
            }
        }
    }()

    next := func() (E, bool) { // next function that has value and bool
        v, ok := <-ch
        return v, ok
    }

    stop := func() {
        close(stopCh)
    }

    return next, stop
}

func PrintAllElementsPull[E comparable](s *Set[E]) {
    next, stop := s.Pull()
    defer stop()
    for v, ok := next(); ok; v, ok = next() {
        fmt.Println(v)
    }
}
```


### Range over Function
#### Why Needed?
Why this is needed is quotoed below from refrences.  
```
In the standard library alone, we have archive/tar.Reader.Next, bufio.Reader.ReadByte, bufio.Scanner.Scan, container/ring.Ring.Do, database/sql.Rows, expvar.Do, flag.Visit, go/token.FileSet.Iterate, path/filepath.Walk, go/token.FileSet.Iterate, runtime.Frames.Next, and sync.Map.Range, hardly any of which agree on the exact details of iteration. Even the functions that agree on the signature don’t always agree about the semantics. For example, most iteration functions that return (T, bool) follow the usual Go convention of having the bool indicate whether the T is valid. In contrast, the bool returned from runtime.Frames.Next indicates whether the next call will return something valid.

When you want to iterate over something, you first have to learn how the specific code you are calling handles iteration. This lack of uniformity hinders Go’s goal of making it easy to easy to move around in a large code base. People often mention as a strength that all Go code looks about the same. That’s simply not true for code with custom iteration.

We should converge on a standard way to handle iteration in Go, and one way to incentivize that is to support it directly in range syntax. Specifically, the idea is to allow range over function values of certain types. If any kind of code providing iteration implements such a function, then users can write the same kind of range loop they use for slices and maps and stop worrying about whether they are using a bespoke iteration API correctly.
```

### Push vs Pull
Push function has its own state maintained in its stack and can automatically clean up when the traversal is over.
(Much powerful and easier)  
In other words, a push function can be thought of as representing an entire collection. The implementation of the push function maintains iterator state implicitly on its stack, so that multiple uses of the push function use separate instances of the iterator state. Therefore, push function can be called multiple times to traverse the sequence multiple times. Also push function can be called simultaneously from different goroutines if they both want to traverse the sequence, without any synchronization.  
(state를 function param 으로 전달받으면서 stack에 저장하므로 하나의 push function instance 는 그 자체로 완전한 collection이 되어서 재사용 및 여러 goroutine에서 병렬적으로 사용이 가능하다.)  

In contrast, a pull function can be thought of as representing an iterator, not an entire collection. So a pull function always represents a specific point in one traversal of the sequence. Therefore pull function can't be reused and goroutines cannot share a pull function without synchronization. But of course,  pull function can be used from multiple call sites in a single goroutine.
(pull function은 function 밖의 state가 존재하고 현재 state의 point(위치)를 나타내기 때문에 재사용이나 병렬적 사용이 불가능 하다. 왜냐면 하나의 state를 여러 pull function이 계속해서 덮어쓰면서 race condition이 일어날 것이기 때문이다.)

#### When to Use either function?
Push and pull functions represent different ways of interacting with data, and one way may be more appropriate than the other depending on the data. For example, many programs process the lines in a file in a single loop, so a push function is appropriate for lines in a file. In contrast, it is difficult to imagine any programs that would process the bytes in a file with a single loop (except maybe wc), while many process bytes in a file incrementally from many call sites (again, lexers are an example), so a pull function is more appropriate for bytes in a file.
 - Push Function: The programs that process the lines in a file in a single loop, so a push function is appropriate for lines in a file.
 - Pull Function: The programs that process the bytes in a file with a single loop. These programs process bytes in a file incrementally from many call sites, so a pull function is more appropriate for bytes in a file.

### Concurrency vs Parallelism

#### Coroutine vs Goroutine vs Thread

#### Implementing Coroutine in Go


### Reference
 - https://github.com/golang/go/discussions/56413
 - https://go.dev/blog/range-functions
 - https://go.dev/wiki/RangefuncExperiment#how-are-more-complicated-loops-implemented



