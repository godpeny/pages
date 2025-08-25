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

### '*" vs '@' in matrices
 - '$@$': Matrix Multiplication(i.e. the dot product)
$$
A @ A \;=\;
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
@
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\;=\;
\begin{pmatrix}
a \cdot a + b \cdot c & a \cdot b + b \cdot d \\
c \cdot a + d \cdot c & c \cdot b + d \cdot d
\end{pmatrix}.
$$
 - '*': Elementwise Multiplication(computes the product of each corresponding element)
$$
A \ast A \;=\;
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\ast
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\;=\;
\begin{pmatrix}
a \cdot a & b \cdot b \\
c \cdot c & d \cdot d
\end{pmatrix}.
$$

### Einsum
Einsum provides further flexibility to compute other array operations by 'explicit' mode. The term 'explicit' mode refers to usage of numpy einsum using identifier ‘->’.  
Using identifier '->', the output of einsum can be directly controlled by specifying output subscript labels. This feature increases the flexibility of the function since summing can be disabled or forced when required.  

For example, 
"np.einsum('ij,jk,ik->i', x, sigma_j, x)"
This can be expressed as, 
$$
x^{(i)^T} \Sigma_j^{-1} x^{(i)} = 
[x_1,\,x_2]\;\Sigma_j^{-1}\;\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
\;=\;
\sum_{l=1}^{2}\sum_{k=1}^{2} (x-\mu)_j \, (\Sigma^{-1}_j)_{l,k} \, (x-\mu)_k
$$
Any index that appears in more than one input but not in the output is summed over. From above, 'j' appears in all three inputs ('ij', 'jk', 'ik') but not in the output, so we sum over 'j'. 'k' appears in the second input ('jk') but not in the output, so we also sum over 'k'.  
'i' appears in the first and third inputs and in the output, so it remains a free index (i.e.,  keep dimension 'i' in the result).
Because only 'i' remains in the output, the result has shape ('i',)

Let's see in expending form.
$$
\Sigma_j^{-1} \, x^{(i)}
=
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\begin{pmatrix}
x^{(i)}_0 \\
x^{(i)}_1
\end{pmatrix}
=
\begin{pmatrix}
a\,x^{(i)}_0 + b\,x^{(i)}_1 \\
c\,x^{(i)}_0 + d\,x^{(i)}_1
\end{pmatrix}
$$
$$
(x^{(i)})^T \,\bigl(\Sigma_j^{-1}\,x^{(i)}\bigr)
=
\begin{pmatrix}
x^{(i)}_0 & x^{(i)}_1
\end{pmatrix}
\begin{pmatrix}
a\,x^{(i)}_0 + b\,x^{(i)}_1 \\
c\,x^{(i)}_0 + d\,x^{(i)}_1
\end{pmatrix}
$$
$$
x^{(i)}_0\,(a\,x^{(i)}_0 + b\,x^{(i)}_1)
\;+\;
x^{(i)}_1\,(c\,x^{(i)}_0 + d\,x^{(i)}_1) 
$$

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

### Sum() in Python
```
z = [3, 2, 3, 1, 3, 2]
i = 3
```
 - ``sum(x for x in z if x == i)`` : Sums the actual values equal to ``i`` (resulting in ``i`` * count). So the result is ``9``.
 - ``sum(z == i)`` : Counts the number of elements equal to ``i`` (resulting in just the count). So the reulst is ``3``.

### Uniform Groupping in Python
```python
m = x.shape[0]
idx = np.random.permutation(m)  # randomly permute indices from 0 to m-1
samples = np.array(np.array_split(x[idx], K))
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

### Gofmt
Gofmt formats Go programs. It uses tabs for indentation and blanks for alignment. Alignment assumes that an editor is using a fixed-width font.  
For example,
 - To check files for unnecessary parentheses:  
 ``gofmt -r '(a) -> a' -l *.go``  
 - To remove the parentheses:  
 ``gofmt -r '(a) -> a' -w *.go``  
 - To convert the package tree from explicit slice upper bounds to implicit ones:  
 ``gofmt -r 'α[β:len(α)] -> α[β:]' -w $GOROOT/src``

The most obvious benefit of using gofmt is that when you open an unfamiliar Go program, your brain doesn't get distracted, even subconsciously, about why that brace is in the wrong place; you can focus on the code, not the formatting.   
But there are many more interesting uses for gofmt. Gofmt can take any file in the Go source tree, parse it into an internal representation, and then write exactly the same bytes back out to the file. So gofmt only has to worry about one formatting convention, and we've agreed to accept that as the official one.


### Naked Return
A return statement without arguments returns the named return values. This is known as a "naked" return.


```golang
package main

import "fmt"

func split(sum int) (x, y int) {
	x = sum * 4 / 9
	y = sum - x
	return // naked return (return x,y)
}

func main() {
	fmt.Println(split(17))
}

```

### Defer
A defer statement defers the execution of a function until the surrounding function returns.
The deferred call's arguments are evaluated immediately, but the function call is not executed until the surrounding function returns.
The arguments to the deferred function (which include the receiver if the function is a method) are evaluated when the defer executes, not when the call executes. 
Also, the Deferred functions are executed in LIFO order. 

ex 1)
```golang
func deferredFunc(s string) { 
    fmt.Println("returned value from returnArg():", s) 
    fmt.Println("(3)") 
} 

func returnArg() string { 
    fmt.Println("(2)") return "!!!" 
} 

func main() { 
    defer deferredFunc(returnArg()) /
    fmt.Println("(1)")
}
// (2) 
// (1) 
// returned value from returnArg(): !!! 
// (3)
```
ex 2)
```golang
func handleInnerPanic() { 
    defer fmt.Println("(4) reachable") 
    fmt.Println("(1) reachable") 
    
    defer func() { 
        v := recover() 
        fmt.Println("(3) recovered:", v) 
    }() 
    
    defer fmt.Println("(2) reachable") 
    panic("panic occured") 
    fmt.Println("unreachable") }

// (1) reachable 
// (2) reachable 
// (3) recovered: panic occured
// (4) reachable
```
### Context
 - https://go.dev/blog/context

### Interface{}
 - https://jordanorelli.com/post/32665860244/how-to-use-interfaces-in-go
 - https://research.swtch.com/interfaces

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
A type assertion provides access to an interface value's underlying concrete value.
```golang
t := i.(T)
```
This statement asserts that the interface value i holds the concrete type T and assigns the underlying T value to the variable t.

If i does not hold a T, the statement will trigger a panic.

To test whether an interface value holds a specific type, a type assertion can return two values: the underlying value and a boolean value that reports whether the assertion succeeded.
```golang
t, ok := i.(T)
```
If i holds a T, then t will be the underlying value and ok will be true.
If not, ok will be false and t will be the zero value of type T, and no panic occurs.(Note the similarity between this syntax and that of reading from a map.)

```golang
package main

import "fmt"

func main() {
	var i interface{} = "hello"

	s := i.(string)
	fmt.Println(s) // hello

	s, ok := i.(string)
	fmt.Println(s, ok) // hello true

	f, ok := i.(float64)
	fmt.Println(f, ok) // 0 false

	f = i.(float64) // panic
	fmt.Println(f)
}
```
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
#### Concurrency
Concurrency relates to an application that is processing more than one task at the same time. Concurrency is an approach that is used for decreasing the response time of the system by using the single processing unit. Concurrency creates the illusion of parallelism, however actually the chunks of a task aren’t parallelly processed, but inside the application, there are more than one task is being processed at a time. It doesn’t fully end one task before it begins ensuing. 
Concurrency is achieved through the interleaving operation of processes on the central processing unit(CPU) or in other words by the context switching. that’s rationale it’s like parallel processing. It increases the amount of work finished at a time. 

#### Parallelism
Parallelism is related to an application where  tasks are divided into smaller sub-tasks that are processed seemingly simultaneously or parallel. It is used to increase the throughput and computational speed of the system by using multiple processors. Therefore it improves the throughput and computational speed of the system.

While concurrency can be done by using a single processing unit, parallelism can’t be done by using a single processing unit. it needs multiple processing units.

### Coroutine vs Goroutine vs Thread (vs python generator)
#### Coroutine
Coroutines provide concurrency without parallelism: when one coroutine is running, the one that resumed it or yielded to it is not. In other words, coroutines run one at a time and only switch at specific points in the program, the coroutines can share data among themselves without races. The explicit switches (``coroutine.yield`` and ``coroutine.resume`` in the  Lua example below) serve as synchronization points, creating happens-before edges.

```lua
function T(l, v, r)
    return {left = l, value = v, right = r}
end

e = nil
t1 = T(T(T(e, 1, e), 2, T(e, 3, e)), 4, T(e, 5, e))
t2 = T(e, 1, T(e, 2, T(e, 3, T(e, 4, T(e, 5, e)))))
t3 = T(e, 1, T(e, 2, T(e, 3, T(e, 4, T(e, 6, e)))))

function visit(t)
    if t ~= nil then  -- note: ~= is "not equal"
        visit(t.left)
        -- "yield" allows a running coroutine to suspend its execution so that it can be resumed later.
        coroutine.yield(t.value) 
        visit(t.right)
    end
end

function cmp(t1, t2)
    co1 = coroutine.create(visit)
    co2 = coroutine.create(visit)
    while true
    do
        -- "resume" (re)starts the execution of a coroutine, changing its state from suspended to running.
        ok1, v1 = coroutine.resume(co1, t1) 
        ok2, v2 = coroutine.resume(co2, t2)
        if ok1 ~= ok2 or v1 ~= v2 then
            return false
        end
        if not ok1 and not ok2 then
            return true
        end
    end
end
```
Reference: https://www.lua.org/pil/9.1.html

Because scheduling is explicit (without any preemption) and done entirely without the operating system, a coroutine switch takes at most around ten nanoseconds, usually even less. Startup and teardown is also much cheaper than threads.

#### Thread
Threads provide more power than coroutines, but with more cost. The additional power is parallelism, and the cost is the overhead of scheduling, including more expensive context switches and the need to add preemption in some form. Typically the operating system provides threads, and a thread switch takes a few microseconds.

#### Goroutine
Go’s goroutines are cheap threads: a goroutine switch is closer to a few hundred nanoseconds, because the Go runtime takes on some of the scheduling work, but goroutines still provide the full parallelism and preemption of threads.

#### Generator (python)
The generator object contains the state of the single call to ``gen`` from below example, meaning local variable values and which line is executing. That state is pushed onto the call stack each time the generator is resumed and then popped back into the generator object at each yield, which can only occur in the top-most call frame.  
In this way, the generator uses the same stack as the original program, avoiding the need for a full coroutine implementation but introducing these confusing limitations instead.
```python
def test_generator():
    yield 1
    yield 2
    yield 3

gen = test_generator()
type(gen) # <class 'generator'>

next(gen) # 1
next(gen) # 2
next(gen) # 3
next(gen)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# StopIteration

```

#### Happened-before
Happend-before is a relation between the result of two events, such that if one event should happen before another event, the result must reflect that, even if those events are in reality executed out of order (usually to optimize program flow).  
Also it indicates that memory written by one statement is visible to another statement. This means that the first statement completes its write before the second statement starts its read. 

#### Implementing Coroutine in Go
Why Coroutines in Go? 
 - The parallelism provided by the goroutines caused races and eventually led to abandoning the design. The proper coroutines would have avoided the races and been more efficient than goroutines.  

Where to use Coroutines in Go? 
 - An anticipated future use case for coroutines in Go is iteration over generic collections. (=Push functions)

Implement  
(DIY) https://research.swtch.com/coro

#### Storing Data in Control Flow (Concurrency and Parallelism) 
https://research.swtch.com/pcdata

### Memory Models in Go
https://research.swtch.com/gomm


### Reference
 - https://github.com/golang/go/discussions/56413
 - https://go.dev/blog/range-functions
 - https://go.dev/wiki/RangefuncExperiment#how-are-more-complicated-loops-implemented



### Go Plugin
Go “plugin” is a separately compiled binary (a shared object, .so) that your Go program can load at runtime to get functions/variables.

- Built with: ``go build -buildmode=plugin -o myplugin.so ./path``
- File type: a shared library (.so) containing Go code.
- Loaded with: the plugin standard package (plugin.Open, Lookup).
- Use-case: add/replace features without rebuilding the main binary (extensible “drivers,” user-provided logic, etc.).

```go
// plugins/greeter/greeter.go
package main

import "fmt"

func Greet(name string) string { return fmt.Sprintf("hi %s", name) }
```
```bash
go build -buildmode=plugin -o greeter.so ./plugins/greeter
```

```go
// main.go
package main

import (
	"fmt"
	"plugin"
)

func main() {
	p, err := plugin.Open("greeter.so")
	if err != nil { panic(err) }

	sym, err := p.Lookup("Greet")
	if err != nil { panic(err) }

	greet := sym.(func(string) string) // type assert
	fmt.Println(greet("world"))
}
```

#### Hashcorp Go-Plugin
A library that lets your Go program load plugins as separate OS processes and talk to them over RPC/gRPC (usually over the plugin’s stdio). The host launches the plugin binary, performs a handshake, and then calls methods through an interface as if it were local—behind the scenes it’s RPC.

Used by tools like Terraform, Vault, Nomad (historically &/or in parts) to let third-party plugins evolve independently and not crash the host.

https://github.com/hashicorp/go-plugin/blob/main/docs/extensive-go-plugin-tutorial.md

```go
// Host (force gRPC)
client := plugin.NewClient(&plugin.ClientConfig{
  AllowedProtocols: []plugin.Protocol{plugin.ProtocolGRPC},
  // ... HandshakeConfig, Plugins map, Cmd: exec.Command("./my-plugin")
})
rpcClient, _ := client.Client() // spawns the child process and connects
```

```go
// Plugin binary (serve over gRPC)
plugin.Serve(&plugin.ServeConfig{
  HandshakeConfig: hs,
  Plugins: map[string]plugin.Plugin{
    "thing": &MyGRPCPlugin{Impl: myImpl},
  },
  GRPCServer: plugin.DefaultGRPCServer, // enable gRPC transport
})
```

#### Go Plugin vs Hashcorp Go-Plugin
With HashiCorp go-plugin the host app spawns the plugin as a separate OS process and talks to it over RPC (net/rpc or gRPC) via stdio. Because it’s a different process:

- Memory/GC isolation: separate address space; a bug in the plugin can’t scribble over the host’s memory.
- Crash isolation: a panic/segfault/leak in the plugin only kills that process; the host stays up (you can restart the plugin).
- Scheduling/resource isolation: you can cap the plugin’s CPU/RAM (cgroups/rlimits), and it has its own file descriptors, threads, etc.
- Privilege isolation: you can run it as a different user, in a container/chroot, with seccomp/AppArmor, separate namespaces, etc.
- Version/dep decoupling: host and plugin don’t have to share an in-process ABI like the stdlib plugin package does; they just need a compatible RPC protocol.

Contrast with the Go stdlib plugin: that loads a ``.so`` in-process. It’s fast, but no isolation—one panic can take down the host, and plugins must be built with the exact same Go/toolchain settings.