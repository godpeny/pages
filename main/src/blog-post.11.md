# Python
## Interpreted vs Compiled Language
 - Interpreted Language : interpreted language is a programming language that executes instructions directly and freely, without previously compiling a program into machine-language instructions.
 - Compiled Language : compiled languages are converted directly into machine code that the processor can execute. As a result, they tend to be faster and more efficient to execute than interpreted languages. They also give the developer more control over hardware aspects, like memory management and CPU usage.
compiled languages need a “build” step – they need to be manually compiled first. You need to “rebuild” the program every time you need to make a change. In our hummus example, the entire translation is written before it gets to you.  
## Interprete vs Compiler
An interpreter is a software tool that directly executes high-level programming code without prior translation into machine code. It reads and executes the code line by line, translating each line into machine instructions on the fly, making it easier to identify errors and debug the code.
A compiler operates in several phases. It first analyzes the source code's structure, ensuring it adheres to the programming language's rules. Then, it converts the code into an intermediate representation, optimizing it for performance. Afterwards, it generates target machine code, utilizing various optimization techniques to improve efficiency. This gives the final output as an executable program.
In result, a compiler translates the entire source code into machine code before execution, resulting in faster execution since no translation is needed during runtime. On the other hand, an interpreter translates code line by line during execution, making it easier to detect errors but potentially slowing down the program.

## Global Interpreer Lock (GIL)
### Basics
- a global interpreter lock is a mechanism used in computer-language interpreters to synchronize the execution of threads so that only one native thread (per process) can execute basic operations (such as memory allocation and reference counting) at a time.
as a general rule, an interpreter that uses GIL will see only one thread to execute at a time, even if runs on a multi-core processor, although someimplementations provide for CPU intensive code to release the GIL, allowing multiple threads to use multiple cores. 
- a global interpreter lock is a mutual-exclusion lock held by a programming language interpreter thread to avoid sharing code that is not thread-safe with other threads.
### Advantages
- increased speed of single-threaded programs (no necessity to acquire or release locks on all data structures separately)
- easy integration of C libraries that usually are not thread-safe,
- ease of implementation (having a single GIL is much simpler to implement than a lock-free interpreter or one using fine-grained locks).
### Disadvantages
- if the process is almost purely made up of interpreted code and does not make calls outside of the interpreter which block for long periods of time (allowing the GIL to be released by that thread while they process), there is likely to be very little increase in speed when running the process on a multiprocessor machine.
- https://www.dabeaz.com/python/GIL.pdf

## Multi-Processing vs Multi-Threading
Multithreading is a technique where multiple threads are spawned by a process to do different tasks, at about the same time, just one after the other. This gives you the illusion that the threads are running in parallel, but they are actually run in a concurrent manner. In Python, the Global Interpreter Lock (GIL) prevents the threads from running simultaneously.
Multiprocessing is a technique where parallelism in its truest form is achieved. Multiple processes are run across multiple CPU cores, which do not share the resources among them. Each process can have many threads running in its own memory space. In Python, each process has its own instance of Python interpreter doing the job of executing the instructions.
https://www.geeksforgeeks.org/difference-between-multithreading-vs-multiprocessing-in-python/

## Asyncio
## Event Loop
## Coroutine

## '*' vs '@' in vectors
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

## '*" vs '@' in matrices
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

## Einsum
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

## '|' In Python
### 1. Union
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
### 2. Logical OR
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

## Sum() in Python
```
z = [3, 2, 3, 1, 3, 2]
i = 3
```
 - ``sum(x for x in z if x == i)`` : Sums the actual values equal to ``i`` (resulting in ``i`` * count). So the result is ``9``.
 - ``sum(z == i)`` : Counts the number of elements equal to ``i`` (resulting in just the count). So the reulst is ``3``.

## Uniform Groupping in Python
```python
m = x.shape[0]
idx = np.random.permutation(m)  # randomly permute indices from 0 to m-1
samples = np.array(np.array_split(x[idx], K))
```
## Other Python Operators
- ``**``: 	Exponentiation.
- ``//``:	Floor division.

## Pysaprk 
### `f.col(name)`
컬럼 객체 참조. 거의 모든 컬럼 연산의 시작점.
```python
f.col("kuid")                   # kuid 컬럼 참조
f.col("kuid") == "click"        # 비교 (Column 객체 반환)
```
### `.cast(type)`
컬럼의 데이터 타입 변환.
```python
f.col("ad_group_id").cast("long")    # string → long
f.col("kuid").cast("string")
```
### `.alias(name)`
컬럼에 이름 부여 (SQL의 `AS`). 필요한 이유: `cast` 후 이름을 안 주면 `CAST(... AS BIGINT)` 같은 더러운 이름이 됨.
```python
f.col("plusfriend_id").cast("long").alias("channel_id")
# 결과 컬럼명: channel_id
```
### `.select(...)`
"이 컬럼들만 남기고 새 DataFrame 생성". 나열 안 한 컬럼은 사라짐.
```python
df.select(
    uk_col,                                 # Column 객체 (변환 포함)
    "action",                               # 문자열 (변환 없이 그대로)
    f.col("x").cast("long").alias("x"),     # Column 표현식
)
```
### `.withColumn(name, expr)`
"컬럼 하나를 추가하거나, 같은 이름이면 덮어쓰기". row 수는 그대로.
```python
df.withColumn("action_label", f.when(f.col("action") == "click", 1).otherwise(0))
df.withColumn("sample_weight", ...)   # 같은 이름이면 덮어쓰기
df.withColumn("has_imp", f.lit(1))    # 상수값으로 컬럼 추가
```
#### `.select` vs `.withColumn`
|                  | row 수 | 다른 컬럼              |
| ---------------- | ------ | ---------------------- |
| `.select`        | 그대로 | **명시한 것만 남음**   |
| `.withColumn`    | 그대로 | **다 유지** + 새 컬럼  |
### `f.lit(value)`
상수값을 Column으로 변환. `withColumn`에 상수 박을 때 필요.
```python
f.lit(1)        # 모든 row에 1
f.lit("KAKAO")  # 모든 row에 "KAKAO"
```
### `f.when(cond, val).otherwise(val)`
조건 분기.
```python
f.when(f.col("action") == "click", 1).otherwise(0)
```
중첩으로 3-way 분기도 가능:
```python
f.when(x < min, min).otherwise(
    f.when(x > max, max).otherwise(x)
)
# CASE WHEN x<min THEN min WHEN x>max THEN max ELSE x END
```
### `.drop_duplicates()` / `.dropDuplicates()`
완전히 동일한 row 제거. 지정 컬럼 기준도 가능.
```python
df.drop_duplicates()                    # 모든 컬럼 기준
df.drop_duplicates(["kuid", "p_dt"])    # 두 컬럼 기준
```
### `.groupby(cols).agg(...)`
"같은 키 묶고 그룹별로 집계". row 수가 줄어듦.
```python
df.groupby(["kuid", "creative_id"]).agg(
    f.sum("action_label").alias("clicks"),
    f.signum(f.sum("action_label")).alias("clicked"),
)
```
`.groupby(...)` 단독은 의미 없음 → 반드시 `.agg(...)` 또는 `.count()` 등 집계 함수와 짝

### 집계 함수들
| 함수                       | 의미                              |
| -------------------------- | --------------------------------- |
| `f.sum("x")`               | 합계                              |
| `f.count("x")`             | 개수                              |
| `f.max("x")` / `f.min("x")`| 최댓값/최솟값                     |
| `f.avg("x")`               | 평균                              |
| `f.signum(x)`              | 부호 함수: 양수→1, 0→0, 음수→-1   |

### `.join(other, on, how)`
SQL의 `JOIN`.
```python
df_send.join(df_reaction, ["kuid", "creative_id"], "left")
```

- `on`: 조인 키 (리스트로 주면 양쪽 동일 컬럼명 매칭)
- `how`: `"inner"`, `"left"`, `"right"`, `"outer"`, `"semi"`, `"anti"`

**LEFT JOIN 결과**
- 좌측 모든 row 유지
- 우측에 매칭 없으면 → 우측 컬럼들이 **NULL**
### `.fillna(value, subset=[...])`
NULL을 채움.
```python
df.fillna(0, subset=["action_label", "has_imp"])
# action_label, has_imp가 NULL이면 0으로 채움
```
### `.where(cond)` / `.filter(cond)`
조건에 맞는 row만 남김 (둘은 동일).
```python
df.where(f.col("kuid").isNotNull())
df.filter(f.col("p_dt") >= "2026-05-01")
```
자주 쓰는 Column 메서드:

- `.isNull()` / `.isNotNull()`
- `.isin([v1, v2])`
- `.between(a, b)`
- `==`, `!=`, `<`, `>`, `<=`, `>=`

### `spark.sql(query)`
문자열 SQL을 그대로 실행해서 DataFrame 반환.

```python
df = spark.sql(f"""
    SELECT ... FROM {table}
    WHERE p_dt BETWEEN '{start}' AND '{end}'
""")
```
- f-string으로 변수 주입
- 결과는 일반 DataFrame이라 이후 `.select`, `.withColumn` 등 체이닝 가능

### `get_json_object(col, '$.path')` (Spark SQL 함수)
JSON 문자열 컬럼에서 특정 경로 값 추출.

```sql
get_json_object(log_data, '$.rawMessage.user.kuid')
```

- `$` = JSON 루트
- 결과는 **무조건 string**. 숫자가 필요하면 이후 `.cast("long")` 필요