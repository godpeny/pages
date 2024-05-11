# Language
## Interpreted vs Compiled

## Python
### GIL
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




## Golang
### Goroutine
### Channel
### Generic
### Interface and Type Assertion


