# 一、问题描述

> 1. 分别采用不同的算法(非分布式算法)例如一般算法、分治算法和Strassen算法等计算计 算矩阵两个300x300的矩阵乘积，并通过Perf工具分别观察cache miss、CPI、 mem_load 等性能指标
>
> 2. Consider a memory system with a level 1 cache of 32KB and DRAM of 512 MB with the processor operating at 1 GHz. The latency to L1 cache is one cycle and the latency to DRAM is 100 cycles. In each memory cycle, the processes fetches four words (cache line size is four words ). What is the peak achievable performance of a dot product of two vectors? Note: Where necessary, assume an optimal cache placement policy.
>
>     ```c++
>     /* dot product loop */
>     for (i = 0; i < dim; i++)
>     	dot_prod += a[i] * b[i];
>     ```
>
> 3. Now consider the problem of multiplying a dense matrix with a vector using a two-loop dot-product formulation. The matrix is of dimension 4K x 4K. (Each row of the matrix takes 16KB of storage.) What is the peak achievable performance of this technique using a two-loop dot-product based matrix-vector product?  
>
>     ```c++
>     /* matrix-vector product loop */
>     for (i = 0; i < dim; i++)
>     	for (j = 0; i < dim; j++)
>     		c[i] += a[i][j] * b[j];
>     ```

# 二、解决方案
## Ⅰ、 问题一

### 1. 算法描述

#### ①一般算法

最直白、直接的算法，即用一个矩阵的每一行的每个元素分别乘另一个矩阵每一列的每个元素，各个乘积相加得到结果矩阵的一个元素。具体实现即三重循环。
#### ②分治算法

不断地将矩阵分为四个矩阵分别计算，直到变为单个元素.

- 三个矩阵可以写成下面的格式：

$$
A=
\left[
\begin{matrix}{A_{11}}&A_{12}\\A_{21}&A_{22}
\end{matrix}
\right],
B=
\left[
\begin{matrix}{B_{11}}&B_{12}\\B_{21}&B_{22}
\end{matrix}
\right],
C=
\left[
\begin{matrix}{C_{11}}&C_{12}\\C_{21}&A_{22}
\end{matrix}
\right]
$$



- 那么相关的计算可以写成

$$
\left[
\begin{matrix}{C_{11}}&C_{12}\\C_{21}&A_{22}
\end{matrix}
\right]=
\left[
\begin{matrix}{A_{11}}&A_{12}\\A_{21}&A_{22}
\end{matrix}
\right]\times
\left[
\begin{matrix}{B_{11}}&B_{12}\\B_{21}&B_{22}
\end{matrix}
\right]\\
C_{11}=A_{11}\times B_{11}+ A_{12}\times B_{21}\\
C_{12}=A_{11}\times B_{12}+ A_{12}\times B_{22}\\
C_{21}=A_{21}\times B_{11}+ A_{22}\times B_{21}\\
C_{22}=A_{21}\times B_{12}+ A_{22}\times B_{22}
$$



- 同理A~11~等一些子矩阵也可以写成相关的子矩阵，就这样将矩阵不断分解为小矩阵进行计算，最后归并为一个矩阵。
    同时，这种方法要求矩阵的行列必须为2的n次方。

#### ③Strassen算法

Strassen算法同样是使用分治的思想解决问题，只不过，不同的是当矩阵的阶很大时就会采取一个递推式进行计算相关递推式。

计算如下表达式：
$$
S_1=B_{12}-B_{22}\\
S_2=A_{11}+A_{12}\\
S_3=A_{21}+A_{22}\\
S_4=B_{21}-B_{11}\\
S_5=A_{11}+A_{22}\\
S_6=B_{11}+B_{22}\\
S_7=A_{12}-A_{22}\\
S_8=B_{21}+B_{22}\\
S_9=A_{11}-A_{21}\\
S_10=B_{11}+B_{12}\\
$$

$$
P_1=A_{11}\times S_1\\
P_2=S_2\times B_{22}\\
P_3=S_3\times B_{11}\\
P_4=A_{22}\times S_4\\
P_5=S_5\times S_6\\
P_6=S_7\times S_8\\
P_7=S_9\times S_{10}\\
$$

那么最终的矩阵结果为：
$$
C_{11}=P_5+P_4-P_2+P_6\\
C_{12}=P_1+P_2\\
C_{21}=P_3+P_4\\
C_{22}=P_5+P_1-P_3-P_7
$$
其中A11，A12，A21，A22和B11，B12，B21，B22分别为两个乘数A和B矩阵的四个子矩阵。C11，C12，C21，C22为最终的结果C矩阵的四个子矩阵。

那么只需要 将这四个矩阵合并就是最终结果。

### 2. 算法实现

> **为了之后测试三种算法的公平性，我准备将三种算法全都写到一个类里，在测试用不同算法计算时，只需要调用类里的不同函数即可。**  
#### ① 类的定义与构造函数实现
其中，构造函数就是传入行数、列数并随机赋值
```c++
class Matrix{

private:
    int row;
    int col;
    vector<vector<int>> data;

public:
    Matrix(int row, int col) :data(row),row(row),col(col){
        for (int i = 0; i < row; i++){
            data[i].resize(col);
        }
        srand(time(0));
        for (int i = 0; i < row; i++){
            for (int j = 0; j < col; j++){
                data[i][j] = rand();
            }
        }
    }

    Matrix(int row1, int col1, int row2, int col2, const Matrix& a) :row(row2 - row1), col(col2 - col1),data(row){
        for (int i = 0; i < row; i++){
            data[i].resize(col);
        }

        for (int i = 0; i < row; i++){
            for (int j = 0; j < col; j++){
                data[i][j] = a.get(col1 + i, row1 + j);
            }
        }
    }
	
	~Matrix(){
    	this->row = 0;
    	this->col = 0;
	}
	
	Matrix(const Matrix& a){
        *this = a;
   }

	Matrix* cal1(const Matrix&); 
	Matrix* cal2(const Matrix&); 
	Matrix* cal3(const Matrix&);

    Matrix* operator+(const Matrix&);
    Matrix* operator-(const Matrix&);
    Matrix* operator*(const Matrix&);
    static Matrix* add(const Matrix*, const Matrix*);
    static Matrix* sub(const Matrix*, const Matrix*);
    static Matrix* plus(const Matrix*, const Matrix*,const Matrix*,const Matrix*);

    vector<int> operator[](const int);
    int get(const int,const int) const;
    void set(const int, const int, int);

    bool isSimilar(const Matrix& x);
    void clear(int);                              

};
```
#### ② 除计算乘积函数之外其他函数的实现
- 与另一个矩阵加减乘
  定义：

  ```c++
  Matrix* operator+(const Matrix&);
  Matrix* operator-(const Matrix&);
  Matrix* operator*(const Matrix&);
  ```

  实现：

  ```c++
  Matrix* Matrix::operator+(const Matrix& a)
  {
      if (!isSimilar(a)){
          return nullptr;
      }
  
      Matrix *ptr = new Matrix(a.row, a.col);
      for (int i = 0; i < row; i++){
          for (int j = 0; j < col; j++){
              ptr->set(i, j, this->get(i, j) + a.get(i, j));
          }
      }
  
      return ptr;
  }
  
  Matrix* Matrix::operator-(const Matrix& a)
  {
      if (!isSimilar(a)){
          return nullptr;
      }
  
      Matrix *ptr = new Matrix(a.row, a.col);
      for (int i = 0; i < row; i++){
          for (int j = 0; j < col; j++){
              ptr->set(i, j, this->get(i, j) - a.get(i, j));
          }
      }
  
      return ptr;
  }
  ```

由于不同方法乘法方法不同，乘法函数后面再说
- 两矩阵相加减

    ```c++
    Matrix* Matrix::add(const Matrix* p1, const Matrix* p2)
    {
        if (!(p1->col == p2->col && p1->row == p2->row)){
            return nullptr;
        }
    
        Matrix *ptr = new Matrix(p1->row, p1->col);
        for (int i = 0; i < p1->row; i++){
            for (int j = 0; j < p1->col; j++){
                ptr->set(i, j, (p1->get(i, j) + p2->get(i, j)));
            }
        }
    
        return ptr;
    }
    
    
    Matrix* Matrix::sub(const Matrix* p1, const Matrix* p2)
    {
        if (!(p1->col == p2->col && p1->row == p2->row)){
            return nullptr;
        }
    
        Matrix *ptr = new Matrix(p1->row, p1->col);
        for (int i = 0; i < p1->row; i++){
            for (int j = 0; j < p1->col; j++){
                ptr->set(i, j, (p1->get(i, j) - p2->get(i, j)));
            }
        }
    
        return ptr;
    }
    ```
- 将四个矩阵合并为一个矩阵

    ```c++
    Matrix* Matrix::plus(const Matrix* p1, const Matrix* p2,const Matrix* p3, const Matrix* p4)
    {
        //不符合可以进行合并的条件
        if (!(p1->row == p2->row && p2->col == p4->col && p4->row == p3->row && p1->col == p3->col)){
            return nullptr;
        }
    
        Matrix* ptr = new Matrix(p1->row + p3->row, p2->col + p1->col);
        ptr->clear(0);
    
        for (int i = 0; i < p1->row; i++){
            for (int j = 0; j < p1->col; j++){
                ptr->set(i, j, p1->get(i, j));
            }
        }
    
        for (int i = 0; i < p2->row; i++){
            for (int j = 0; j < p2->col; j++){
                ptr->set(i, j + p1->col, p2->get(i, j));
            }
        }
    
        for (int i = 0; i < p3->row; i++){
            for (int j = 0; j < p3->col; j++){
                ptr->set(i + p1->row, j, p3->get(i, j));
            }
        }
    
        for (int i = 0; i < p4->row; i++){
            for (int j = 0; j < p4->col; j++){
                ptr->set(p1->row + i, p1->col + j, p4->get(i, j));
            }
        }
    
        return ptr;
    }
    ```
#### ③三种计算方法实现
- 定义
```c++
Matrix* cal1(const Matrix&); 
Matrix* cal2(const Matrix&); 
Matrix* cal3(const Matrix&);
```
- 一般算法
  即最简单的三重循环

  ```c++
  Matrix* Matrix::cal1(const Matrix& a)
  {
      Matrix *ptr = new Matrix(row, a.col);
      ptr->clear();
      for (int i = 0; i < row; i++){
          for (int j = 0; j < a.col; j++){
              for (int k = 0; k < col; k++){
                  ptr->set(i, j, ptr->get(i, j) + get(i, k) * a.get(k, j));
              }
          }
      }
      return ptr;
  }
  ```
- 归并算法
  原理上面已说过，不断地将矩阵分为四个矩阵分别计算，直到变为单个元素

  ```c++
  Matrix* Matrix::cal2(const Matrix& a)
  {
  
  	if (a.row == 1)             //当前的矩阵为单个的元素
      {
          Matrix *ptr = new Matrix(a.row, a.col);
          ptr->clear((this->get(0, 0))*(a.get(0, 0)));
          return ptr;
      }
      //将第一个矩阵分解为四个子矩阵
      Matrix A11(0,0,(row+1)/2-1,(col+1)/2-1,*this);
      Matrix A12((row+1) / 2, 0, row, (col+1) / 2-1, *this);
      Matrix A21(0, (col+1) / 2, (row+1) / 2-1, col, *this);
      Matrix A22((row+1) / 2, (col+1) / 2, row, col, *this);
      //将第二个矩阵分解为四个子矩阵
      Matrix B11(0,0,(row+1)/2-1,(col+1)/2-1, a);
      Matrix B12((row+1) / 2, 0, row, (col+1) / 2-1, a);
      Matrix B21(0, (col+1) / 2, (row+1) / 2-1, col, a);
      Matrix B22((row+1) / 2, (col+1) / 2, row, col, a);
  
      Matrix *C11 = Matrix::add(A11.cal2(B11), A12.cal2(B21));
      Matrix *C12 = Matrix::add(A11.cal2(B12), A12.cal2(B22));
      Matrix *C21 = Matrix::add(A21.cal2(B11), A22.cal2(B21));
      Matrix *C22 = Matrix::add(A21.cal2(B12), A22.cal2(B22));
  
      //将C11，C12，C21，C22合并为一个完整的矩阵
      Matrix* ptr = Matrix::plus(C11, C12, C21, C22);
  
      return ptr;
  }
  ```
- Strassen算法

    原理上面也已经说过

    ```c++
    Matrix* Matrix::cal3(const Matrix& a)
    {
    	if (a.row < 2)
        {
            return this->cal1(a);
        }
        //将第一个矩阵分解为四个子矩阵
        Matrix A11(0,0,(row+1)/2-1,(col+1)/2-1,*this);
        Matrix A12((row+1) / 2, 0, row, (col+1) / 2-1, *this);
        Matrix A21(0, (col+1) / 2, (row+1) / 2-1, col, *this);
        Matrix A22((row+1) / 2, (col+1) / 2, row, col, *this);
        //将第二个矩阵分解为四个子矩阵
        Matrix B11(0,0,(row+1)/2-1,(col+1)/2-1, a);
        Matrix B12((row+1) / 2, 0, row, (col+1) / 2-1, a);
        Matrix B21(0, (col+1) / 2, (row+1) / 2-1, col, a);
        Matrix B22((row+1) / 2, (col+1) / 2, row, col, a);
    
        Matrix* S1 = B12 - B22;
        Matrix* S2 = A11 + A12;
        Matrix* S3 = A21 + A22;
        Matrix* S4 = B21 - B11;
        Matrix* S5 = A11 + A22;
        Matrix* S6 = B11 + B22;
        Matrix* S7 = A12 - A22;
        Matrix* S8 = B21 + B22;
        Matrix* S9 = A11 - A21;
        Matrix* S10 = B11 + B12;
    
        Matrix* P1 = B12 - B22;
        Matrix* P2 = B12 - B22;
        Matrix* P3 = B12 - B22;
        Matrix* P4 = B12 - B22;
        Matrix* P5 = B12 - B22;
        Matrix* P6 = B12 - B22;
        Matrix* P7 = B12 - B22;
    
        P1 = A11.cal3(*S1);
        P2 = S2->cal3(B22);
        P3 = S3->cal3(B11);
        P4 = A22.cal3(*S4);
        P5 = S5->cal3(*S6);
        P6 = S7->cal3(*S8);
        P7 = S9->cal3(*S10);
    
        Matrix *C11 = Matrix::sub(Matrix::add(P5, P4), Matrix::sub(P2, P6));
        Matrix *C12 = Matrix::add(P1, P2);
        Matrix *C21 = Matrix::add(P3, P4);
        Matrix *C22 = Matrix::sub(Matrix::add(P5, P1), Matrix::add(P3, P7));
    	
    	Matrix* ptr = Matrix::plus(C11, C12, C21, C22);
    
        return ptr;
    }
    ```
#### ④ 主函数

首先，三个文件的类是相同的，只是调用的函数不同，例如在分治算法中的乘法函数如下：
```c++
Matrix* Matrix::operator*(const Matrix& a)
{
    if (a.row != this->col){
        return nullptr;
    }
    return this->cal2(a);
}
```
只需要在最后的return不同函数即可
则主函数：

```c++
int main()
{
	Matrix a(255,255),b(255,255),c(255,255);
	Matrix& p=a;
	Matrix& q=b;
	Matrix* i=&c;
	i=p*q;
	cout<<"一般算法已完成"<<endl;
	return 0;
}
```
声明256x256的矩阵，之后便开始计算。

## Ⅱ、 问题二

- 假设所需要的数据均在cache内
- 那么所有的数据，共32KB即4K个word，大约需要$400\mu s$
- 这些数据计算，即每个向量有2K个 word，则共需要4K次计算，即1K个周期，时间是$1 \mu s$
- 那么总时间为$400\mu s +1 \mu s=401\mu s$
- 那么可以计算得到结果：$4K\div 401\mu s=99.75\ FLOPS$

## Ⅲ、 问题三

- 每一行占16KB，那么每两行就占满了cache，取完两行的 数据就要从DRAM中取数

- 那么取所有数据需要的时间：

  - 从cache中取的：在cache中取一次是4K个word，约$400\mu s$,一共需要取2K次，即$0.8$ s
  - 从DRAM：共$2\times 10^5 $个周期，$200\mu s$ 
  - 共$800200\mu s$
  
- 一共进行了$2\times {\left(4\times 10^3  \right)}^3 $次计算，等于$2^7\times 10^9 $

- 那么可计算出结果：
    $$
    \frac{2^7 \times 10^9}{800200\mu s}=159.96\ GFLPOS
    $$
    

# 三、实验结果

## 1. 问题一

① cache miss

<img src="https://www.hz-heze.com/wp-content/uploads/2020/04/338c197c479ebe2c807fb159c8f1f27.jpg" alt="338c197c479ebe2c807fb159c8f1f27" style="zoom:80%;" />

可以看出，一般算法最简单，没有递归所以cache miss最少；而分治算法和Strassen就相对多很多了，时间也长很多；而这两个相比由于Strassen算法更复杂，cache miss也就多一些。 

② IPC

<img src="https://www.hz-heze.com/wp-content/uploads/2020/04/2f27d1da84c5b25aec82e3298d8c1f0.jpg" alt="2cc171a9775c0b78c8eb47f23b5cb72" style="zoom:80%;" />

可以看到，一般算法的IPC为2.86，分治算法为2.21，Strassen为2.27。一般算法和另两种相比稍大，一个周期执行的指令稍多，这应该是因为运行的是256x256的矩阵，相对还是较小，所以相对有递归的算法更快一些。而另两个相比，可以看出Strassen比单纯的分治算法要快一些的。

再看branch-miss占的比率，一般算法同样是最少的，而另两者同样是Strassen更优。

③ mem_load 

<img src="https://www.hz-heze.com/wp-content/uploads/2020/04/2cc171a9775c0b78c8eb47f23b5cb72.jpg" alt="2f27d1da84c5b25aec82e3298d8c1f0" style="zoom:80%;" />

这方面三者均为0。



**问题二、三已在上面说明**
