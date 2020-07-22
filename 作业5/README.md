# 并行与分布式作业    五

| 年级 |       班级       |   学号   | 姓名 |
| :--: | :--------------: | :------: | :--: |
| 18级 | 计科（超算方向） | 18340052 | 何泽 |

[TOC]

## Ⅰ. Sparse Matrix

> ​		Consider a sparse matrix stored in the compressed row format (you may find a description of this format on the web or any suitable text on sparse linear algebra). Write an OpenMP program for computing the product of this matrix with a vector. Download sample matrices from the Matrix Market (http://math.nist.gov/MatrixMarket/) and test the performance of your implementation as a function of matrix size and number of threads.

### 1.运行环境

- 我的运行系统是`Win10`，编译器为`MinGW-w64`，版本如下：

  <img src="C:\Users\03031\iCloudDrive\大二下\并行与分布式\作业\作业5\图片\8.PNG" alt="8" style="zoom: 50%;" />

- 我使用的`IDE`为`JetBrains`的`Clion`，该`IDE`需要写`CMakeLists`，内容如下：

  ```cmake
  cmake_minimum_required(VERSION 3.16)
  project(Sparse_Matrix)
  
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_FLAGS "-fopenmp")
  
  add_executable(Sparse_Matrix main.cpp)
  add_executable(Sparse_Matrix2 main2.cpp)
  ```

  其中`c++`标准为`c++20`，并指明需要`openmp`，此外有两个`cpp`文件，`main.cpp`为同一矩阵用不同线程数运行，`main2.cpp`为不同大小的矩阵用相同线程数运行。

- `CPU`为`i7-9750H`，有`6`核`12`线程

  <img src="C:\Users\03031\iCloudDrive\大二下\并行与分布式\作业\作业5\图片\1.PNG" alt="1" style="zoom:67%;" />

### 2. 程序实现

- 首先矩阵从https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/psadmit/1138_bus.html下载，第一行为信息，第二行为行数、列数以及非零元素个数，从第三行开始每行为一个元素，格式为行数+列数+元素值；

- 那么首先将文件内容读到矩阵中来，并输出行数、列数和非零元素个数

  ```c++
  ifstream matrix;
  matrix.open("D:/1138_bus.txt");
  cout << "Reading from the file" << endl;
  char a[100];
  matrix.getline(a,100);						//读第一行信息
  int row,col,num;
  matrix>>row>>col>>num;						//读行数、列数和非零元素个数
  cout<<"There are "<<row<<" rols and "<<col<<" cows. "<<endl;
  cout<<"There are "<<num<<" entries."<<endl;
  static double m[size][size];				 //将文件中的矩阵读到m中
  for(int i=1;i<=num;i++){
      int p,q;
      double b;
      matrix>>p>>q>>b;
      m[p][q]=b;
  }
  ```

- 然后随机生成元素值为`0-10`的向量`n`

  ```c++
  static double n[size];
  for(int i=1;i<=row;i++) {
      n[i] = rand()%10;
  }
  ```

- 然后调用`omp_get_num_procs()`输出`CPU`线程数（非物理核心数）

  ```c++
  cout<<"There are "<<omp_get_num_procs()<<" cores in this computer."<<endl;
  ```

- 为了让效果更明显，我将矩阵向量乘法进行`10`次，并计时,计时的函数`GET_TIME`如下（我写成了`define`的形式）：

  ```c++
  #define GET_TIME(now) { \
     struct timeval t; \
     gettimeofday(&t, NULL); \
     now = t.tv_sec + t.tv_usec/1000000.0; \
  }
  ```

- 首先是串行计算，并输出`10`次的运行时间

  ```c++
  int test_times=10;
  GET_TIME(start);
  for(int w=0;w<test_times;w++) {
      for (int i = 1; i <= row; i++) {
          for (int u = 1; u <= col; u++) {
              double buf = 0;
              for (int y = 1; y <= row; y++) {
                  buf += n[y] * m[y][u];
              }
              res[i][u] = buf;
          }
      }
  }
  GET_TIME(end);
  double t1=end-start;
  cout<<"Serial run time: "<<t1<<"s."<<endl;
  ```

- 然后使用`OpenMP`计算，此外最外层循环表示线程数从`2-18`递增，每一个线程数的值计算一次并输出该线程数时的运行时间以及相对于串行计算的加速比

  ```c++
      for (int thread_num=2;thread_num<=18;thread_num++) {	//线程数递增循环
          GET_TIME(start);
          for(int w=0;w<test_times;w++) {					   //计算10次
              for (int i = 1; i <= row; i++) {
  #pragma omp parallel for num_threads(thread_num)schedule(guided)
                  for (int u = 1; u <= col; u++) {
                      double buf = 0;
                      for (int y = 1; y <= row; y++) {
                          buf += n[y] * m[y][u];
                      }
                      res[i][u] = buf;
                  }
              }
          }
          GET_TIME(end);
          double t2 = end - start;
          cout << "OpenMP run time: " << t2 << "s with " << thread_num << " cores running. SpeedUP is "<<t1/t2<<"."<< endl;
      }
  ```

以上便是同一矩阵用不同线程数运行的设计，对于不同大小的矩阵用相同线程数运行，我下载了https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/psadmit/psadmit.html下的四种稀疏矩阵，如下图：

<img src="C:\Users\03031\iCloudDrive\大二下\并行与分布式\作业\作业5\图片\9.PNG" alt="9" style="zoom:80%;" />

算法设计与上面相同，只不过是读不同的文件。

### 3. 运行结果

- 同一矩阵用不同线程数运行，输出不同线程数的运行时间以及相对于串行计算的加速比

  <img src="C:\Users\03031\iCloudDrive\大二下\并行与分布式\作业\作业5\图片\2.PNG" alt="2" style="zoom: 50%;" />

- 不同大小的矩阵用相同线程数运行，输出不同矩阵大小所对应的运行时间

  <img src="C:\Users\03031\iCloudDrive\大二下\并行与分布式\作业\作业5\图片\3.PNG" alt="3" style="zoom: 50%;" />

### 4. 结果可视化

- 我将上面的运行结果在`Matlab`中输出可视化结果图像，命令如下：

  <img src="C:\Users\03031\iCloudDrive\大二下\并行与分布式\作业\作业5\图片\7.PNG" alt="7" style="zoom:80%;" />

- 线程数—运行时间

  <img src="C:\Users\03031\iCloudDrive\大二下\并行与分布式\作业\作业5\图片\4.png" alt="4" style="zoom:150%;" />

- 线程数—加速比

  <img src="C:\Users\03031\iCloudDrive\大二下\并行与分布式\作业\作业5\图片\5.png" alt="5" style="zoom:150%;" />

- 矩阵大小-运行时间

  ![6](C:\Users\03031\iCloudDrive\大二下\并行与分布式\作业\作业5\图片\6.png)

### 5. 结果分析

  - 对于相同大小的矩阵，由于我的计算机有`6`个物理核心，所以以`6`个线程运行的时候速度最快，加速比最高，而超过`6`个线程之后由于线程间通信的时间开销导致加速比比`6`个的时候略小，但`6-18`个速度接近，加速比趋于平稳
  - 而对于不同大小的矩阵，矩阵越大运行时间越多

## Ⅱ. Producer-Consumer

> ​		Implement a producer-consumer framework in OpenMP using sections to create a single producer task and a single consumer task. Ensure appropriate synchronization using locks. Test your program for a varying number of producers and consumers.

### 1. 程序实现

- 首先设计一个类代表缓冲区，然后根据缓冲区的状态制定生产者消费者规则

- 首先，生产者、消费者不能对缓冲区同时访问，实现方法是用`OpenMP`的`critical`对缓冲区加锁，同一时间只能有一个线程访问

- 之后定义一个变量`p`表示缓冲区元素最大个数，定义变量`b`表示现有元素个数，用来记录缓冲区用来判断缓冲区是否满或者空，如果缓冲区空，`b==0`，消费者不能访问，忙等`sleep`；缓冲区满，`b==p-1`，生产者不能访问，忙等`sleep`。

- 上面就是缓冲区的实现描述，代码如下：

  ```c++
  const int p = 1000;
  
  class buf {
  public:
      buf():b(0){}
      void push(int n){
          while(b==(p - 1))
              sleep(1);
  #pragma omp critical
          {
              a[b++]=n;
          }
      }
      void pop(){
          while(b==0)
              sleep(1);
  #pragma omp critical
          {
              b--;
          }
      }
  private:
      int a[p];
      int b;
  }t;
  ```

- 生产者、消费者就是调用上面的push和pop

  ```c++
  void producer(int n) {
      for (int i = 0; i < n; ++i)
          t.push(i);
  }
  
  void consumer(int n) {
      for (int i = 0; i < n; ++i)
          t.pop();
  }
  ```

- 主函数就是调用上面两个函数，为了测试生产者消费者不同数量，我用变量n代表，从25开始每次循环乘2，直到800，并输出运行时间

  ```c++
  for(int n=25;n<=800;n*=2) {
      cout<<"The number of Producer-Consumer is "<<n<<". ";
      double start, end;
      GET_TIME(start);
  #pragma omp parallel sections
      {
  #pragma omp section
          {
  #pragma omp parallel for
              for (int i = 0; i < n; ++i) {
                  producer(i);
              }
          }
  #pragma omp section
          {
  #pragma omp parallel for
              for (int i = 0; i < n; ++i) {
                  consumer(i);
              }
          }
      }
      GET_TIME(end);
      cout<<"The time is "<<end-start<<"s."<<endl;
  }
  ```

### 2. 运行结果

<img src="C:\Users\03031\iCloudDrive\大二下\并行与分布式\作业\作业5\图片\10.PNG" alt="10" style="zoom: 50%;" />

### 3. 结果可视化

和上一个一样，使用Matlab输出图像结果：

<img src="C:\Users\03031\iCloudDrive\大二下\并行与分布式\作业\作业5\图片\11.png" alt="11"  />

### 4. 结果分析

可以看出，随着数量的增加，运行时间增长得也越来越快。 

## Ⅲ. MPI测量通信时延和带宽

> 利用MPI通信程序测试本地进程以及远程进程之间的通信时延和带宽。

### 1. 程序实现

- 总体思路就是两个线程之间传送数据，一个发一个收，统计发送的大小和接收需要的时间，从而计算出带宽和延时。

- 所以要求进程数必须是2的倍数，并提前分配好组合，最后统计平均值得出结果

- 主要代码如下：

  - 首先，主进程打印一些相关信息

    ```c
    if (my_rank == 0) {
            printf("There are %d threads totally in the test.\n", size);
            printf("The message size is %d bytes.\n", BUF_SIZE);
            printf("----------------------------------------------------\n");
    }
    ```

  - 对于前一半进程，发送并接收数据后计算时间并发送到主进程

    ```c
    double best = .0, worst = .99E+99, total = .0;
    double total_time = .0;
    for (int i = 0; i < TEST_TIMES; ++i) {
        double nbytes = sizeof(char) * BUF_SIZE;
        double start_time = MPI_Wtime();
        MPI_Send(&buffer, BUF_SIZE, MPI_CHAR, partner_rank, tag, MPI_COMM_WORLD);
        MPI_Recv(&buffer, BUF_SIZE, MPI_CHAR, partner_rank, tag, MPI_COMM_WORLD, &status);
        double end_time = MPI_Wtime();
        double run_time = end_time - start_time;
        double bw = (2 * nbytes) / run_time;
        total += bw;
        best = bw > best ? bw : best;
        worst = bw < worst ? bw : worst;
        total_time += run_time;
    }
    best /= 1000000.0;
    worst /= 1000000.0;
    double avg_bw = (total / 1000000.0) / TEST_TIMES;
    total_time /= TEST_TIMES;
    ```

    接下来，判断是不是主进程即rank=0，如果不是，则给主进程发送时间信息

    ```c
    double tmp_timings[4];
    tmp_timings[0] = best;
    tmp_timings[1] = avg_bw;
    tmp_timings[2] = worst;
    tmp_timings[3] = total_time;
    MPI_Send(tmp_timings, 4, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    ```

    如果是主进程，则接收各个进程发送过来的信息，并打印每一组的信息，最后，计算各组的平均值得出结果

    ```c++
    timings[0][0] = best;
    timings[0][1] = avg_bw;
    timings[0][2] = worst;
    timings[0][3] = total_time;
    
    double best_all = .0, worst_all = .0, avg_all = .0;
    double time_all = .0;
    for (int j = 1; j < size / 2; ++j) {
    	MPI_Recv(&timings[j], 4, MPI_DOUBLE, j, tag, MPI_COMM_WORLD, &status);
    }
    for (int j = 0; j < size / 2; ++j) {
    	printf("Test between %d and %d, best bandwidth is %lfMBps, worst bandwidth is %lfMBps, average bandwidth is %lfMBps, time is %lfs.\n", j, task_pair[j], timings[j][0], timings[j][2], timings[j][1], timings[j][3]);
    	best_all += timings[j][0];
    	avg_all += timings[j][1];
    	worst_all += timings[j][2];
    	time_all += timings[j][3];
    }
    	printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    	printf("Averagely, best bandwidth is %lfMBps, worst bandwidth is %lfMBps, average bandwidth is %lfMBps, time is %lfs.\n", best_all / (size / 2), worst_all / (size / 2),avg_all / (size / 2),  time_all / (size / 2));
    	printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    } 
    ```

  - 对于后一半进程，只需要先接收数据，再发出去即可

    ```c
    for (int i = 0; i < TEST_TIMES; ++i) {
    	MPI_Recv(&buffer, BUF_SIZE, MPI_CHAR, partner_rank, tag, MPI_COMM_WORLD, &status);
    	MPI_Send(&buffer, BUF_SIZE, MPI_CHAR, partner_rank, tag, MPI_COMM_WORLD);
    }
    ```

### 2. 运行

> 因为我没有多个节点，所以运行测试是在单机本地测试的，远程测试也只需要写host文件即可，代码都是一样的，所以本地测试原理一致，就用本地测试代替了。

- 使用`mpicc`编译再用`mpirun`运行，结果如下：

<img src="C:\Users\03031\iCloudDrive\大二下\并行与分布式\作业\作业5\图片\13.PNG" alt="13" style="zoom:150%;" />

我用了12个线程跑，每一组的结果都已输出，最后得到平均带宽为`10155MBps`，时延为`0.000315s`，测试结束。

## Ⅳ. 实验总结

这次实验使用了`OpenMP`和`MPI`完成了很多任务，让我对并行编程更加了解，使用起来也更加熟练。

