#include <iostream>
#include <omp.h>
#include <unistd.h>
#include <array>
#include <sys/time.h>

using namespace std;

#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}

const int value_max = 1000;

class buf {
public:
    buf() : cur(0) {}
    void push(int n) {
        while (cur == value_max - 1)
            sleep(1);
#pragma omp critical
        {
            a[cur++] = n;
        }
    }
    void pop() {
        while (cur == 0)
            sleep(1);
#pragma omp critical
        {
            cur--;
        }
    }
private:
    array<int, value_max> a;
    int cur;
}t;

void producer(int n) {
    for (int i = 0; i < n; ++i)
        t.push(i);
}

void consumer(int n) {
    for (int i = 0; i < n; ++i)
        t.pop();
}

int main() {
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
    return 0;
}