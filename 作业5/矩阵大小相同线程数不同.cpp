#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>
#include <cstdio>
#include <string>
#include <omp.h>
using namespace std;

#define size 1139
#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}

int main() {
    ifstream matrix;
    matrix.open("D:/1138_bus.txt");
    cout << "Reading from the file" << endl;
    char a[100];
    matrix.getline(a,100);
    int row,col,num;
    matrix>>row>>col>>num;
    cout<<"There are "<<row<<" rols and "<<col<<" cows. "<<endl;
    cout<<"There are "<<num<<" entries."<<endl;
    static double m[size][size];
    for(int i=1;i<=num;i++){
        int p,q;
        double b;
        matrix>>p>>q>>b;
        m[p][q]=b;
    }
    static double n[size];
    for(int i=1;i<=row;i++) {
        n[i] = rand()%10;
    }

    double start,end;
    static double res[size][size];
    srand((unsigned)time(NULL));
    cout<<"There are "<<omp_get_num_procs()<<" cores in this computer."<<endl;
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

    for (int thread_num=2;thread_num<=18;thread_num++) {
        GET_TIME(start);
        for(int w=0;w<test_times;w++) {
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
    matrix.close();

    return 0;
}
