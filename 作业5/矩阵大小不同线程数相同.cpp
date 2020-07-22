#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>
#include <cstdio>
#include <string>
#include <omp.h>
using namespace std;

#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}

int main() {
    ifstream matrix;

    matrix.open("D:/1138_bus.txt");
    cout<<"******************************************************************"<<endl;
    cout << "Reading from the file 1138_bus.txt." << endl;
    int size=1139;
    char a[100];
    matrix.getline(a,100);
    int row,col,num;
    matrix>>row>>col>>num;
    cout<<"There are "<<row<<" rols and "<<col<<" cows. "<<endl;
    cout<<"There are "<<num<<" entries."<<endl;
    static double m[1139][1139];
    for(int i=1;i<=num;i++){
        int p,q;
        double b;
        matrix>>p>>q>>b;
        m[p][q]=b;
    }
    static double n[1139];
    for(int i=1;i<=row;i++) {
        n[i] = rand()%10;
    }
    double start,end;
    static double res[1139][1139];
    srand((unsigned)time(NULL));
    int test_times=10;
    GET_TIME(start);
    for(int w=0;w<test_times;w++) {
        for (int i = 1; i <= row; i++) {
#pragma omp parallel for num_threads(6)schedule(guided)
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
    cout << "OpenMP run time: " << t2 << "s with 6 cores running. "<< endl;
    cout<<"******************************************************************"<<endl;
    matrix.close();



    matrix.open("D:/685_bus.txt");
    cout<<"******************************************************************"<<endl;
    cout << "Reading from the file 685_bus.txt." << endl;
    size=686;
    matrix.getline(a,100);
    matrix>>row>>col>>num;
    cout<<"There are "<<row<<" rols and "<<col<<" cows. "<<endl;
    cout<<"There are "<<num<<" entries."<<endl;
    for(int i=1;i<=num;i++){
        int p,q;
        double b;
        matrix>>p>>q>>b;
        m[p][q]=b;
    }
    for(int i=1;i<=row;i++) {
        n[i] = rand()%10;
    }
    srand((unsigned)time(NULL));
    GET_TIME(start);
    for(int w=0;w<test_times;w++) {
        for (int i = 1; i <= row; i++) {
#pragma omp parallel for num_threads(6)schedule(guided)
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
    t2 = end - start;
    cout << "OpenMP run time: " << t2 << "s with 6 cores running. "<< endl;
    cout<<"******************************************************************"<<endl;
    matrix.close();



    matrix.open("D:/662_bus.txt");
    cout<<"******************************************************************"<<endl;
    cout << "Reading from the file 662_bus.txt." << endl;
    size=663;
    matrix.getline(a,100);
    matrix>>row>>col>>num;
    cout<<"There are "<<row<<" rols and "<<col<<" cows. "<<endl;
    cout<<"There are "<<num<<" entries."<<endl;
    for(int i=1;i<=num;i++){
        int p,q;
        double b;
        matrix>>p>>q>>b;
        m[p][q]=b;
    }
    for(int i=1;i<=row;i++) {
        n[i] = rand()%10;
    }
    srand((unsigned)time(NULL));
    GET_TIME(start);
    for(int w=0;w<test_times;w++) {
        for (int i = 1; i <= row; i++) {
#pragma omp parallel for num_threads(6)schedule(guided)
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
    t2 = end - start;
    cout << "OpenMP run time: " << t2 << "s with 6 cores running. "<< endl;
    cout<<"******************************************************************"<<endl;
    matrix.close();


    matrix.open("D:/494_bus.txt");
    cout<<"******************************************************************"<<endl;
    cout << "Reading from the file 494_bus.txt." << endl;
    size=495;
    matrix.getline(a,100);
    matrix>>row>>col>>num;
    cout<<"There are "<<row<<" rols and "<<col<<" cows. "<<endl;
    cout<<"There are "<<num<<" entries."<<endl;
    for(int i=1;i<=num;i++){
        int p,q;
        double b;
        matrix>>p>>q>>b;
        m[p][q]=b;
    }
    for(int i=1;i<=row;i++) {
        n[i] = rand()%10;
    }
    srand((unsigned)time(NULL));
    GET_TIME(start);
    for(int w=0;w<test_times;w++) {
        for (int i = 1; i <= row; i++) {
#pragma omp parallel for num_threads(6)schedule(guided)
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
    t2 = end - start;
    cout << "OpenMP run time: " << t2 << "s with 6 cores running. "<< endl;
    cout<<"******************************************************************"<<endl;
    matrix.close();
    return 0;
}
