#include <iostream>
#include <thread>
using namespace std;
int a[1000][1000];
int b[1000];
int c[1000];
void mul(int p);
int main() {
    for(int i=0;i<1000;i++){
        for(int j=0;j<1000;j++){
            a[i][j]=rand()+100;
        }
    }
    for(int j=0;j<1000;j++){
        b[j]=rand()+100;
    }
    thread threads[1000];
    for (int i = 0; i < 1000; i++) {
        threads[i] = thread(mul, i );
    }
    for (int i = 0; i < 1000; i++) {
        threads[i] .join();
    }
    return 0;
}
void mul(int p)
{
    int sum=0;
    for(int i=0;i<1000;i++)
        sum+=a[p][i]*b[i];
    c[p]=sum;
}
