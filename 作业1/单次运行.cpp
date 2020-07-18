#include <iostream>
#include <cstdlib>
#include <ctime>
#include <windows.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <thread>

using namespace std;

int a[1000000],b[1000000],c[1000000];
void add(int a);	//This is the function that the thread called.

int main()
{   
	//Following is the serial process.
	double start,end,t1,t2,t3;
	srand( (int)time(0) );
	cout<<"-------------- Serial process is starting. --------------"<<endl;
	for(int i=0;i<999999;i++){
		a[i]=rand()+1000000;
		b[i]=rand()+1000000;
	}
	start=clock();
	for(int i=0;i<999999;i++){
		c[i]=a[i]+b[i]; 
	}
	end=clock();
    t1=(double)(end-start);
	printf("The time of serial process is %lf\n\n\n",t1);
	
	
	
	//Following is the parallel process.
	__m128i m,n,p;
	cout<<"------------- Parallel process is starting. -------------"<<endl;
	for(int i=0;i<999997;i+=4){
		m=_mm_set_epi32(a[i],a[i+1],a[i+2],a[i+3]);
		n=_mm_set_epi32(b[i],b[i+1],b[i+2],b[i+3]); 
		start=clock();
		p=_mm_add_epi32(m,n);
		end=clock();
		t2+=(double)(end-start);
	}
    printf("The time of parallel process is %lf\n\n",t2);
    printf("The speedup of parallel process is %lf\n\n\n",t1/t2);
	
	
	
	//Following is the multithreading.
	cout<<"-------------- Multithreading is starting. --------------"<<endl;
	auto core=thread::hardware_concurrency();
	cout<<"The number of the CPU threads is  "<<core<<endl;
	start=clock();
	thread th1(add,0);
	thread th2(add,250000);
	thread th3(add,500000);
	thread th4(add,750000);
	th1.join();
	th2.join();
	th3.join();
	th4.join();
	end=clock();
    t3=(double)(end-start);
    printf("The time of multithreading is %lf\n\n",t3);
    printf("The speedup of multithreading is %lf\n\n\n",t1/t3);
    return 0;
}

//The following is the function that the thread called.
void add(int p)
{
	for(int i=p;i<p+250000;i++){
		c[i]=a[i]+b[i]; 
	}
}
