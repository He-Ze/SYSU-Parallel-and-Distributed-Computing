#include <iostream> 
#include <thread>
#include <mutex>
using namespace std;
 
int sum = 0;

void work(int index)
{
	for (int i = 0; i < 100; i++) {
		sum++;
	}
}

int main()
{
	thread t[tCount];
	for (int n = 0; n < 2; n++){
		t[n] = thread(work, n);
    }
	for (int n = 0; n < 2; n++){
		t[n].join();
	}
	cout << sum << endl;
	return 0;
}
