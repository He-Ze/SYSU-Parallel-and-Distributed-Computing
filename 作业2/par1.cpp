#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime> 

using namespace std;


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

Matrix* Matrix::plus(const Matrix* p1, const Matrix* p2,const Matrix* p3, const Matrix* p4)
{
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


vector<int> Matrix::operator[](const int row)
{
    return data[row];
}

int Matrix::get(int row, int col)const
{
    return this->data[row][col];
}

void Matrix::set(int row, int col, int a)
{
    this->data[row][col] = a;
}

bool Matrix::isSimilar(const Matrix& a)
{
    return a.row == this->row && this->col == a.col;
}

void Matrix::clear(int a=0)
{
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            data[i][j] = a;
        }
    }
}


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


Matrix* Matrix::cal2(const Matrix& a)
{

	if (a.row == 1)
    {
        Matrix *ptr = new Matrix(a.row, a.col);
        ptr->clear((this->get(0, 0))*(a.get(0, 0)));
        return ptr;
    }
    Matrix A11(0,0,(row+1)/2-1,(col+1)/2-1,*this);
    Matrix A12((row+1) / 2, 0, row, (col+1) / 2-1, *this);
    Matrix A21(0, (col+1) / 2, (row+1) / 2-1, col, *this);
    Matrix A22((row+1) / 2, (col+1) / 2, row, col, *this);
    Matrix B11(0,0,(row+1)/2-1,(col+1)/2-1, a);
    Matrix B12((row+1) / 2, 0, row, (col+1) / 2-1, a);
    Matrix B21(0, (col+1) / 2, (row+1) / 2-1, col, a);
    Matrix B22((row+1) / 2, (col+1) / 2, row, col, a);

    Matrix *C11 = Matrix::add(A11.cal2(B11), A12.cal2(B21));
    Matrix *C12 = Matrix::add(A11.cal2(B12), A12.cal2(B22));
    Matrix *C21 = Matrix::add(A21.cal2(B11), A22.cal2(B21));
    Matrix *C22 = Matrix::add(A21.cal2(B12), A22.cal2(B22));

    Matrix* ptr = Matrix::plus(C11, C12, C21, C22);

    return ptr;
}

Matrix* Matrix::cal3(const Matrix& a)
{
	if (a.row < 2)
    {
        return this->cal1(a);
    }
    Matrix A11(0,0,(row+1)/2-1,(col+1)/2-1,*this);
    Matrix A12((row+1) / 2, 0, row, (col+1) / 2-1, *this);
    Matrix A21(0, (col+1) / 2, (row+1) / 2-1, col, *this);
    Matrix A22((row+1) / 2, (col+1) / 2, row, col, *this);
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

Matrix* Matrix::operator*(const Matrix& a)
{
    if (a.row != this->col){
        return nullptr;
    }

    return this->cal1(a);
}



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
