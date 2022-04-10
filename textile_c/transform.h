#ifndef __TRANSFORM2_H__
#define __TRANSFORM2_H__
#ifdef __cplusplus
extern "C"{
#endif
void printMatrix(double* , int , int , int );
void transform(double* , double* , double* , int , int , int* , int , int , int*);
double sum(double* , int );
double ip(double* , double* , int );
double wip(double* , double* , double* , int );
void subMatrix(double* , double* , int* , int );
int Dsygv(double* , double* , int , int , double* , double , int* );
#ifdef __cplusplus
}
#endif

#endif	//__TRANSPOSE2_H__
