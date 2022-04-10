#include <stdio.h>
#include "blaswrap.h"
#include "f2c.h"
#include "clapack.h"
#include "transform.h"
#define TINY 1.0e-10


/* sample main start */
int main2(void){
	double x[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
	//printMatrix(x, 10, 4, 10);
	//printMatrix(w, 10, 3, 10);
	double y[17];
	int q[] = {0,1};
	
	subMatrix(x, y, q, 4);
	printMatrix(x, 4,4,4);
	printMatrix(y, 2, 2, 4);
	return 0;
}

int main(void){
	double x[] = {
-0.001271673,-0.008055735, 0.997513967, 0.000000000,-0.466050483,
-2.831182321,-0.295235185, 0.321523554, 0.452224977, 0.000000000,
 1.094754192, 0.516446048,-0.038293932,-2.447391202,-0.735928229,
-0.004057757, 0.000000000,-0.657263241, 0.055907470, 0.000000000,
 0.000000000, 0.000000000, 1.000000000, 0.000000000, 0.000000000,
 1.000000000, 1.000000000, 1.000000000, 0.000000000, 1.000000000,
 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000,
 1.000000000, 0.000000000, 1.000000000, 0.000000000, 0.000000000};
	double w[] = {1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,0,1,0,1,1,1,1,1,1,1,1};
	//printMatrix(x, 10, 4, 10);
	//printMatrix(w, 10, 3, 10);
	double y[51];
	int q[] = {0,1,2,4};
	int ordered[] = {0};
	
	transform(x, w, y, 10, 3, q, 4, 1, ordered);
	printMatrix(y, 10, 4, 10);
	return 0;
}

void printMatrix(double* m, int n, int p, int ln){
	int i, j;
	for(i=0; i<n; i++){
		for(j=0; j<p; j++){
			if(n==p){
				if(j>=i){
					printf("%.5f ", m[j*ln+i]);
				}else{
					printf("*\t");
				}
			}else{
				printf("%.5f ", m[j*ln+i]);
			}
		}
		printf("\n");
	}
	printf("\n");
}


/* sample main end */




void subMatrix(double* a, double* b, int* q, int p){//a is a p x p matrix.
	int i, j, k, l, r, s;
	l=0;
	r=0;
	for(i=0; i<p; i++){
		if(q[l] == i){
			l++;
		}else{
			k=0;
			s=0;
			for(j=0; j<=i; j++){
				if(q[k] == j){
					k++;
				}else{
					b[r*p+s] = a[i*p+j];
					s++;
				}
			}
			r++;
		}
	}
}


double sum(double* a, int m){
	double res=0;
	int i;
	for(i=0; i<m; i++){
		res += a[i];
	}
	return(res);
}


double ip(double* a, double* b, int m){
	double res=0;
	int i;
	for(i=0; i<m; i++){
		res += a[i]*b[i];
	}
	return(res);
}
double wip(double* a, double* b, double* c, int m){
	double res=0;
	int i;
	for(i=0; i<m; i++){
		res += a[i]*b[i]/c[i];
	}
	return(res);
}

void ginv(double* ainv, long int n){
	int i, j, k;
	char jobz='V';
	char uplo='U';
	
	long int lda;
	double w[n+1];
	double work[3*n];
	double a[n*n+1];
	long int lwork;
	long int info;
	
	lda=n;
	lwork=3*n-1;
	
	for(i=0; i<n*n; i++){
		a[i] = ainv[i];
		ainv[i] = 0.0;
	}
	
	dsyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info);
	//printMatrix(a, n, 2, n);
	
	for(i=0; i<n; i++){
		if(w[i]>TINY){
			for(j=0; j<n; j++){
				for(k=0; k<=j; k++){
					ainv[j*n+k] += a[i*n+j]*a[i*n+k]/w[i];
				}
			}
		}
	}
}

void ca(double* a11, double* a12, double* a22, int p, int Q){
	int i, j, k;
	double temp[p*Q+1];
	double temp2;
	
	for(i=0; i<Q; i++){
		for(j=0; j<p; j++){
			temp[i*p+j] = 0.0;
			for(k=0; k<p; k++){
				if(j>k){
					temp[i*p+j] += a11[j*p+k]*a12[i*p+k];
				}else{
					temp[i*p+j] += a11[k*p+j]*a12[i*p+k];
				}
			}
		}
	}
	//printMatrix(temp, p, Q, p);
	for(i=0; i<Q; i++){
		for(j=0; j<=i; j++){
			temp2 = 0.0;
			for(k=0; k<p; k++){
				temp2 += a12[j*p+k]*temp[i*p+k];
			}
			a22[i*Q+j] = temp2 - a22[i*Q+j];
		}
	}
	for(i=0; i<p*Q; i++){
		a12[i] = temp[i];
	}
}

int Dsygv(double* a, double* b, int p, int Q, double* ev, double maxev, int* comp){
	
	long int itype = 1;
	char jobz = 'V';
	char uplo = 'U';
	
	long int lda = Q;
	long int ldb = Q;
	long int pp = p;
	
	double work[3*Q];
	long int lwork = 3*Q;
	long int info;
	
	if(0){
		double suba[p*Q+1];
		double subb[p*Q+1];
		
		subMatrix(a, suba, comp, Q);
		
		
		dsygv_(&itype, &jobz, &uplo, &pp, suba, &lda, subb, &ldb, ev, work, &lwork, &info);
		if(0){
			
		}else{
			if(maxev < ev[p]){
				printf("hoge");
			}else{
				
			}
		}
	}else{
		dsygv_(&itype, &jobz, &uplo, &pp, a, &lda, b, &ldb, ev, work, &lwork, &info);
	}
	return info;
}

void transform(double* xx, double* ww, double* y, int n, int p, int* q, int Q, int dim, int* ordered){
	
	/*
	n       (INPUT)   INTEGER
	        The number of recoreds.
	        
	p       (INPUT)   INTEGER
	        The number of variables.
	        
	q       (INPUT)   INTEGER * (p + 1)
	        (p+1)-dimensional array. XÉš¯é0©çnÜéeÏÊÌæªÌñÔ. 
	        
	Q       (INPUT)   INTEGER
	        The number of colmuns of encoded data matrix X.
	        
	xx      (INPUT)   DOUBLE * (n * Q)
	        An n x Q encoded data matrix X.
	        R[fBO³êœf[^sñ
	        
	ww      (INPUT)   DOUBLE * (n * p)
	        An n x p weight matrix W, reflecting missing informations.
	        ¹Å0C»êÈOÅ1ÌdÝsñ
	        
	y       (OUTPUT)  DOUBLE * (n * p)
	        An n x p transformed data matrix Y.
	        
	dim     (INPUT)   INTEGER
	        The dimension you want to choose, ¡ÍÆè Šž1.
	        
	ordered (INPUT)   INTEGER * r
	        r{ÌöqªÜÜêÄ¢éêC»êçÌÏÊÔðzñÅ^Šé.
	        àµöqªÜÜêÈ¯êÎæêvfÉ-1ðZbgD
	*/
	
	double* X[Q+1]; //X   n x Q
	double* W[p+1]; //W   n x p
	double dd[n+1]; //w = Z*t(Z)*W*1
	double d[n+1];  //w = Z*t(Z)*W*1
	double x[n*Q+1];
	double w[n*p+1];
	double alpha;
	//Z   Indicator Matrix of Target Object
	double a11[p*p+1]; //A11 p x p
	double a12[p*Q+1]; //A12 p x Q
	double a22[Q*Q+1]; //A22 Q x Q
	double b[Q*Q+1]; //B
	
	double ev[Q+1];
	double maxev=0.0;
	
	int i, j, k, l, m;
	double temp;
	for(i=0; i<n; i++){// making d
		dd[i] = 0;
		for(j=0; j<p; j++){
			dd[i] += ww[n*j+i];
		}
	}
	
	m=0;
	for(i=0; i<n; i++){
		if(dd[i]>0.0){
			for(j=0; j<p; j++){
				w[j*n+m] = ww[j*n+i];
				for(k=q[j]; k<q[j+1]; k++){
					x[k*n+m] = xx[k*n+i]*ww[j*n+i];
				}
			}
			d[m] = dd[i];
			m++;
		}
	}
	
	for(j=0; j<Q; j++){
		X[j] = &x[j*n];
	}
	for(j=0; j<p; j++){
		W[j] = &w[j*n];
	}
	
	
	temp=0.0;
	for(i=0; i<p; i++){
		for(j=0; j<=i; j++){
			if(i==j){
				a11[i*p+j] = -wip(W[i], W[j], d, m) + sum(W[j], m);
				for(k=q[j]; k<q[j+1]; k++){
					for(l=q[j]; l<=k; l++){
						a22[k*Q+l] = -wip(X[k], X[l], d, m) + sum(X[k], m)*sum(X[l], m)/sum(W[j], m);
						b[k*Q+l] = ip(X[k], X[l], m) - sum(X[k], m)*sum(X[l], m)/sum(W[j], m);
					}
				}
			}else{
				a11[i*p+j] = -wip(W[i], W[j], d, m);
				for(k=q[i]; k<q[i+1]; k++){
					for(l=q[j]; l<q[j+1]; l++){
						a22[k*Q+l] = -wip(X[k], X[l], d, m);
						b[k*Q+l] = 0;
					}
				}
			}
		}
		for(j=0; j<p; j++){
			if(i==j){
				for(k=q[i]; k<q[i+1]; k++){
					a12[k*p+j] = wip(W[j], X[k], d, m) - ip(W[j], X[k], m);
				}
			}else{
				for(k=q[i]; k<q[i+1]; k++){
					a12[k*p+j] = wip(W[j], X[k], d, m);
				}
			}
		}
	}
	//printMatrix(a11, p, p, p);
	//printMatrix(a12, p, Q, p);
	//printMatrix(a22, Q, Q, Q);
	ginv(a11, p);
	//printMatrix(a11, p, p, p);
	ca(a11, a12, a22, p, Q);
	//printMatrix(b, Q, Q, Q);

	//dsygv_(&itype, &jobz, &uplo, &nn, a22, &lda, b, &ldb, &ev, &work, &lwork, &info);
	Dsygv(a22, b, Q, Q, ev, maxev, ordered);
	//printf("info=%d\n", info);
	
	dim = Q*(Q-dim);
	
	for(i=0; i<n; i++){
		y[i] = 0.0;
	}
	
	for(j=0; j<p; j++){
		alpha = 0.0;
		for(k=0; k<Q; k++){
			alpha += a12[p*k+j]*a22[dim+k];
			
		}
		//printf("%f\n", alpha);
		for(i=0; i<n; i++){
			y[(j+1)*n+i] = 0.0;
			for(k=q[j]; k<q[j+1]; k++){
				y[(j+1)*n+i] += xx[k*n+i]*a22[dim+k];
			}
			y[(j+1)*n+i] += alpha;
			y[i] += y[(j+1)*n+i]*w[j*n+i]/p;
		}
	}
}
