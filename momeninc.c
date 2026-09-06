#include <stdio.h>
#include<stdlib.h>

double loss(double x[],double y[],int n,double m,double c){
    double ans=0;
    for(int i=0;i<n;i++){
        double term=y[i]-(m*x[i]+c);
        ans=ans+term*term;
    }
    return ans/n;
}

double mgrad(double x[],double y[],int n,double m,double c){
    double sum=0;
    for(int i=0;i<n;i++){
        double currx=x[i];
        double curry=y[i];
        sum=sum-currx*(curry-(m*currx + c));
    }
    return sum/n;
}

double cgrad(double x[],double y[],int n,double m,double c){
    double sum=0;
    for(int i=0;i<n;i++){
        double currx=x[i];
        double curry=y[i];
        sum=sum-(curry-(m*currx + c));
    }
    return sum/n;
}

void momentum(double* m,double* c,double x[],double y[],double lr,double alpha,int epoch,int n){
    double vm=0;
    double vc=0;
    for(int i=0;i<epoch;i++){
        double gradm=mgrad(x,y,n,*m,*c);
        double gradc=cgrad(x,y,n,*m,*c);
        vm=alpha*vm-lr*gradm;
        vc=alpha*vc-lr*gradc;
        *c=*c+vc;
        (*m)=*m+vm;
        if(i%10==0){
            printf("loss on epoch %d is %f\n",i,loss(x,y,n,*m,*c));
        }
    }
}

int main(){
    double lr=0.001;
    double alpha=0.9;
    int epochs=1500;
    double x[4]={1.0,10.0,15.0,20.0};
    double y[4]={2.0,12.0,17.0,22.0};
    double m=0.0;
    double c=0.0;
    momentum(&m,&c,x,y,lr,alpha,epochs,4);
    //printf("value of m:%f, value of c:%f\n",m,c);
    //for(int i=0;i<4;i++){
    //    double val=m*x[i]+c;
    //    printf("%f ",val);
    //}
    return 0;
}