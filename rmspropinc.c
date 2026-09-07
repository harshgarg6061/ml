#include <stdio.h>
#include<stdlib.h>
#include<math.h>

void grad(double*gradm,double*gradc,double*x,double*y,double m,double c,int n){
    for(int i=0;i<n;i++){
        *gradm-=x[i]*(y[i]-(m*x[i]+c));
        *gradc-=(y[i]-(m*x[i]+c));
    }
    *gradm/=n;
    *gradc/=n;
}

double loss(double*x,double*y,double m,double c,int n){
    double ans=0.0;
    for(int i=0;i<n;i++){
        double y_=y[i]-(m*x[i]+c);
        ans+=y_*y_;
    }
    return ans/n;
}

void rmsprop(double*x,double*y,int n,double* m,double* c,int epochs,double lr,double rho){
    double rm=0.0;double rc=0.0;
    double dell=0.0000001;
    for(int i=0;i<=epochs;i++){
        if(i%100==0){
            printf("loss on %d epoch is %f\n",i,loss(x,y,*m,*c,n));
        }
        double gradm=0.0;double gradc=0.0;double vm=0.0;double vc=0.0;
        grad(&gradm,&gradc,x,y,*m,*c,n);
        rm=rho*rm+(1-rho)*gradm*gradm;
        rc=rho*rc+(1-rho)*gradc*gradc;
        vm=-lr*gradm/(dell+sqrt(rm));
        vc=-lr*gradc/(dell+sqrt(rc));
        *m+=vm;*c+=vc;
    }
}

int main(){
    double x[5]={1.0,5.0,10.0,15.0,20.0};
    double y[5]={2.0,8.0,22.0,31.2,43.3};
    double lr=0.005;double rho=0.9;
    int epochs=2000;
    double m=0.0;double c=0.0;
    rmsprop(x,y,5,&m,&c,epochs,lr,rho);
    printf("final valus of m:%f and c:%f",m,c);
    return 0;
}