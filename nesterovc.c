#include <stdio.h>

double loss(double x[],double y[],int n,double m,double c){
    double ans=0;
    for(int i=0;i<n;i++){
        double term=y[i]-(m*x[i]+c);
        ans=ans+term*term;
    }
    return ans/n;
}

void grads(double *x,double *y,double m,double c,int n,double* mgram,double* cgram){
    *mgram=0.0;
    *cgram=0.0;
    for(int i=0;i<n;i++){
        *mgram-=x[i]*(y[i]-m*x[i]-c);
        *cgram-=(y[i]-m*x[i]-c);
    }
    *mgram=*mgram/n;
    *cgram=*cgram/n;
}

void nesterov(double*x,double*y,double* m,double* c,double lr,double alpha,int epochs,int n){
    double vm=0;
    double vc=0;
    for(int i=0;i<epochs;i++){
        double m_hat=*m+alpha*vm;
        double c_hat=*c+alpha*vc;
        double mgrad=0;double cgrad=0;
        grads(x,y,m_hat,c_hat,n,&mgrad,&cgrad);//only difference is where gradient is being calculted here we calulate gradient after adding the velocity component, we are kinda looking ahead in the direction of velocity
        vm=alpha*vm-lr*mgrad;
        vc=alpha*vc-lr*cgrad;
        *m+=vm;
        *c+=vc;
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
    nesterov(x,y,&m,&c,lr,alpha,epochs,4);
    //printf("value of m:%f and value of c:%f\n",m,c);
    //for(int i=0;i<4;i++){
    //    double val=m*x[i]+c;
    //   printf("%f ",val);
    //}
    //printf("\nloss=%f",loss(x,y,4,m,c));
    return 0;
}