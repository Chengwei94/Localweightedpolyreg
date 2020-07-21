// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "stdafx.h"
#include <RcppEigen.h>
#include "alglibmisc.h"
#include <map>
#include <string>
#include <iostream>
#include <math.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Rcpp::as;
using Rcpp::List;


// [[Rcpp::depends(RcppEigen)]]
inline VectorXd query(alglib::kdtree &kdt, 
                      const VectorXd& x0,
                      const double& h)
{
    int p = x0.size(); 
    
    alglib::real_1d_array bmin, bmax;
    bmin.setlength(p);
    bmax.setlength(p);
    
    for(int i=0; i<p; i++)
    {
        bmin[i] = x0(i) - h;
        bmax[i] = x0(i) + h;
    }
    
    int k = kdtreequerybox(kdt, bmin, bmax);
    alglib::integer_1d_array r;
    alglib::kdtreequeryresultstags(kdt,r);
    
    VectorXd output(r.length());
    for(int i=0; i < r.length(); i++)
    {
        output[i] = r[i];
    }
    
    return output;
}

inline double eval_kernel(int kcode, double z)
{
    double tmp;
    switch(kcode)
    {
    case 1: return 3*(1-z*z)/4; // Epanechnikov
    case 2: return 0.5; // rectangular
    case 3: return 1-abs(z); // triangular
    case 4: return 15*(1-z*z)*(1-z*z)/16; // quartic
    case 5: 
        tmp = 1-z*z;
        return 35*tmp*tmp*tmp/32; // triweight
    case 6: 
        tmp = 1- z*z*z;
        return 70 * tmp * tmp * tmp / 81; // tricube
    case 7:
        return M_PI * cos(M_PI*z/2) / 4; // cosine
    case 21:
        return exp(-z*z/2) / sqrt(2*M_PI); // gauss
    case 22:
        return 1/(exp(z)+2+exp(-z)); // logistic
    case 23:
        return 2/(M_PI*(exp(z)+exp(-z))); // sigmoid
    case 24:
        return exp(-abs(z)/sqrt(2)) * sin(abs(z)/sqrt(2)+M_PI/4); // silverman
    default: Rcpp::stop("Unsupported kernel"); 
    }
    
    return 0;
}   


// [[Rcpp::export]]
Eigen::VectorXd wlocpoly(const Eigen::MatrixXd x,
                   const Eigen::VectorXd y, 
                   const Eigen::MatrixXd newx,
                   const int d, 
                   const std::string kernel_type, 
                   const double h) 
{
    if(d < 0 || d > 2) Rcpp::stop("Only d=0,1,2 is supported");
    if(x.rows() != y.size()) Rcpp::stop("length of y must be equal to the number of rows of x");
    
    std::map<std::string,int> supported_kernels;
    supported_kernels["epanechnikov"]    = 1;   
    supported_kernels["rectangular"]    = 2;
    supported_kernels["triangular"]   = 3;   
    supported_kernels["quartic"]    = 4;   
    supported_kernels["triweight"]    = 5;
    supported_kernels["tricube"]   = 6; 
    supported_kernels["cosine"] = 7;
    supported_kernels["gauss"]   = 21;
    supported_kernels["logistic"] = 22;
    supported_kernels["sigmoid"] = 23;
    supported_kernels["silverman"] = 24;
    
    alglib::ae_int_t n = x.rows(); 
    alglib::ae_int_t p = x.cols();
    
    alglib::kdtree kdt; 
    
    int kcode = 1; 
    if(kcode <= 20)
    {
        alglib::real_2d_array a; 
        a.setlength(n,p);
        for (int i =0 ; i<n ; i++)
        {
            for(int j =0 ; j<p; j++)
                a(i,j) = x(i,j);
        }
        alglib::real_1d_array b;
        b.setlength(n); 
        for(int i = 0; i<n ; i++)
            b(i) = x(i); 
        
        ptrdiff_t *pTag = new ptrdiff_t[n];
        for(int i=0; i < n; i++)
        {
            pTag[i] = i;
        }
         alglib::integer_1d_array tags;
         tags.setcontent(n, pTag);
        
        delete[] pTag; 
        
        alglib::kdtreebuildtagged(a,tags,p,0,2,kdt);
        
        Rcpp::Rcout<< "Tree built successfully" << std::endl;
    }
    
    VectorXd result(newx.rows());

    for (int i =0; i < newx.rows(); i++)
    {
        const VectorXd x0 = newx.row(i);
        
        VectorXd idx;
        idx = query(kdt, x0, h);  // retrieve vector of tags 
        
        int m = idx.rows(); // number of items within distance 

        int D = 1 + p;  // local linear (fix as local lienar at first)
       
        MatrixXd X = MatrixXd::Constant(m,D,1);
        MatrixXd W = MatrixXd::Identity(m,m);
        VectorXd Y = MatrixXd::Constant(m,1,0);

        Rcpp::Rcout  << "Matrix created successfully for point" << i <<std::endl; 
        
        for(int j =0; j<m ; j++)
            Y(j,0) = y(idx[j]); 
        
        for(int v=0; v < m; v++) // for each data point within th bandwidth
        {
            VectorXd x1 = x.row(idx[v]);
            double dist1 = 0; 
            
            for(int q=0; q < p; q++) // for the first derivative
            {
                X(v,1+q) = x1(q) - x0(q);
                //Rcpp::Rcout << "datapointused " << v << std::endl; 
                //Rcpp::Rcout << "datapoint = " << i <<std::endl;
                dist1 += pow(X(v,1+q),2)/h;
                //Rcpp::Rcout << "dist ==" << X(v,1+q) << std::endl; 
                //Rcpp::Rcout << "Euclidean distance of " << v << "point is" << dist1 << std::endl;  
            }
            //  Rcpp:: Rcout << dist1 << "Dist" << std::endl;
            W(v,v) = eval_kernel(kcode,dist1)/h; 
            Rcpp::Rcout << W(v,v) << "Weight for point" <<  v <<std::endl;

            // Rcpp::Rcout << "Weight successfully created for "  << std::endl;         
            const MatrixXd& tmp = X.transpose() * W;
            const MatrixXd& res = (tmp * X).inverse() * tmp * Y;
            result(i) = res(0,0);
           
        }
    }
    
    return result; 
}

/***R
x = cbind(c(1,5,8),c(1,5,8))
y = as.matrix(c(3,4,5))
d = 1 
h = 2
xnew = x
kernel = 'epanechnikov' 
wlocpoly(x,y,xnew,d,kernel,h)
*/

