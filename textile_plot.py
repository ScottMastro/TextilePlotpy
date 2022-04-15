#------------------------------------------------------------------
# Original source:
#    Natsuhiko Kumasakaa, Ritei Shibata
#    High-dimensional data visualisation: The textile plot
#    https://doi.org/10.1016/j.csda.2007.11.016
#    Computational Statistics and Data Analysis
#
# Python implementation:
#    Scott Mastromatteo
#------------------------------------------------------------------

import numpy as np
import pandas as pd
import scipy

def _flatten_list(lst):
    ''' Internal function '''

    n=lst[0].shape[0]
    cols = []
    for x in lst:
        if x.ndim == 1: 
            cols.append(x.reshape(n))
        else:
            for i in range(x.shape[1]):
                cols.append(x[:,i].reshape(n))
    return np.stack(cols, axis=1)

def _textile_transform_with_missing(Xj, wj, qj, eigen_choice=1, zscale=0):
    ''' Internal function '''

    n=len(Xj[0]) ; p=len(Xj)

    W = np.stack(wj, axis=1)
    d = np.sum(W, axis=1)
    if sum(d==0) > 0:
        raise ValueError("One or more records is completely missing data.")
    
    Wpad = np.stack([wj[i] for i in qj], axis=1)
    X=_flatten_list(Xj)
    
    # A11 ------------------------
    d11 = np.ones(n).transpose().dot(W)
    A11 = -W.transpose().dot(W/d.reshape(n, 1)) + np.diag(d11)
    
    # A12 ------------------------
    A12= Wpad.transpose().dot(Wpad*X/d.reshape(n, 1))
    d12 = np.diag(Wpad.transpose().dot(X))
    d12 = np.diag(d12)
    
    to_remove = [] ; s=set()
    for j,q_ in enumerate(qj):
        if q_ in s:
            to_remove.append(j)
            d12[q_] += d12[j]
        s.add(q_)
    
    A12 = np.delete(A12, to_remove, axis=0) - np.delete(d12, to_remove, axis=0)
    
    # A22 ------------------------
    d22_full = X.transpose().dot(Wpad)*Wpad.transpose().dot(X)/np.ones(n).reshape(1,n).dot(Wpad)
    d22 = np.zeros((d22_full.shape))
    skip = 0       
    for k,j in enumerate(qj):
        j=j+skip
        if j != k: skip+=1
        while j <= k:
            d22[j,k] = d22_full[j,k] ; d22[k,j] = d22_full[k,j]
            j+=1
    
    A22 = -(Wpad*X).transpose().dot(Wpad*X/d.reshape(n, 1)) + d22
    
    # B ------------------------
    B_full = np.dot(X.transpose(), Wpad*X) - d22_full
    B = np.zeros((B_full.shape))
    
    skip = 0       
    for k,j in enumerate(qj):
        j=j+skip
        if j != k: skip+=1
        while j <= k:
            B[j,k] = B_full[j,k] ; B[k,j] = B_full[k,j]
            j+=1
    
    # A ------------------------
    A11inv = np.linalg.pinv(A11, rcond=1e-10)
    
    A=(np.dot(A12.transpose(), A11inv).dot(A12) - A22)
    
    eigenvectors, eigenvalues, _ = scipy.linalg.lapack.dsygv(A, B, uplo="U")
    # in ascending order, last col has biggest eigenvalue
    beta = eigenvectors[:,-eigen_choice]

    # arbitray constant
    if zscale == 0:
        alpha=np.dot(A11inv, A12).dot(beta)
    else:
        z=np.ones(p) * zscale 
        alpha=np.dot(A11inv, A12).dot(beta) + (np.ones(p) - np.dot(A11inv, A11)).dot(z)
    
    return(A, B, alpha, beta)

def _textile_transform_no_missing(Xj, qj, eigen_choice=1, a0=0):
    ''' Internal function '''

    p=len(Xj) ; n=len(Xj[0])
    
    X = _flatten_list(Xj)
    
    XTX = X.transpose().dot(X)
    ones = np.ones(n).reshape(n, 1)
    XT11TX = X.transpose().dot(ones).dot(ones.transpose()).dot(X)
    A = (XTX - XT11TX/n)/p
    
    B_full = XTX - XT11TX/n
    B = np.zeros((B_full.shape))
    
    skip = 0
    for k,j in enumerate(qj):
        j=j+skip
        if j != k: skip+=1
        while j <= k:
            B[j,k] = B_full[j,k] ; B[k,j] = B_full[k,j]
            j+=1
        
    eigenvectors, eigenvalues, _ = scipy.linalg.lapack.dsygv(A, B, uplo="U")    
    # in ascending order, last col has biggest eigenvalue
    beta = eigenvectors[:,-eigen_choice]
    
    alpha = np.zeros(len(Xj))
    for j,xj in enumerate(Xj):
        k=[i for i,q in enumerate(qj) if q == j]
        alpha[j]=a0-(np.ones(n).reshape(1,n).dot(xj)/n).dot(beta[k])

    return(A, B, alpha, beta)

def textile_transform(X, is_categorical=None, eigen_choice=1, category_options=dict(), get_location_scale=False):
    '''  
    Textile plot transformation

    X : [p x n numpy matrix] Use None for missing data. Note: uses faster method if no data is missing
    
    is_categorical : [list] of length p. True if x[i] is categorical variable, False if quantitative.
                    Use is_categorical=None if no variables are categorical.
                    Use is_categorical="all" if ALL variables are categorical.
    
    category_options: [dict[int]:list] Key=variable index, value=all categorical classes available to that variable.
                      If category options are not provided, the categories are inferred and placed in sorted order.
    
    eigen_choice: [int] eigenvector to use for transformation. Default=1 (largest eigenvalue). Must be <= p
    
    get_location_scale: [boolean] If true, returns the location and scale vectors as a tuple along with the transformed matrix.
                        Intended for internal use for genetic application to calculate height for each allele.
    Returns p x n numpy matrix transformed data.
    '''
    
    Xj=[]; wj=[] ; qj=[] ; j=0
    
    x = [col.reshape(col.shape[0]) for col in np.hsplit(X, X.data.shape[1])]
    if is_categorical is None:
        is_categorical = [False for x_ in x]
    
    if is_categorical == "all":
        is_categorical = [True for x_ in x]

    p=len(x) ; n=len(x[0])

    def is_none(item):
        return item is None or np.isnan(item) or str(item) == "NA"
    def most_frequent(itr):
        lst = [x for x in itr if not is_none(x)]
        return max(set(lst), key = lst.count)
    
    for xi, cat in zip(x, is_categorical):
        wi = np.array([0 if is_none(x_) else 1 for x_ in xi])
        wj.append(wi)
        if cat:
            common = most_frequent(xi)
            xi=[common if w_ == 0 else x_ for w_,x_ in zip(wi,xi)]
            
            if j in category_options:
                cats = category_options[j]
            else:
                cats = sorted(list(dict.fromkeys(xi)))
            
            cat_dict = {cat_: l for l,cat_ in enumerate(cats)}
            Z = np.zeros((len(xi), len(cats)))
            for k,x_ in enumerate(xi):
                Z[k][cat_dict[x_]] = 1
                
            C=pd.get_dummies(cats, drop_first=True).to_numpy()
            X_=Z.dot(C)
            Xj.append(X_)
            for col in range(X_.shape[1]):
                qj.append(j)
        else:
            xi=[0 if w_ == 0 else x_ for w_,x_ in zip(wi,xi)]
            Xj.append(np.array(xi).reshape(len(x[0]),1))
            qj.append(j)
        j+=1
        
    if sum([sum(1-w_) for w_ in wj]) > 0:
        A, B, alpha, beta = \
            _textile_transform_with_missing(Xj, wj, qj, eigen_choice=eigen_choice)
    else:
        A, B, alpha, beta = \
            _textile_transform_no_missing(Xj, qj, eigen_choice=1)
    
    alpha = alpha * (n*p)
    beta = beta * (n*p)

    yj=[]
    for j,xj in enumerate(Xj):
        k=[i for i,q in enumerate(qj) if q == j]
        yj.append(xj.dot(beta[k]) + alpha[j])
    
    Y=np.stack(yj, axis=1)
    #y0=(W*Y).sum(axis=1)/Y.shape[1]
    #yj = [y0] + yj
    #Y=np.stack(yj, axis=1)

    #print("A:\n", A.round(4))
    #print("B:\n", B.round(4))
    #print("alpha:\n", alpha.round(4))
    #print("beta:\n", beta.round(4)) 
    #print("Y:\n", Y.round(4))
    if get_location_scale:
        return(Y, alpha, beta)
    
    return Y

def get_example_data():
    ''' Example dataset included in original Textile Plot package '''

    x1 = [-0.001271673,-0.008055735, 0.997513967, None, -0.466050483,
    -2.831182321,-0.295235185, 0.321523554, 0.452224977, None]
    x2 = [1.094754192, 0.516446048,-0.038293932,-2.447391202,-0.735928229,
    -0.004057757, None, -0.657263241, 0.055907470, None]
    x3 = [0,None,1,0,0,2,1,2,0,1]
    x=[np.array(x1),np.array(x2),np.array(x3)]
    x=np.stack(x).transpose()
    is_categorical=[False,False,True]
    return (["x1","x2","x3"], x, is_categorical)

def get_iris_data():
    ''' Iris dataset '''

    x1=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    x2=[5.1, 4.9, 4.7, 4.6, 5. , 5.4, 4.6, 5. , 4.4, 4.9, 5.4, 4.8, 4.8,
        4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5. ,
        5. , 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5. , 5.5, 4.9, 4.4,
        5.1, 5. , 4.5, 4.4, 5. , 5.1, 4.8, 5.1, 4.6, 5.3, 5. , 7. , 6.4,
        6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5. , 5.9, 6. , 6.1, 5.6,
        6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7,
        6. , 5.7, 5.5, 5.5, 5.8, 6. , 5.4, 6. , 6.7, 6.3, 5.6, 5.5, 5.5,
        6.1, 5.8, 5. , 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3,
        6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5,
        7.7, 7.7, 6. , 6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2,
        7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6. , 6.9, 6.7, 6.9, 5.8,
        6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9]
    x3=[3.5, 3. , 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3. ,
        3. , 4. , 4.4, 3.9, 3.5, 3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3. ,
        3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 4.1, 4.2, 3.1, 3.2, 3.5, 3.6, 3. ,
        3.4, 3.5, 2.3, 3.2, 3.5, 3.8, 3. , 3.8, 3.2, 3.7, 3.3, 3.2, 3.2,
        3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2. , 3. , 2.2, 2.9, 2.9,
        3.1, 3. , 2.7, 2.2, 2.5, 3.2, 2.8, 2.5, 2.8, 2.9, 3. , 2.8, 3. ,
        2.9, 2.6, 2.4, 2.4, 2.7, 2.7, 3. , 3.4, 3.1, 2.3, 3. , 2.5, 2.6,
        3. , 2.6, 2.3, 2.7, 3. , 2.9, 2.9, 2.5, 2.8, 3.3, 2.7, 3. , 2.9,
        3. , 3. , 2.5, 2.9, 2.5, 3.6, 3.2, 2.7, 3. , 2.5, 2.8, 3.2, 3. ,
        3.8, 2.6, 2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2, 2.8, 3. , 2.8, 3. ,
        2.8, 3.8, 2.8, 2.8, 2.6, 3. , 3.4, 3.1, 3. , 3.1, 3.1, 3.1, 2.7,
        3.2, 3.3, 3. , 2.5, 3. , 3.4, 3.]
    x4=[1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4,
        1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5, 1.7, 1.5, 1. , 1.7, 1.9, 1.6,
        1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.4, 1.3,
        1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.7, 4.5,
        4.9, 4. , 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4. , 4.7, 3.6,
        4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4. , 4.9, 4.7, 4.3, 4.4, 4.8, 5. ,
        4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4. , 4.4,
        4.6, 4. , 3.3, 4.2, 4.2, 4.2, 4.3, 3. , 4.1, 6. , 5.1, 5.9, 5.6,
        5.8, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 5.5, 5. , 5.1, 5.3, 5.5,
        6.7, 6.9, 5. , 5.7, 4.9, 6.7, 4.9, 5.7, 6. , 4.8, 4.9, 5.6, 5.8,
        6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8, 5.4, 5.6, 5.1, 5.1,
        5.9, 5.7, 5.2, 5. , 5.2, 5.4, 5.1]
    x5=[0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1,
        0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2,
        0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2,
        0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 1.4, 1.5,
        1.5, 1.3, 1.5, 1.3, 1.6, 1. , 1.3, 1.4, 1. , 1.5, 1. , 1.4, 1.3,
        1.4, 1.5, 1. , 1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7,
        1.5, 1. , 1.1, 1. , 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3, 1.2,
        1.4, 1.2, 1. , 1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.5, 1.9, 2.1, 1.8,
        2.2, 2.1, 1.7, 1.8, 1.8, 2.5, 2. , 1.9, 2.1, 2. , 2.4, 2.3, 1.8,
        2.2, 2.3, 1.5, 2.3, 2. , 2. , 1.8, 2.1, 1.8, 1.8, 1.8, 2.1, 1.6,
        1.9, 2. , 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9,
        2.3, 2.5, 2.3, 1.9, 2. , 2.3, 1.8]

    labels=["species", "sepal length (cm)", "sepal width (cm)", 
            "petal length (cm)", "petal width (cm)"]
    x = [x1,x2,x3,x4,x5]
    x = np.stack(x).transpose()
    is_categorical=[True,False,False,False,False]
    return (labels, x, is_categorical)

def get_1kgenome_data():
    ''' Example genetic dataset using 1000Genomes data chr1:1471000-1481000. Requires PyVCF. '''
    import vcf

    VCF="data/1kgenomes_chr1_1471000_1481000.vcf"
    vcf_reader = vcf.Reader(filename=VCF)
    vcf_records = [record for record in vcf_reader]
    return vcf_records

def genotype_textile(vcf_records, haplotype=False, set_missing_to_ref=False):
    '''  
    Textile plot transformation for genetic data

    vcf_records : [list] A list of p VCF records (generated from PyVCF) to transform.
    haplotype : [boolean] Whether to transform genotype (GT) or phased haplotypes (uses PS if available) from records
                Defaults to genotype.
    set_missing_to_ref: [boolean] Set all GT values with missing data to 0/0.
    Returns a list of length p corresponding to y-position of each variant in textile plot.
    '''
    
    gt_matrix=[]
    for record in vcf_records:
        if len(record.alleles) != 2:
            raise ValueError("Only bialleleic variants are allowed when haplotype=False : " + str(record))        
        samples=[]
        for sample in record.samples:
            gt=sample.gt_alleles
            if None in gt: 
                if set_missing_to_ref:
                    samples.append(0)
                else:
                    samples.append(None) 
                continue
            samples.append(int(gt[0]) + int(gt[1]))

        if not set_missing_to_ref and len([x for x in samples if x is not None]) < 1:
            raise ValueError("All data for variant is missing: " + str(record))        
        gt_matrix.append(samples)
        
    M = np.stack(gt_matrix, axis=1)
    options = dict()
    for j in range(M.shape[1]):
        options[j] = [0,1,2]
    Y, alpha, beta = textile_transform(M, is_categorical="all", category_options=options, get_location_scale=True)
    
    y=[]
    k=0
    for j in range(M.shape[1]):
        y.append( {options[j][0] : alpha[j]} )
        for cat in options[j][1:]:
            y[j][cat] = alpha[j] + beta[k]
            k+=1
            
                    

    for record in vcf_records:
        for sample in record.samples:
            gt=sample.gt_alleles




