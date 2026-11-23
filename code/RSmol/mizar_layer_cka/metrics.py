"""Linear CKA with standard centering and unbiased-HSIC U-centering.

Reference: Kornblith et al., ICML 2019 and their official Google Research
representation_similarity/Demo.ipynb. CKA's ratio is not itself unbiased.
"""
import numpy as np


def center_gram(gram,debiased=False):
    gram=np.asarray(gram,dtype=np.float64).copy()
    if gram.ndim!=2 or gram.shape[0]!=gram.shape[1] or not np.isfinite(gram).all() or not np.allclose(gram,gram.T):
        raise ValueError('expected a finite symmetric Gram matrix')
    n=len(gram)
    if debiased:
        if n<4: raise ValueError('unbiased HSIC requires at least four examples')
        np.fill_diagonal(gram,0)
        means=gram.sum(axis=0)/(n-2)
        means-=means.sum()/(2*(n-1))
        gram-=means[:,None]+means[None,:]
        np.fill_diagonal(gram,0)
    else:
        means=gram.mean(axis=0)
        gram-=means[:,None]+means[None,:]-means.mean()
    return gram


def normalized_grams(features,debiased=False):
    features=np.asarray(features,dtype=np.float64)
    if features.ndim!=3 or features.shape[0]!=30 or not np.isfinite(features).all():
        raise ValueError('expected finite [30, examples, features]')
    features=features-features.mean(axis=1,keepdims=True)
    grams=np.stack([center_gram(x@x.T,debiased) for x in features])
    norms=np.linalg.norm(grams.reshape(30,-1),axis=1)
    if not np.isfinite(norms).all() or np.any(norms<=np.finfo(np.float64).eps):
        raise ValueError('degenerate representation: CKA denominator is zero')
    return grams/norms[:,None,None]


def matrix(features,debiased=False):
    grams=normalized_grams(features,debiased).reshape(30,-1)
    result=grams@grams.T
    if not np.allclose(result,result.T,atol=1e-12) or not np.allclose(np.diag(result),1,atol=1e-12):
        raise ValueError('CKA symmetry/self-similarity invariant failed')
    return result


def permutation_null(features,debiased=False,repeats=100,seed=1234):
    if type(repeats) is not int or repeats<1: raise ValueError('positive permutation count required')
    grams=normalized_grams(features,debiased); flat=grams.reshape(30,-1); rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(repeats):
        order=rng.permutation(grams.shape[1])
        permuted=grams[:,order][:,:,order].reshape(30,-1)
        value=flat@permuted.T
        samples.append((value+value.T)/2)
    values=np.stack(samples)
    return values.mean(0),values.std(0)
