# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reference formulation from Google Research representation_similarity/Demo.ipynb.

https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
Kept in tests to compare the production vectorized implementation independently.
"""
import numpy as np


def center_gram(gram,unbiased=False):
    gram=gram.copy()
    if unbiased:
        n=gram.shape[0]
        np.fill_diagonal(gram,0)
        means=np.sum(gram,0,dtype=np.float64)/(n-2)
        means-=np.sum(means)/(2*(n-1))
        gram-=means[:,None]
        gram-=means[None,:]
        np.fill_diagonal(gram,0)
    else:
        means=np.mean(gram,0,dtype=np.float64)
        means-=np.mean(means)/2
        gram-=means[:,None]
        gram-=means[None,:]
    return gram


def cka(x,y,debiased=False):
    gx=center_gram(x@x.T,unbiased=debiased)
    gy=center_gram(y@y.T,unbiased=debiased)
    return gx.ravel().dot(gy.ravel())/(np.linalg.norm(gx)*np.linalg.norm(gy))
