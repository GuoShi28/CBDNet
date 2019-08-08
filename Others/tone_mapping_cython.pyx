# Power by Zongsheng Yue 2019-06-11 21:23:27

import numpy as np
from math import floor

def CRF_Map_Cython(double[:, :, :] img, double[:] I, double[:] B):
    cdef Py_ssize_t h = img.shape[0]
    cdef Py_ssize_t w = img.shape[1]
    cdef Py_ssize_t c = img.shape[2]
    cdef Py_ssize_t bin = I.shape[0]

    cdef int ii, jj, cc, start_bin, b, index
    out = np.zeros((h,w,c), dtype=np.float64)
    cdef double[:,:,:] out_view = out

    cdef double tiny_bin = 9.7656e-04      # 1/1024 = 9.7656e-04
    cdef double min_tiny_bin = 0.0039
    cdef temp, tempB, comp1, comp2

    for ii in range(h):
        for jj in range(w):
            for cc in range(c):
                temp = img[ii, jj, cc]
                start_bin = 1
                if temp > min_tiny_bin:
                    start_bin = floor(temp / tiny_bin - 1)
                for b in range(start_bin, bin):
                    tempI = I[b]
                    if tempI >= temp:
                        index = b
                        if index > 1:
                            comp1 = tempI - temp
                            comp2 = temp - I[index - 1]
                            if comp2 < comp1:
                                index -= 1
                        out_view[ii, jj, cc] = B[index]
                        break
    return out

def ICRF_Map_Cython(double[:, :, :] img, double[:] invI, double[:] invB):
    cdef Py_ssize_t h = img.shape[0]
    cdef Py_ssize_t w = img.shape[1]
    cdef Py_ssize_t c = img.shape[2]
    cdef Py_ssize_t bin = invI.shape[0]

    cdef int ii, jj, cc, start_bin, b, index
    out = np.zeros((h,w,c), dtype=np.float64)
    cdef double[:,:,:] out_view = out

    cdef double tiny_bin = 9.7656e-04      # 1/1024 = 9.7656e-04
    cdef double min_tiny_bin = 0.0039
    cdef temp, tempB, comp1, comp2

    for ii in range(h):
        for jj in range(w):
            for cc in range(c):
                temp = img[ii, jj, cc]
                start_bin = 1
                if temp > min_tiny_bin:
                    start_bin = floor(temp / tiny_bin - 1)
                for b in range(start_bin, bin):
                    tempB = invB[b]
                    if tempB >= temp:
                        index = b
                        if index > 1:
                            comp1 = tempB - temp
                            comp2 = temp - invB[index - 1]
                            if comp2 < comp1:
                                index -= 1
                        out_view[ii, jj, cc] = invI[index]
                        break
    return out

