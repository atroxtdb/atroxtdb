
#!/usr/bin/env python
import numpy as np
from math import sqrt
import asyncio
from functools import lru_cache
import pickle
import cProfile,pstats


def normal(triangles):
    return np.cross(triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0], axis=1)

def round_off_rating(number):
    return round(number*2)/2

def area(triangles):
    return np.linalg.norm(normal(triangles), axis=1) / 2

def plane(triangles):
    n = normal(triangles)
    u = n / np.linalg.norm(n, axis=1, keepdims=True)
    d = -np.einsum('ij,ij->i', triangles[:, 0], u)
    return np.hstack((u, d[:, None]))

def mincoords(triangles):
    return np.array(np.amin(triangles[:, :, :], axis=1))

def maxcoords(triangles):
    return np.array(np.amax(triangles[:, :, :], axis=1))

def filereader(path):
    """Helps read a file with a delimiter into a list"""
    coordinates1 = []
    Vertex = []
    facet_normals = []
    with open(path, encoding="utf-8") as file1:
        for lines in file1:
            Vertex = lines.split()
            k = []
            norm =[]
            if Vertex[0] =="facet":
                Vertex =Vertex[2:]
                for i in Vertex:
                    j = i.split('e')
                    if len(j) == 2:
                        norm.append(float(j[0]) * 10 ** float(j[1]))
                    else:
                        norm.append(float(j[0]))
                facet_normals.append(np.array(norm))
            if Vertex[0] == 'vertex':
                Vertex = Vertex[1:]
                for i in Vertex:
                    j = i.split('e')
                    if len(j) == 2:
                        k.append(round(float(j[0]) * 10 ** float(j[1]), 2))
                    else:
                        k.append(round(float(j[0]), 2))
                coordinates1.append(np.array(k))
    file1.close()
    return np.array(coordinates1, dtype=float),np.array(facet_normals,dtype=float)

def Validate(Vertex_combined, Points,filename,facetnormals):
    x = []
    counter=[]
    Validpoints=[]
    for i in range(len(Points)):
        for j in range(len(Points[i])):
            if Points[i][j] != None:
                x.append([Vertex_combined[i][0],Vertex_combined[i][1],Vertex_combined[i][2],Points[i][j]])
                counter.append(i)
    x = np.array(x)
    if len(counter)>0:
        areas_a = area(x[:, [0, 1, 3]])
        areas_b = area(x[:, [0, 2, 3]])
        areas_c = area(x[:, [1, 2, 3]])
        areas_d = area(x[:, [1, 2, 0]])
        areas_t = np.round(areas_a + areas_b + areas_c - areas_d,2)
        for i in range(len(areas_t)):
            if areas_t[i] == 0:
                line = np.hstack((x[i][3],facetnormals[counter[i]]))
                Validpoints.append(line)
        with open(filename+'.txt', 'a') as f:
            np.savetxt(f, Validpoints)
    return

def f(planes,blist):
    return -np.round_(((planes[0] * np.array(blist)[:, 0] + planes[1] * np.array(blist)[:, 1] + planes[
        3]) / planes[2]), 2)

def main():
    res = 1
    filename = "part1"
    Vertex_combined,facetnormals = filereader("D:\\PycharmProjects\\Python_trail\\STL\\" + filename + ".stl")
    with open(filename+".txt",'a') as f:
        line=[]
        for i in range(len(Vertex_combined)):
            line.append(np.hstack((Vertex_combined[i],facetnormals[i//3])))
        np.savetxt(f,line)
    Vertex_combined = np.reshape(Vertex_combined, (-1, 3, 3))
    planes = plane(Vertex_combined)
    mins = mincoords(Vertex_combined)
    maxs = maxcoords(Vertex_combined)

    """

    For XY points and finding Z in that grid

    """
    alist = [np.mgrid[round_off_rating(mins[i, 0]):round_off_rating(maxs[i, 0]):res, round_off_rating(mins[i, 1]):
                      round_off_rating(maxs[i, 1]):res] for i in range(len(mins))]
    blist = [x.reshape(2, -1).T for x in alist]
    temp = [-np.round_((planes[i][0] * np.array(blist[i])[:, 0] + planes[i][1] * np.array(blist[i])[:, 1] + planes[i][
        3]) / planes[i][2],2) for i in range(len(planes))]
    #temp = np.ndarray.round(temp,decimals=2)
    XY_Points = [np.hstack((np.array(blist[i]), np.atleast_2d(temp[i]).swapaxes(0, 1))) for i in range(len(temp))]
    #temp = [f(planes[i], blist[i]) for i in range(len(planes))]
    Validate(Vertex_combined, XY_Points, filename,facetnormals)
    XY_Points=[]
    """

    For YZ points and finding X in that grid

    """
    alist = [np.mgrid[round_off_rating(mins[i, 1]):round_off_rating(maxs[i, 1]):res,
             round_off_rating(mins[i, 2]):round_off_rating(maxs[i, 2]):res] for i in range(len(mins))]
    blist = [x.reshape(2, -1).T for x in alist]
    temp = [-np.round_((planes[i][1] * np.array(blist[i])[:, 0] + planes[i][2] * np.array(blist[i])[:, 1] +
                             planes[i][3]) / planes[i][0],2) for i in range(len(planes))]

    YZ_Points = [np.hstack((np.atleast_2d(temp[i]).swapaxes(0, 1), np.array(blist[i]))) for i in range(len(temp))]
    Validate(Vertex_combined, YZ_Points, filename,facetnormals)
    YZ_Points=[]
    """

    For XZ points and finding Y in that grid

    """
    alist = [np.mgrid[round_off_rating(mins[i, 0]):round_off_rating(maxs[i, 0]):res,
             round_off_rating(mins[i, 2]):round_off_rating(maxs[i, 2]):res] for i in range(len(mins))]
    blist = [x.reshape(2, -1).T for x in alist]
    temp = [-np.round_((planes[i][0] * np.array(blist[i])[:, 0] + planes[i][2] * np.array(blist[i])[:, 1] + planes[i][
        3]) / planes[i][1],2) for i in range(len(planes))]
    XZ_Points = [np.hstack((np.atleast_2d(np.array(blist[i])[:, 0]).swapaxes(0, 1),
                    np.atleast_2d(temp[i]).swapaxes(0, 1),
                    np.atleast_2d(np.array(blist[i])[:, 1]).swapaxes(0, 1))) for i in range(len(temp))]
    Validate(Vertex_combined, XZ_Points, filename,facetnormals)
    XZ_Points=[]



if __name__ == "__main__":
    with cProfile.Profile() as pr:
         main()
    stats=pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats()
    np.geterr()
