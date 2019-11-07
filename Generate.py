# File to generate the periodic lattice input data
# Code written by Jordan
from pymatgen.core.structure import Structure
import numpy as np
from pymatgen import Molecule
from pymatgen.analysis.local_env import MinimumVIRENN, VoronoiNN
from pymatgen.core.sites import *
import glob
from mpi4py import MPI
import gzip
import pickle


def get_structure(file):
    '''
    Input:  file name to get the strucutre of
    Return: The crystal structure
    '''
    crystal = Structure.from_file(file)
    return crystal

def read_in_properties(dataset):
    '''
    Input: Properties file
    Return: column of interest.
    '''
    data = np.genfromtxt(dataset,delimiter=',')
    return data[:,1]

def generate(crystal,a,b,c):
    '''
    Input: Crystal and offsets a,b,c for sampling
    Outut: Species and density matrix
    '''
    Atomic_Numbers = [Element(crystal[i].species_string).Z for i in range(len(crystal))]
    lattice = crystal.lattice
    mat = np.zeros((30,30,30))
    mat2 = np.zeros((30,30,30))
    sigma = 1.0
    bins = np.linspace(0,10,30)
    for i in range(30):
        for j in range(30):
            for k in range(30):
                s = PeriodicSite(1,[bins[i]+a,bins[j]+b,bins[k]+c],lattice,coords_are_cartesian=True)
                for x in range(len(crystal)):
                    (dist,im) = crystal[x].distance_and_image(s)
                    mat[i,j,k] += 1.0/((2.0*np.pi)**1.5)*Atomic_Numbers[x]*(1.0/sigma**3)*np.exp(-dist**2/(2*sigma**2))
                    if dist < 0.65:
                        if mat2[i,j,k] > 0.0:
                            if np.random.rand() > 0.5:
                                mat2[i,j,k] = int(Atomic_Numbers[x])
                        else:
                            mat2[i,j,k] = int(Atomic_Numbers[x])
    return mat,mat2

if __name__=='__main__':
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        directory = '/DIRECTORY/'
        upper = 46381
        delta = int(46381/size)
        saveQ = True

        target = read_in_properties(directory+'./id_prop.csv')
        print(target[0:7])

        files = glob.glob(directory+'*')
        np.random.seed(1)
        file_list = np.random.permutation(np.arange(0,upper+1))
        start = delta*rank
        stop  = start + delta
        if rank == (size-1):
                stop = 46381

        to_save = []
        for i in range(start,stop):
                if rank == 0:
                        print("I am on "+str(i)+" and I am going to "+str(stop))
                crystal = get_structure('/DIRECTORY/'+str(file_list[i])+'.cif')
                abc = np.array(crystal.lattice.abc)
                if min(abc) < 10:
                        electron_density,electron_density2 = generate(crystal,0,0,0)
                        to_save.append([file_list[i],electron_density,electron_density2,target[file_list[i]]])
                        electron_density,electron_density2 = generate(crystal,10*np.random.rand(),10*np.random.rand(),10*np.random.rand())
                        to_save.append([file_list[i],electron_density,electron_density2,target[file_list[i]]])
                        electron_density,electron_density2 = generate(crystal,10*np.random.rand(),10*np.random.rand(),10*np.random.rand())
                        to_save.append([file_list[i],electron_density,electron_density2,target[file_list[i]]])


        if saveQ == True:
                with open('/SAVE/DIRECTORY/S_3x_'+str(rank)+'.pickle', 'wb') as f:
                        print('SAVING')
                        pickle.dump(to_save, f)
                        print('SAVED OK')
