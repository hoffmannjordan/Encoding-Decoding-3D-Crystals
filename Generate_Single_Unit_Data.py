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
    Input: Crystal and offsets a,b,c for sampling [not used]
    Outut: Species and density matrix
    '''
    abc = np.array(crystal.lattice.abc)
    Zs = np.zeros(len(crystal))
    XYZs = np.zeros((len(crystal),3))
    for i in range(len(crystal)):
        Element_ID = Element(crystal[i].species_string).Z
        x       = crystal[i].x
        y       = crystal[i].y
        z       = crystal[i].z
        XYZ     = np.array([x,y,z])
        Zs[i]   = Element_ID
        XYZs[i] = XYZ
    mean = np.mean(XYZs,axis=0)
    shift = np.array([5,5,5]) - mean
    XYZs_shifted = np.copy(XYZs)
    for i in range(len(XYZs_shifted)):
        XYZs_shifted[i] += shift
    max_v = np.amax(XYZs_shifted)
    min_v = np.amin(XYZs_shifted)
    if max_v < 10.0 and min_v > 0.0:
        mat = np.zeros((30,30,30))
        mat2 = np.zeros((30,30,30))
        bins = np.linspace(0,10,30)
        sigma = 1.0
        for i in range(30):
            for j in range(30):
                for k in range(30):
                    coordinate = np.array([bins[i],bins[j],bins[k]])
                    for x in range(len(XYZs_shifted)):
                        dist = np.linalg.norm(coordinate - XYZs_shifted[x])
                        mat[i,j,k] += 1.0/((2.0*np.pi)**1.5)*Zs[x]*(1.0/sigma**3)*np.exp(-dist**2/(2*sigma**2))
                        if dist < 0.667:
                            if mat2[i,j,k] > 0.0:
                                print("Overwrite?")
                                if np.random.rand() > 0.5:
                                    mat2[i,j,k] = int(Zs[x])
                            else:
                                mat2[i,j,k] = int(Zs[x])
        return mat,mat2,Zs,XYZs_shifted
    else:
        return 0,0,0,0



if __name__=='__main__':
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        print('I am size ',rank)
        directory = '/scratch3/jordan/Mila/Crystal-Project/dataset/code/data/'
        upper = 46381
        delta = int(upper/size)
        saveQ = True
        target = read_in_properties(directory+'./id_prop.csv')
        print(target[0:7])
        files = glob.glob(directory+'*')
        np.random.seed(1)
        file_list = np.random.permutation(np.arange(0,upper+1))
        start = delta*rank
        stop  = start + delta
        if rank == (size-1):
            # Ensure the last one is done
            stop = 46381

        to_save = []
        for i in range(start,stop):
                if rank == 0:
                        print("I am on "+str(i)+" and I am going to "+str(stop))
                crystal = get_structure('/scratch3/jordan/Mila/Crystal-Project/dataset/code/data/'+str(file_list[i])+'.cif')
                
                electron_density,species_mat,Zs,XYZs_shifted = generate(crystal,0,0,0)
                if len(np.shape(electron_density)) != 0:
                    # Ignore things that were [0,0,0,0]
                    to_save.append([file_list[i],electron_density,species_mat,target[file_list[i]],Zs,XYZs_shifted])


        if saveQ == True:
                with open('/DIRECTORY/Unit_'+str(rank)+'.pickle', 'wb') as f:
                        print('SAVING ',rank)
                        pickle.dump(to_save, f)
