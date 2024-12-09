import numpy as np
import pandas as pd

def generate_wouters_input_data(N, Z, binding_energy, NMN = [8,20,28,50,82,126], PNM = [8,20,28,50,82,126]):

  """
    Generate a complete set of nuclear data from the nuclear masses. 
    Requires the file 'masses_ZN_shuf.dat' which contains in random order
        Z  N  BE(Z,N)
    This routine will generate a numpy array with 11 columns:
    
      0  1  2   3    4      5        6          7      8      9     10
      N  Z  PiN PiZ  A^2/3  Coul  (N-Z)**2/A  A^(-1/2)distN distZ  BE(Z,N)
    
    where   
      Coul = Z * (Z-1)/(A**(1./3.))
      distN= distance to nearest neutron magic number, selected from input
             array NMN
      distZ= same but for proton magic numbers from PNM
    
    --------------------------------------------------------------------------

    Input : 
      NMN, PNM: list of neutron and proton magic numbers    
    Output: 
      complete_dat : numpy array as described above
      N_input      : column dimension of complete_dat
    
  """

  # Size of the input, i.e. the number of parameters passed in to the MLNN for 
  # any given input 
  N_input = 10
  A = N + Z
  
  # Creating a complete data set from the (N,Z,BE)-data the table
  # Note that this has N_input + 1 columns: we store the binding energy here too
  complete_dat = np.zeros((len(N),N_input+1))

  for i in range(len(N)):
    # - - - - - - - - - - - - - - - - - - 
    # Basic information
    complete_dat[i,0] = N[i]         # Neutron number N
    complete_dat[i,1] = Z[i]       # Proton number Z
    complete_dat[i,2] = (-1)**(N[i]) # Number parity of neutrons
    complete_dat[i,3] = (-1)**(Z[i]) #                  protons

    #- - - - - - - - - - - - - - - - - - 
    # Liquid drop parameters
    complete_dat[i,4] = (A[i])**(2./3.)            # A^2/3
    complete_dat[i,5] = Z[i] * (Z[i]-1)/(A[i]**(1./3.))  # Coulomb term
    complete_dat[i,6] =(N[i]-Z[i])**2/A[i]               # Asymmetry
    complete_dat[i,7] = A[i]**(-1./2.)             # Pairing
    
    # - - - - - - - - - - - - - - - -  
    # Distance to the next magic number for both protons and neutrons
    dist_N = 100000
    dist_Z = 100000
    for k in NMN:
      dist_Nb = abs(N[i] - k)
      if(dist_Nb < dist_N):
        dist_N = dist_Nb

    for k in PNM:   
      dist_Zb = abs(Z[i] - k)
      if(dist_Zb < dist_Z):
        dist_Z = dist_Zb
   
    complete_dat[i,8] = dist_N
    complete_dat[i,9] = dist_Z
    complete_dat[i,10] = binding_energy[i]
    # - - - - - - - - - - - - - - - -
    
  return complete_dat, N_input

def modified_wouter(Z, N, M, param, NMN = [8,20,28,50,82,126], PNM = [8,20,28,50,82,126]):
    """
    A modified version of wouter input data. This consists of:
    Z = Proton Number
    N = Neutron number
    M = mass (MeV)
    param = a list of parameters that is used, this follows the order 
    t_0;t_1;t_2;t_3;t_4;t_5;x_0;x_1;x_2;x_3;x_4;x_5;alpha;beta;gamma;W_0;f_n+;f_n-;f_p+;f_p-;epsilon_A
    """
    
    wouter_dat, N_input = generate_wouters_input_data(N, Z, M, NMN = NMN, PNM = PNM)
    N_input = 31

    complete_dat = np.concatenate((wouter_dat[:,:-1], param, wouter_dat[:,-1][:,np.newaxis]), axis=1)

    return complete_dat, N_input
