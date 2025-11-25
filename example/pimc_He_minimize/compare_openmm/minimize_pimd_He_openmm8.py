#!/usr/bin/env python3
'''
Description:
------------  

Use: 

Requirements: openmm >=7.4, scipy >=  

'''

DEBUG=True


# %% Utilities
# %%
def is_unwrapped(coords, box):
    '''
    coords: of a ring polymers system, shape=(npoly,nbeads,ndim)
    box: [Lx, Ly, Lz]
    
    A ring polymer is _unwrapped_ if its consecutive beads are at their minimal image position.
    Returns True if each ring polymers is unwrapped.
    '''
    for poly in coords: 
        for ib in range(0, len(poly)): 
            dr = np.abs( poly[ib] - poly[ib-1] )  # p[-1]=p[nb-1]
            #print(dr)    
            if dr[0] >= box[0]/2. or dr[1]>= box[1]/2. or dr[2]>= box[2]/2.:
                return False
    return True 




# %% Beginning of scripts
import json
import sys,os


if len(sys.argv) < 6:
  raise Exception("Syntax: ", sys.argv[0], "<topology.pdb>  <input parameters.txt>  <initial configuration.pos>  <output(minimzed) .pos>  <output energy.dat>")

''' 
Help message 
'''

input_pdb = sys.argv[1]
input_pars = sys.argv[2]
input_pos = sys.argv[3]

out_pos = sys.argv[4]
out_energy = sys.argv[5]



# %%
from datetime import datetime
from openmm.app import *
from openmm import *
from openmm.unit import *
from scipy import optimize
import numpy as np
import pandas as pd

print("------------------------------------------")
print("Potential Energy minimization of RPMD Helium \n")
print("\nOpenMM version:", version.full_version)
print( datetime.now() )
print("------------------------------------------")


# %%
print("\nReading input parameters...")

with open(input_pars,'r') as fp:
    for line in fp.readlines():
        words= None

        if 'Temp_Bath' in line :  # temperature bath
            words = line.split()
            temp = float(words[1]) 

        if 'Box_length' in line :  # box length [anstrome]
            words = line.split()
            boxLen = float(words[1]) 

        if 'Nbeads' in line :  # polymer length / num copies
            words = line.split()
            nbeads = int(words[1])

        if 'Nparticle' in line :  # number of particles(polymers)
            words = line.split()
            Npatc = int(words[1]) 

        if 'Platform' in line :  # name of the platform: {CPU, CUDA, OpenCL, Reference}
            words = line.split()
            mplatform = words[1] 


""" always using real units in this program """


# %% default paras

E0 = 0. ; Emin = 0.

ndim = 3
mass = 6.6464731e-27   # mass: He atom [kg] 
boltz = 1.38064852e-23 # boltzmann constant [J/K]
mol = 6.02214086e23    # Avogadro constant
hbar = 1.054571817e-34 #   [J*second]  


## processing paras

beta = 1.0 / (temp*boltz)
spring_constant = (mass* nbeads)/(beta*beta * hbar*hbar)* mol *1e-3 * 1e-9 * 1e-9 * nbeads      # kJ/mol * nm-2  #  

## Reading box L size
pos_head = np.loadtxt(input_pos, float, max_rows=1)  # TODO:  from pos class
#print(data)

L = pos_head[1]*angstrom   
if pos_head[1] != boxLen:
    raise ValueError('The box length read from input.pos is different from input file! \n')


# %%
print("\nCreating physical system...")

# create a system model from pdb
pdb = PDBFile(input_pdb)

# Force Field. Defining the atoms of the system.
forcefield = ForceField( os.path.dirname(__file__) +'/o.xml') 

system = forcefield.createSystem(pdb.topology)

nparticles = system.getNumParticles() 
if not nparticles == Npatc:
    print("Warning!: inconsistent number of particles: input pdb and input file ")
print("  number of particles read =", nparticles)


# %%
# Set the periodic box vectors
Lx = [L,    0,    0]
Ly = [0,    L,    0]
Lz = [0,    0,    L]
system.setDefaultPeriodicBoxVectors(a = Lx, b = Ly, c = Lz)

box = [L.value_in_unit(nanometer)]*3

# %%
print("\nCreating interactions...")

# Add interactions using a CustomNonbondedForce
force = CustomNonbondedForce( 
                " ( A_star * exp(- alpha_star * (r/rm) + beta_star * (r/rm)^2 )  - (c6 /(r/rm)^6 + c8 /(r/rm)^8 + c10 /(r/rm)^10 ) * Fx ) * epsilon ;" # energy expression without rCut. Aziz 1995
                " Fx = step(D- r/rm )* exp(- (D/(r/rm) -1.)*( D/(r/rm) -1.) ) + step(r/rm -D)*1. ;"
                )
force.addGlobalParameter( "A_star", 1.86924404e5 )
force.addGlobalParameter( "alpha_star", 10.5717543 )
force.addGlobalParameter( "beta_star", -2.07758779 )
force.addGlobalParameter( "c6", 1.35186623 )
force.addGlobalParameter( "c8", 0.41495143 )
force.addGlobalParameter( "c8", 0.41495143 )
force.addGlobalParameter( "c10", 0.17151143 )
force.addGlobalParameter( "D", 1.438)
force.addGlobalParameter( "epsilon", 0.0910933 )   # 0.0910933 kJ/mol  = 10.956 Kelvin * kB
force.addGlobalParameter(  "rm", 0.29683 )   #  = 2.9683 angstrom

force.setNonbondedMethod(2)  #  ??  equals to  NonbondedForce.CutoffPeriodic

force.setCutoffDistance(0.7 *nanometer) # force field cutoff distance.

force.setUseSwitchingFunction(True) 
force.setSwitchingDistance(0.63 *nanometer) # turn on switch at 0.9 rcut

# force.setUseLongRangeCorrection(True) 

for index in range(nparticles): 
    force.addParticle()     # here can add particle specific parameters. ( e.g. charge )  particle index same as in the System object. 

force_index = system.addForce(force) # system takes ownership of the NonbondedForce object

# %%
print("\nSetting up simulation ...")

integrator = RPMDIntegrator(nbeads, temp*kelvin, 0.5/picosecond, 0.0005*picoseconds)

platform = Platform.getPlatformByName(mplatform)      
#properties = {'CudaPrecision': 'double'}  ## TODO: need double precision.
# simulation = Simulation(pdb.topology, system, integrator) #,properties)
simulation = Simulation(pdb.topology, system, integrator, platform) 


print( "    Using platform: " ,simulation.context.getPlatform().getName() )



##########################################

# %% Helper to calculate the total spring energy of the system   # TODO: JIT !
def calculate_Esp(pos):
    '''
    pos: the full system configuration, pos[j,i,:] is the j-th replica, i-th particle. 
    using global spring constant.
    '''
    energy = 0.0
    for i in range(Npatc):
        # nb >= 1
        for j in range(nbeads):  
            dx,dy,dz = pos[j,i,:] - pos[j-1,i,:]   # r_0 - r_-1 (r_N-1) ; r_1 - r_0; .. ; r_N-1 - r_N-2  
            drsq = dx*dx + dy*dy + dz*dz
            energy = energy + 0.5*spring_constant*drsq
        # for nb=2, Esp = |r0-r1|^2 + |r1-r0|^2
        # for nb=1, Esp = |r0-r0|^2 = 0
    return energy 

# %% Helper to calculate the total spring forces of the system
def calculate_Fsp(pos):
    '''
    pos: the full system configuration, pos[j,i,:] is the j-th replica, i-th particle.  
    using global spring constant.
    '''
    forces = np.zeros( (nbeads,Npatc, ndim) )
    for i in range(Npatc):
        for j in range(nbeads):
            dx,dy,dz = - pos[j,i,:] + pos[j-1,i,:] + pos[j+ 1 - nbeads,i,:] - pos[j,i,:] 
            
            forces[j,i,:] = spring_constant* dx , spring_constant*dy , spring_constant*dz 
            
    return forces
####################################################################################################


# %% 
print("    Setting up initial state...")

coords_0_pos = np.loadtxt(input_pos, float, skiprows=1)  # POS class reader    unit = angstrome   
xyz_00 = coords_0_pos * 0.1           # from angstrom to nanometers.

xyz_0 = xyz_00.reshape(-1,) # for scipy initial x0
# print(xyz_0.shape)

# change to openMM format
coords = np.zeros([nbeads,Npatc,3],float)
s = 0
for i in range(Npatc):
    for j in range(nbeads):
        coords[j,i,:] = xyz_00[s,:]        
        s = s + 1

# check if fully unwrapped 
if not is_unwrapped(np.transpose(coords,(1,0,2)), box ):
    raise ValueError("Error: Input coordinates is not fully unwrapped!")


for k in range(integrator.getNumCopies()):
    simulation.integrator.setPositions(k,coords[k,:,:])  


# Calculate the initial energy
Eint0 = 0.0    
for i in range(simulation.integrator.getNumCopies()):
    state = simulation.integrator.getState(i,getEnergy=True)
    Eint0 = Eint0 + state.getPotentialEnergy().value_in_unit(kilojoule/mole)
Esp_0 = calculate_Esp(coords)
E0 = Eint0 + Esp_0    # NOTE: 
print('E0=' ,E0)
print('Eint_0=' ,Eint0)
print('Esp_0=' ,Esp_0)

step = 0
debug_info = {}

# %% target function to be minimized.
def target_function(xyz):  # TODO input pos directly.
    '''
    Input:  a system configuration (positions, shape = (Npatc * nbeads * 3,) )
    Output: objective function (Etot), and its gradient (forces, shape = (Npatc * nbeads * 3,) )
    '''
    # Converting the coords from (Npatc*nbeads,3) to (nbeads,Npatc,3) to be consistent with OpenMM
    xyz = xyz.reshape(-1,3)
    coords = np.zeros([nbeads,Npatc,ndim],float)
    s = 0
    for i in range(Npatc):
        for j in range(nbeads):
            coords[j,i,:] = xyz[s,:]
            s = s + 1
    for k in range(integrator.getNumCopies()):
        simulation.integrator.setPositions(k,coords[k,:,:])
        

    # initialize 
    Etot = 0.0      # total energy , to be minimized
    Eint = 0.0      # interaction potential energy
    Esp  = 0.0       # spring energy
    ftot = np.zeros( (nbeads,Npatc, ndim) ) # total force vector on each beads
    fsp = np.zeros( (nbeads,Npatc, ndim) )
    fint = np.zeros( (nbeads,Npatc, ndim) )
    pos = np.zeros( (nbeads,Npatc, ndim) )   # TODO: reuse coords
    
    
    # get E, force, pos  from openMM 
    for j in range(simulation.integrator.getNumCopies()):
        state = simulation.integrator.getState(j,getEnergy=True, getForces=True,getPositions=True)        
        Eint = Eint + state.getPotentialEnergy().value_in_unit(kilojoule/mole)
 
        fint[j] = state.getForces().value_in_unit(kilojoules/nanometer/mole)
        pos[j] = state.getPositions().value_in_unit(nanometer)
    
    
    # get from computing explicitly
    Esp = calculate_Esp(pos)
    fsp = calculate_Fsp(pos)

    
    # combine to total
    Etot = Eint + Esp    # Etot is the energy to be minimized
    ftot = fint + fsp
    #print('ftot= ',ftot)

    # Converting the forces back to the form consistent with the optimization routine
    rforces = np.zeros( [ Npatc*nbeads ,ndim ],float)    # TODO: faster transform
    s = 0
    for i in range(Npatc):
        for j in range(nbeads):
            rforces[s,:] = ftot[j,i,:]
            s = s + 1
    #print('-rforces= ', -rforces )
    global step
    global debug_info
    if DEBUG:
        if step == 0:
            debug_info.update({
                "box": box,
                "xyz": xyz.reshape(Npatc,nbeads, ndim).tolist(),
                "E": Etot,
                "Esp": Esp,
                "grad_E": (-rforces.reshape(Npatc,nbeads, ndim)).tolist()
            })
            step += 1
    return Etot, -rforces.flatten()
###########################################################################################



# %%
########################################################
#          Minimization
########################################################
print("\n\n    Minimizing    \n\n")

results = optimize.minimize(target_function, xyz_0, method='L-BFGS-B', jac = True, options=dict(maxiter=50000,disp=True, maxfun=100000, gtol=1e-8)) 


# %% 
print("\nProcessing Results ...")

## Minimized coordinates
new_coords = results.x.reshape(-1,3)
coords_min = np.zeros([nbeads,Npatc,3],float)
s = 0
for i in range(Npatc):
    for j in range(nbeads):
        coords_min[j,i,:] = new_coords[s,:] 
        s = s + 1

## save 
coords_min_pos = coords_min * 10  # from nanomters to angstrom
f = open( out_pos ,'w')   # use POS class
f.write( '%d %.6lf %.6lf %.6lf \n' % (pos_head[0], pos_head[1], pos_head[2], pos_head[3] ) )
for i in range(Npatc):
    for j in range(nbeads):
        f.write('%20.16g %20.16g %20.16g\n' % (coords_min_pos[j,i,0],coords_min_pos[j,i,1],coords_min_pos[j,i,2]))
f.close()    


# %% 
## Minimized potential and spring energy
for k in range(integrator.getNumCopies()):
    simulation.integrator.setPositions(k,coords_min[k,:,:])
pe = 0.0
for i in range(simulation.integrator.getNumCopies()):
    state = simulation.integrator.getState(i,getEnergy=True)
    pe = pe + state.getPotentialEnergy().value_in_unit(kilojoule/mole)
Esp_min = calculate_Esp(coords_min)
Emin = pe + Esp_min

## save
df = pd.DataFrame(data={'E_0':[E0], 'Esp_0':[Esp_0], 'E_min':[Emin], 'Esp_min':[Esp_min] })
#print(df)
df.to_csv(out_energy, index=False)

if DEBUG:
    debug_info.update({
        "xyz_IS": np.array(results.x).reshape(Npatc,nbeads, ndim).tolist(),
        "E_IS": Emin,
        "Esp_IS": Esp_min
    })
    with open(f"aziz1995-N{Npatc}-Nbeads{nbeads}.json", "w") as f:
        json.dump(debug_info, f, indent=4)




# %%
print("------------------------------------------")
print("End")
print( datetime.now() )
print("------------------------------------------")
