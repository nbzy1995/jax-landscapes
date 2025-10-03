"""
This file handles the loading of PIMC worldline data, output from a Path Integral Monte Carlo simulation for bosons.

"""

from __future__ import annotations

from typing import Dict, Any
import logging
import numpy as np
import jax.numpy as jnp


def load_pimc_worldline_file(fileName: str, Lx=None, Ly=None, Lz=None) -> Dict[int, 'Path']:
    """
    Load a pimc output worldline file ("ce-wl-*.dat") and return a dictionary of Path objects.

    Args:
        fileName: Path to the worldline file
        Lx, Ly, Lz: Optional box dimensions (Angstroms). If not provided, Path objects
                    will have None for box dimensions.

    Returns:
        Dictionary mapping configuration number to Path object: {config_number: Path}
    """
    with open(fileName, 'r') as wlFile:
        lines = wlFile.readlines()

    paths_dict = {}
    cfg_id = None
    data = []

    for line in lines:
        if 'START_CONFIG' in line:
            data = []
            parts = line.split()
            cfg_id = int(parts[-1])  # Extract configuration number
        elif 'END_CONFIG' in line:
            if cfg_id is not None and len(data) > 0:
                paths_dict[cfg_id] = Path(data, Lx=Lx, Ly=Ly, Lz=Lz)
            data = []
        elif line and line[0] == '#':
            continue
        else:
            data.append(line.split())

    return paths_dict


class Path:
    """
    Holds all the information about one PIMC configuration (snapshot at one MC step). The connectivity information is stored.
    """

    def __init__(self, wlData, Lx=None, Ly=None, Lz=None):
        ''' 
        wlData: for one snapshot, it is a list of lines, each line is information about one bead in the worldline.
        '''

        if len(wlData) == 0:
            raise ValueError("wlData is empty. Cannot initialize Path object.")
        self.wlData = np.array(wlData,dtype=float)

        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        # Determine the number of time slices and particles from the data
        self.numTimeSlices = int(np.max(self.wlData[:,0])) + 1
        self.numParticles = int(np.max(self.wlData[:,1])) + 1

        if self.numParticles <= 0:
            raise ValueError("Invalid wlData: Number of particles must be greater than zero.")
        if self.numTimeSlices <= 0:
            raise ValueError("Invalid wlData: Number of time slices must be greater than zero.")

        # Read out the data from the worldline path file `(g)ce-wl-*.dat`
        self.is_closed_worldline = True  # whether the worldline is closed or has some open ends
        self.beadCoord    = np.zeros([self.numTimeSlices,self.numParticles,3],float)
        self.next    = np.zeros([self.numTimeSlices,self.numParticles,2],int)
        self.prev    = np.zeros([self.numTimeSlices,self.numParticles,2],int)
        self.wlIndex = np.zeros([self.numTimeSlices,self.numParticles],int)

        # Store the order in which beads appear in the input file for output preservation
        self.write_order = np.zeros([len(self.wlData), 2], int)

        for idx, line in enumerate(self.wlData):
            m = int(line[0])                    # time slice idx, 0 to numTimeSlices-1
            n = int(line[1])                    # particle idx, 0 to numParticles-1
            self.write_order[idx] = [m, n]      # Store the order for writing output
            self.wlIndex[m,n] = int(line[2])    # TODO: what is this for?
            if self.wlIndex[m,n] < 0:
                # logging.warning(f"Open worldline path found: negative wlIndex found: {self.wlIndex[m,n]} at time slice {m}, particle {n}. ")
                self.is_closed_worldline = False
            self.beadCoord[m,n,0] = float(line[3])   # current x position
            self.beadCoord[m,n,1] = float(line[4])   # current y position
            self.beadCoord[m,n,2] = float(line[5])   # current z position
            self.prev[m,n,0] = int(line[6])     # time slice idx of the prev bead connected to bead [m,n]
            self.prev[m,n,1] = int(line[7])     # particle idx of the prev bead connected to bead [m,n]
            self.next[m,n,0] = int(line[8])     # time slice idx of the next bead connected to bead [m,n]
            self.next[m,n,1] = int(line[9])     # particle idx of the next bead connected to bead [m,n]
  
        # Compute cycle information        
        self.cycleIndex = -1* np.ones([self.numTimeSlices,self.numParticles],int) # store the cycle index which each bead belongs to
        self.cycleSizeDist = np.array([], int) # store the size of each cycle, with index i corresponding to cycleIndex = i
        if self.is_closed_worldline:
            self.label_cycles()
        else:
            logging.info("Open worldline path found: cycleIndex will not be set.")


    def label_cycles(self):
        ''' 
        Identify and label cycles index for all beads in the worldline path.
        '''
        # Need to loop through all beads in the system. we label bead_todo[m,n] = 0 after the bead[m,n] is touched.
        bead_todo = np.ones([self.numTimeSlices, self.numParticles])
        
        cycle_count = 0
        # start a new cycle if there are beads left to touch
        while np.any(bead_todo > 0):
            # print(f"Cycle {cycle_count+1} started...")  # Debugging output
            cycle_count += 1 
            # find a bead which hasn't been touched yet and use it
            # as a starting bead of a new cycle.
            startBeadIdx = np.argwhere(bead_todo)[0] # gives for e.g., [2,3]
            bead_todo[*startBeadIdx] = 0 # touch it
            bead_count = 1 # count the number of beads in the cycle
            self.cycleIndex[*startBeadIdx] = cycle_count-1 # to make the cycle index start from 0
            
            # # store array of all bead in cycle.
            # beadCoords = np.array([self.beadCoord[startBeadIdx]])

            # step within cycle, break once we return to the startBeadIdx
            currBeadIdx = self.next[*startBeadIdx]

            while ( np.any(currBeadIdx != startBeadIdx) ):
                bead_todo[*currBeadIdx] = 0
                bead_count += 1
                # store current location
                # beadCoords = np.append(beadCoords, [self.beadCoord[currBeadIdx]], axis=0)
                self.cycleIndex[*currBeadIdx] = cycle_count-1
                currBeadIdx = self.next[*currBeadIdx]
            
            # store the size of the cycle, where index i corresponds to cycleIndex = i
            self.cycleSizeDist = np.append(self.cycleSizeDist, bead_count)

            # # reshape the bead location arrays
            # beadCoords = np.reshape(beadCoords, (bead_count,3))
            # permutation number of current cycle
            # permNum = bead_count/self.numTimeSlices

            # -----------------------------------------------------------------
            # # compute winding of the current worldline
            # W = self.computeWinding(beadCoords)

            # ----
            # TODO: check that it is indeed closed cycle. throw error if not. This means the wl file has wrong info about closedness.


def write_pimc_worldline_config(file_handle, path, config_number):
    """
    Write a single PIMC configuration to an open file in worldline format.

    This format is compatible with load_pimc_worldline_file() and visualization tools.

    Args:
        file_handle: Open file handle to write to
        path: Path object containing beadCoord, next, prev, wlIndex arrays
        config_number: Configuration number (e.g., iteration number for minimization)
    """
    file_handle.write(f"# START_CONFIG {config_number:06d}\n")

    M, N, _ = path.beadCoord.shape  # time slices, particles

    # Use write_order if available to preserve input file ordering
    if hasattr(path, 'write_order') and path.write_order is not None:
        # Iterate in the order specified by write_order
        for m, n in path.write_order:
            # Get coordinates
            x, y, z = path.beadCoord[m, n]

            # Get connectivity
            prev_m, prev_n = path.prev[m, n]
            next_m, next_n = path.next[m, n]

            # Get wlIndex (if available)
            wl_idx = path.wlIndex[m, n] if hasattr(path, 'wlIndex') else 1

            # Format: fixed width fields, scientific notation for coordinates
            file_handle.write(
                f"{m:7d} {n:8d} {wl_idx:8d}   "
                f"{x:12.3E}   {y:12.3E}   {z:12.3E}   "
                f"{prev_m:8d} {prev_n:8d} {next_m:8d} {next_n:8d}\n"
            )
    else:
        # Fall back to (m,n) nested loop order for backwards compatibility
        for m in range(M):
            for n in range(N):
                # Get coordinates
                x, y, z = path.beadCoord[m, n]

                # Get connectivity
                prev_m, prev_n = path.prev[m, n]
                next_m, next_n = path.next[m, n]

                # Get wlIndex (if available)
                wl_idx = path.wlIndex[m, n] if hasattr(path, 'wlIndex') else 1

                # Format: fixed width fields, scientific notation for coordinates
                file_handle.write(
                    f"{m:7d} {n:8d} {wl_idx:8d}   "
                    f"{x:12.3E}   {y:12.3E}   {z:12.3E}   "
                    f"{prev_m:8d} {prev_n:8d} {next_m:8d} {next_n:8d}\n"
                )

    file_handle.write("# END_CONFIG\n") 
