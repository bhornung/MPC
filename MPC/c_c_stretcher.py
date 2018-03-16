"""
This script will generate a strectched molecule. It tries to localise the stretching to the
C--C bond by aligning the molecule to the X axis.

replace the <Atoms/> tag in the Guitar.xml with the list of atoms generated in this file.

"""
import numpy as np

def read_pdb(path_to_pdb):

  at_dict = {'C' : 'Carbon', 'H' : 'Hydrogen'}
  coords = []
  atoms = []

  with open(path_to_pdb, 'r') as fproc:

    for line in fproc:
      if line.find('HETATM') > -1:
          coords.append(list(map(lambda x: float(x),  line.split()[5:8])))
          atoms.append(at_dict[line.split()[2]])

  coords = np.array(coords) / 10.0

  return coords, atoms

def centre(coords):
  cms = np.mean(coords[:54], axis = 0)
  coords = coords - cms
  return coords 

def assemble_rot_mat(angle):
  rot_mat = np.array([[ np.cos(angle), np.sin(angle), 0.0],
                      [-np.sin(angle), np.cos(angle), 0.0],
                      [ 0.0, 0.0, 1.0]])
  return rot_mat

def rotate(coords, rot_mat):
  cp = np.dot(coords[:,None,:], rot_mat)[:,0,:]
  return cp

def scale_xy(coords, scafac):
  scf = np.array([scafac, scafac, 1.0])
  coords = coords * scf[None,] 
  return coords

def print_atoms(coords, atoms, names, fn):
  with open(fn,'w') as fproc:
    fproc.write("<Atoms>\n")

    for _at, _nm, _xyz in zip(atoms, names, coords):
      fproc.write('<Atom Element="{0}" Name="{1}" Position="{2},{3},{4}"/>\n'.format(_at, _nm, *_xyz))
    fproc.write("</Atoms>\n")


if __name__ == "__main__":

# ---> change this to your path to the pdb file
  f_pdb = r'C:\Users\Balazs\Desktop\bhornung_movies\guitar\string_pymol.pdb'

  names_ = ["C_{0}".format(idx) for idx in range(53)]
  names_.append("H")
  names_.insert(0,"C")
  _names = ["H_{0}".format(idx) for idx in range(109)]  
  names = names_ + _names

  rot_mat_p = assemble_rot_mat(np.radians(109.42/2))
  rot_mat_m = assemble_rot_mat(-np.radians(109.42/2))
  
# ---> change this to scale with different factors
  scafacs = [1.0, 1.2, 1.4, 1.5, 2.0]
  for scafac in scafacs:
    coords, atoms = read_pdb(f_pdb)

    coords = centre(coords)

    coords = rotate(coords, rot_mat_p)

    coords = scale_xy(coords,scafac)

    coords = rotate(coords, rot_mat_m)

# ---> change this to a meaningful path. It currently writes stuff to the cwd
    f_out = "atom_list_"+str(scafac)+"_.xml"
    print_atoms(coords, atoms, names, f_out)