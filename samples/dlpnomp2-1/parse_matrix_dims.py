import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import exists
import re
import statistics
import subprocess
import sys

#== matrices of interest ==#
MATRICES = [ 
     "S_mn_nn (Fme)", "T_ia_[n] (Fme)", "S_mn_ij (Fme)", "L_iajb_[mn] (Fme)", "T_n_temp (Fme)", # from compute_Fme
     "S_ii_mn (Wmnij)", "T_ia_[i] (Wmnij)", "K_bar_[mn] (Wmnij)", "T_i_mn (Wmnij)", "K_bar_[nm] (Wmnij)", "S_mn_ij (Wmnij)", "K_iajb_[mn] (Wmnij)" # from compute_Wmnij
]

#== split matrix output data into separate .txt files, one per matrix ==#
def split_output(filename):
  for matrix in MATRICES:
    #== determine name of new matrix-specific output file ==#
    filename_clean = filename[:-4]
    
    matrix_clean = matrix.replace(" ", "_")
    matrix_clean = matrix_clean.replace("[", "")
    matrix_clean = matrix_clean.replace("(", "")
    matrix_clean = matrix_clean.replace("]", "")
    matrix_clean = matrix_clean.replace(")", "")

    #== actually do split, using grep outputted to a separate file ==#
    with open(os.path.join(os.getcwd(), f'{filename_clean}_{matrix_clean}.txt'), "w") as f:
      #subprocess.run(['grep', matrix.replace("[", "\[").replace("]", "\]"), filename], stdout=f)
      grep = subprocess.Popen(['grep', matrix.replace("[", "\[").replace("]", "\]"), filename], stdout=subprocess.PIPE)
      sort = subprocess.run(['sort', '-u'], stdin=grep.stdout, stdout=f)
      grep.wait() 

#== read and acquire matrix dimensions for each combination of lmos (ij) and (mn) from calculation ==#
def parse_dims(filename):
  filename_clean = filename[:-4]

  matrix_dims = { matrix: {} for matrix in MATRICES }
 
  for matrix in MATRICES:
    matrix_clean = matrix.replace(" ", "_")
    matrix_clean = matrix_clean.replace("[", "")
    matrix_clean = matrix_clean.replace("(", "")
    matrix_clean = matrix_clean.replace("]", "")
    matrix_clean = matrix_clean.replace(")", "")

    #== open and scan matrix-specific file ==# 
    with open(f'{filename_clean}_{matrix_clean}.txt', "r") as f:
      for line in f.readlines():
        #== this line has matrix dimension information; read it in ==# 
        if "LMO pairs" in line:
          ij_str, mn_str, rows, columns = re.findall("[-]?[0-9]+", line)

          ij = int(ij_str)
          mn = int(mn_str)

          if ij not in matrix_dims[matrix].keys():
            matrix_dims[matrix][ij] = {}
          if mn not in matrix_dims[matrix][ij].keys():
            matrix_dims[matrix][ij][mn] = {}
            
          #== make sure matrix dimension info hasnt changed across LCCSD iterations if data already present... ==#
          if matrix_dims[matrix][ij][mn] != {}:
            assert matrix_dims[matrix][ij][mn]["rows"] == int(rows), f'LMO pair ({ij}), ({mn}) {matrix} row dim matches previous (' + str(matrix_dims[matrix][ij][mn]["rows"]) + " vs. " + rows + ")"
            assert matrix_dims[matrix][ij][mn]["columns"] == int(columns), f'LMO pair ({ij}), ({mn}) {matrix} column dim matches previous (' + str(matrix_dims[matrix][ij][mn]["columns"]) + " vs. " + columns + ")"
            matrix_dims[matrix][ij][mn]["count"] += 1

          #== ... otherwise store matrix dimensions ==#
          else:
            matrix_dims[matrix][ij][mn] = {"rows": int(rows), "columns": int(columns), "area": int(rows)*int(columns), "count": 1 }

  #== check dimensional sanity of read-in matrices ==#
  for ij in matrix_dims["S_mn_nn (Fme)"].keys(): # check Fme multiplies
    for mn in matrix_dims["S_mn_nn (Fme)"][ij].keys():
      assert(matrix_dims["S_mn_nn (Fme)"][ij][mn]["rows"] == matrix_dims["T_n_temp (Fme)"][ij][mn]["rows"])
      assert(matrix_dims["S_mn_nn (Fme)"][ij][mn]["columns"] == matrix_dims["T_ia_[n] (Fme)"][ij][mn]["rows"])
      assert(matrix_dims["T_ia_[n] (Fme)"][ij][mn]["columns"] == matrix_dims["T_n_temp (Fme)"][ij][mn]["columns"])

      assert(matrix_dims["S_mn_ij (Fme)"][ij][mn]["rows"] == matrix_dims["L_iajb_[mn] (Fme)"][ij][mn]["rows"])
      assert(matrix_dims["L_iajb_[mn] (Fme)"][ij][mn]["columns"] == matrix_dims["T_n_temp (Fme)"][ij][mn]["rows"])

  for ij in matrix_dims["S_ii_mn (Wmnij)"].keys(): # check Wmnij multiplies involving single LMOs
    for mn in matrix_dims["S_ii_mn (Wmnij)"][ij].keys():
      assert(matrix_dims["S_ii_mn (Wmnij)"][ij][mn]["rows"] == matrix_dims["T_ia_[i] (Wmnij)"][ij][mn]["rows"])
     
  for ij in matrix_dims["K_bar_[mn] (Wmnij)"].keys(): # check Wmnij multiplies involving other single LMOs
    for mn in matrix_dims["K_bar_[mn] (Wmnij)"][ij].keys():
      assert matrix_dims["K_bar_[mn] (Wmnij)"][ij][mn]["columns"] == matrix_dims["T_i_mn (Wmnij)"][ij][mn]["columns"] 

  for ij in matrix_dims["S_mn_ij (Wmnij)"].keys(): # check Wmnij multiplies involving LMOs pairs
    for mn in matrix_dims["S_mn_ij (Wmnij)"][ij].keys():
      assert(matrix_dims["S_mn_ij (Wmnij)"][ij][mn]["rows"] == matrix_dims["K_iajb_[mn] (Wmnij)"][ij][mn]["rows"])
      assert(matrix_dims["K_iajb_[mn] (Wmnij)"][ij][mn]["columns"] == matrix_dims["S_mn_ij (Wmnij)"][ij][mn]["rows"])

  #== we are done! ==#
  return matrix_dims

#== squash down read-in matrix dimensions, from per-lmo pair combo, to a single array per matrix ==#
def process_dims(matrix_dims):
  matrix_dims_processed = {} 
  for calculation in matrix_dims.keys():
    matrix_dims_processed[calculation] = { matrix: [] for matrix in MATRICES }
  
    for matrix, ijs in matrix_dims[calculation].items():
      for ij, mns in ijs.items():
        for mn, dims in mns.items():
          matrix_dims_processed[calculation][matrix].append(dims) 

  return matrix_dims_processed

#== generate histograms of area distributions for each matrix, for a given calculation ==#
def matrix_histograms(matrix_dims_processed):
  for calculation in matrix_dims_processed.keys():
    mol = calculation.split("_")[0]
 
    with open(os.path.join(os.getcwd(), f'postprocess/{mol}/{calculation}_mat.txt'), "w") as f:
      for matrix, results in matrix_dims_processed[calculation].items():
        f.write(f'{mol} {matrix} \n') 
      
        #== first, we want the number of operations this matrix was involved in throughout the calculation ==#
        f.write(f'  {matrix} Number of occurrences per LCCSD iteration: ' + str(sum([ result["count"] for result in results ])) + " \n")
 
        #== now we collect area, row, and column data ==#
        for key in filter(lambda _: _ != "count", results[0].keys()): 
          key_counts = [ result[key] for result in results ]          
          f.write(f'  {matrix} Maximum {key} count: ' + str(max(key_counts)) + "\n")
          f.write(f'  {matrix} Mean {key} count: ' + str(statistics.mean(key_counts)) + " \n")
          f.write(f'  {matrix} Median {key} count: ' + str(statistics.median(key_counts)) + " \n")
          f.write(f'  {matrix} Minimum {key} count: ' + str(min(key_counts)) + "\n")
          f.write("\n")

        #== now we take a ratio of the area to the # of occurrences ==#
        area_counts = [ result["area"] for result in results ]
        f.write(f'  {matrix} Median Area/Occurrence Ratio: ' + str(statistics.median(area_counts)/sum([ result["count"] for result in results ])) + "\n")
     
        #== now we make a histogram showing the frequency of different matrix sizes ==#
        np_counts, np_bins = np.histogram(area_counts)
        plt_counts, plt_bins, bars = plt.hist(np_bins[:-1], np_bins, weights=np_counts, edgecolor='white')

        #== clean up/label histogram some ==#
        matrix_clean = matrix.split(" ")[0]
        matrix_clean = matrix_clean.replace("[", "")
        matrix_clean = matrix_clean.replace("(", "")
        matrix_clean = matrix_clean.replace("]", "")
        matrix_clean = matrix_clean.replace(")", "")

        plt.xlabel(f'Area of {matrix_clean}')
        plt.ylabel("Number of Occurrences per Iter")

        plt.minorticks_on()
  
        plt.bar_label(bars, fontsize=10, color='black')

        #== save current histogram to file ==#
        plt.title(f'Histogram of Areas of {matrix_clean} in {mol}')
        filename = f'{mol}_{matrix_clean}_hist.png'
        plt.savefig(os.path.join(os.getcwd(), f'postprocess/{mol}/{filename}'))

        #== clear out internal data for next histogram to write properly ==#
        plt.clf()
        plt.cla() 

#== generate scatterplots comparing values "keyword" of a given matrix for each calculation ==#
def matrix_scatterplot(matrix_dims_processed, keyword="area"):
  for matrix in MATRICES:
    mol_counts = []
    keyword_results = []
    for calculation in matrix_dims_processed.keys():
      mol_counts.append( int(re.search( "[0-9]+", calculation.split("_")[0] ).group()) ) 
      
      areas = [] 
      for ilmopair, lmopair_data in enumerate(matrix_dims_processed[calculation][matrix]):
        areas.append(lmopair_data[keyword])
       
      keyword_results.append(statistics.median(areas))

      keyword_results = [ x for _, x in sorted(zip(mol_counts, keyword_results)) ]
      mol_counts.sort()

    print("  ", matrix, mol_counts, keyword_results)

#== driver, of course ==#
def main(force_overwrite=False):
  #== where our calculation outputs are stored ==#
  test_dir = os.path.join(os.getcwd(), "tests")
 
  #== we will store analysis data in a separate directory ==#
  if not exists(os.path.join(os.getcwd(), "postprocess")): 
    os.mkdir(os.path.join(os.getcwd(), "postprocess"))

  #== determine range of molecules to analyze ==#
  mols_to_counts = {
    #"organic": [-1], 
    "water": range(1, 34 + 1),
    #"benzene": range(1, 9 + 1),
  }

  #== limit calculations to analyze by other factors (e.g., basis set) 
  basis_sets = [ "cc-pVDZ" ]

  #== actually collect info on matrix dimensions ==# 
  matrix_dims = {}
  for mol_dir in os.listdir(test_dir):
    #== check if calculation fits our molecule range ==#
    if any([ x in mol_dir for x in mols_to_counts.keys() ]): 
      mol_name = re.search("[A-Za-z]+", mol_dir).group()
      mol_count = -1 

      if re.search("[0-9]+", mol_dir) is not None:
        mol_count = int(re.search("[0-9]+", mol_dir).group())
 
      if mol_count in mols_to_counts[mol_name]: 
        full_mol_dir = os.path.join(test_dir, mol_dir)
        
        #== loop over output data in molecule to analyze ==# 
        for mol_result in os.listdir(full_mol_dir):
          #== matrix data is stored in *.out file ==#
          if mol_result[-4:] == ".out": 
            #== apply other limiting factors for calculation ==#
            _, basis_set, pno_convergence, thread_count = mol_result.split("_")

            if basis_set not in basis_sets:
              continue 

            #== we can skip this calculation if data for it already exists ==#
            if exists(os.getcwd() + f'/postprocess/{mol_dir}/' + mol_result[:-4] + "_mat.txt") and not force_overwrite:
              print("Skip", mol_name, mol_count, basis_set)
              continue
  
            #== if we are here, we will analyze this calculation ==#
            print(mol_name, mol_count, basis_set)

            if not exists(os.path.join(os.getcwd(), f'postprocess/{mol_dir}')):
              os.mkdir(os.path.join(os.getcwd(), f'postprocess/{mol_dir}'))

            output_file = os.path.join(full_mol_dir, mol_result)
 
            #== split full output file into matrix-specific outputs ==#
            split_output(output_file)
          
            #== retrieve info about matrix dimensions from calculation ==#
            matrix_dims[mol_result[:-4]] = parse_dims(output_file)      
  
  #== reformatting of matrix dims for easier postprocessing ==# 
  matrix_dims_processed = process_dims(matrix_dims)
  
  #== analyze matrices results ==# 
  matrix_histograms(matrix_dims_processed)
  matrix_scatterplot(matrix_dims_processed)
  
if __name__ == "__main__":
  main(True)
