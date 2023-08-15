import matplotlib.pyplot as plt
import numpy as np
import os
import re
import statistics

#== matrices of interest ==#
MATRICES = [ 
     "S_mn_nn (Fme)", "T_ia_[n] (Fme)", "S_mn_ij (Fme)", "L_iajb_[mn] (Fme)", "T_n_temp (Fme)", # from compute_Fme
     "S_ii_mn (Wmnij)", "T_ia_[i] (Wmnij)", "K_bar_[mn] (Wmnij)", "T_i_mn (Wmnij)", "K_bar_[nm] (Wmnij)", "S_mn_ij (Wmnij)", "K_iajb_[mn] (Wmnij)" # from compute_Wmnij
]
  
#== read and acquire matrix dimensions for each combination of lmos (ij) and (mn) from calculation ==#
def parse_dims(filename):
  matrix_dims = { matrix: {} for matrix in MATRICES }
 
  #== open and scan file ==# 
  with open(filename, "r") as f:
    for line in f.readlines():
      #== this line has matrix dimension information; read it in ==# 
      if "LMO pairs" in line:
        ij_str, mn_str, rows, columns = re.findall("[-]?[0-9]+", line)

        ij = int(ij_str)
        mn = int(mn_str)

        #== sort out matrix dimension info in read-in line ==#
        for matrix in MATRICES:
          #== check which matrix the current line describes ==#
          if f'LMO pairs ({ij}), ({mn}) {matrix}' in line:
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
            continue

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
  matrix_dims_processed = { matrix: [] for matrix in MATRICES }
  
  for matrix, ijs in matrix_dims.items():
    for ij, mns in ijs.items():
      for mn, dims in mns.items():
        matrix_dims_processed[matrix].append(dims) 

  return matrix_dims_processed

def analyze_matrices(mol, matrix_dims_processed):
  for matrix, results in matrix_dims_processed.items():
    print(matrix) 
    #print(results)
    print("  Number of occurrences: ", sum([ result["count"] for result in results ]))
    print()
  
    for key in filter(lambda _: _ != "count", results[0].keys()): 
      key_counts = [ result[key] for result in results ]
      print(f'  Maximum {key} count: ', max(key_counts))
      print(f'  Mean {key} count: ', statistics.mean(key_counts))
      print(f'  Median {key} count: ', statistics.median(key_counts))
      print(f'  Minimum {key} count: ', min(key_counts))
      print()

    area_counts = [ result["area"] for result in results ]
    print(len(area_counts))
    counts, bins = np.histogram(area_counts)
    ax = plt.hist(bins[:-1], bins, weights=counts) 
    #ax = plt.stairs(counts, bins)

    #print(ax)
    #ax.set_xlabel("Area") 
    #ax.ylabel("Number of Occurrences")

    matrix_clean = matrix.split(" ")[0]
    matrix_clean = matrix_clean.replace("[", "")
    matrix_clean = matrix_clean.replace("(", "")
    matrix_clean = matrix_clean.replace("]", "")
    matrix_clean = matrix_clean.replace(")", "")

    #ax.set_title(f'{mol} {matrix_clean}') 
    filename = f'{mol}_{matrix_clean}_hist.png'
    plt.savefig(os.path.join("hist", filename))

def main():
  #== determine calculation to analyze ==#
  mol = "benzene1"
  output_file = os.path.join(os.getcwd(), f'{mol}.out')
  
  #== retrieve info about matrices from calculation ==#
  matrix_dims = parse_dims(output_file)      
  matrix_dims_processed = process_dims(matrix_dims)
 
  #== analyze matrices results ==#
  analyze_matrices(mol, matrix_dims_processed)
 
if __name__ == "__main__":
  main()
