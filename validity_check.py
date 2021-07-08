import cv2
import glob

def sort_by_number(path_names_):
  if 'laplacian' in path_names_:
    
    return int(path_names_.split('laplacian')[-1][1:-4])
  elif 'real' in path_names_:
    
    return int(path_names_.split('real')[-1][1:-4])
    
def validity_check_data(processed, transform):
  for i in range(len(processed)):
    
    p = int(processed[i].split('real')[-1][1:-4])
    t = int(transform[i].split('laplacian')[-1][1:-4])
    flag = []
    if p == t:
      flag.append(1)
    else:
      flag.append(0)
  if set(flag) == {1}:
    print("-._.-._.-Valid-._.-._.-")
  else:
    print("WARNING : Validity Check Failed. Sync Error keys : {} and {}".format(p, t))
