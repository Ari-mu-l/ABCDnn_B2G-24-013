# python unweight_pd.py Major_2018UL_mc_p100.root

import pandas as pd
import numpy as np
import sys
import uproot

def unweight_pd(pddata):
  nrows = pddata.shape[0]
  xsec_unique = pd.unique(pddata['xsecWeight'])
  #print('xsecWeight unique: {}'.format(xsec_unique))

  xsec_min = xsec_unique.min()
  frac_per_xsec = xsec_unique/(xsec_unique.sum())
  nExpect_per_xsec = (frac_per_xsec * nrows).astype(int)

  np.random.seed(0)

  for i in range(len(xsec_unique)):
    datatoexpand = pddata.loc[pddata['xsecWeight']==xsec_unique[i]]
    nexist = datatoexpand.shape[0]
    del_n = nExpect_per_xsec[i] - nexist

    if (del_n > 0):
      if (del_n > nexist):
        for j in range(del_n//nexist):
          pddata = pddata.append(datatoexpand.sample(n=nexist, random_state=1))
        pddata = pddata.append(datatoexpand.sample(n=del_n%nexist, random_state=1))
      else:
        pddata = pddata.append(datatoexpand.sample(n=del_n, random_state=1))
    else:
      pddata = pddata.drop(np.random.choice(datatoexpand.index, -del_n, replace=False))
        
    print("N pddata before: {}, N pddata expected: {}, N pddata after: {}".format(nexist, nExpect_per_xsec[i], pddata[pddata['xsecWeight']==xsec_unique[i]].shape[0]))
  return pddata

# read MC and data
fMajor  = uproot.open( sys.argv[1] )
tMajor  = fMajor[ 'Events' ]
dfMajor  = tMajor.arrays(library="pd")
dfMajor_unweighted = unweight_pd(dfMajor)

outfile = uproot.recreate("{}_unweighted.root".format(sys.argv[1].split(".")[0]))
outfile["Events"] = dfMajor_unweighted

print("Saved {}_unweighted.root to file.".format(sys.argv[1].split(".")[0]))
