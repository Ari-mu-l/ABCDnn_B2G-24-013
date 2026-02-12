# python3 makeHist_abcdnn.py -s rootFiles_logBpMlogST_Oct2024Run2_pNetWeight/Case14/OctMajor_all_mc_p100.root -b rootFiles_logBpMlogST_Oct2024Run2_pNetWeight/Case14/OctMinor_all_mc_p100.root -t rootFiles_logBpMlogST_Oct2024Run2_pNetWeight/Case14/OctData_all_data_p100.root -f 1.0 -m logBpMlogST_mmd1_case14_random104

################
# NOTE: Temporarily commented out tgt and mnr related lines, becuase exceed memory limit with pNet applied
################

import numpy as np
import os, tqdm
import abcdnn
import uproot, ROOT
from argparse import ArgumentParser
from json import loads as load_json
from array import array
import samples
from utils import *

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import config

parser = ArgumentParser()
parser.add_argument( "-s", "--source", required = True  )
parser.add_argument( "-t", "--target", required = True  )
parser.add_argument( "-b", "--minor" , required = True  )
parser.add_argument( "-m", "--tag"   , required = True  )
#parser.add_argument( "-c", "--case"  , required = False )

args = parser.parse_args()

# histogram settings
bin_lo = 400 #0
bin_hi = 2500
Nbins  = 420 # 42
validationCut = 850

log = False # set to False if want BpM instead of log(BpM)
if log:
  validationCut = np.log(validationCut)
  bin_lo = np.log(bin_lo)
  bin_hi = np.log(bin_hi)

folder = config.params[ "MODEL" ][ "SAVEDIR" ]
folder_contents = os.listdir( folder )

isTest = True
if isTest:
  testDir = args.tag
  if not os.path.exists(testDir):
    os.makedirs(testDir+'/')
else: testDir = ''

print( ">> Reading in {}.json for hyper parameters...".format( args.tag ) )
with open( os.path.join( folder, args.tag + ".json" ), "r" ) as f:
  params = load_json( f.read() )

print( ">> Setting up NAF model..." )

variables = [ str( key ) for key in sorted( config.variables.keys() ) if config.variables[key]["TRANSFORM"] ]
if config.regions["Y"]["VARIABLE"] in config.variables and config.regions["X"]["VARIABLE"] in config.variables:
  variables.append( config.regions["Y"]["VARIABLE"] )
  variables.append( config.regions["X"]["VARIABLE"] )
else:
  sys.exit( "[ERROR] Control variables not listed in config.variables, please add. Exiting..." )

categorical = [ config.variables[ vName ][ "CATEGORICAL" ] for vName in variables ]
lowerlimit  = [ config.variables[ vName ][ "LIMIT" ][0] for vName in variables ]
upperlimit  = [ config.variables[ vName ][ "LIMIT" ][1] for vName in variables ]

print( ">> Found {} variables: ".format( len( variables ) ) )
for i, variable in enumerate( variables ):
  print( "  + {}: [{},{}], Categorical = {}".format( variable, lowerlimit[i], upperlimit[i], categorical[i] ) )

def processInput(fileName):
  # fileName: args.target, args.minor, args.source
  print( ">> Load the data" )
  fFile = uproot.open( fileName )
  fTree = fFile[ "Events" ]
  
  if "Major" in fileName:
    inputs = fTree.arrays( variables, library="pd" )
  elif "Minor" in fileName:
    inputs = fTree.arrays( variables+["xsecWeight"], library="pd" )
    for variable in variables:
      inputs[variable] = inputs[variable].clip(upper = config.variables[variable]["LIMIT"][1])
  elif "Data" in fileName:
    inputs = fTree.arrays( variables, library="pd" )
    for variable in variables:
      inputs[variable] = inputs[variable].clip(upper = config.variables[variable]["LIMIT"][1])

  Bdecay = fTree.arrays( ["Bdecay_obs", "pNetTtagWeight", "pNetTtagWeightUp", "pNetTtagWeightDn", "pNetWtagWeight", "pNetWtagWeightUp", "pNetWtagWeightDn"], library="pd" )

  ##Bdecay_tgt = tTree.arrays( ["Bdecay_obs"], library="pd" )[inputs_tgt["Bprime_mass"]>400]
  ##Bdecay_mnr = mTree.arrays( ["Bdecay_obs"], library="pd" )[inputs_mnr["Bprime_mass"]>400]
  ##inputs_tgt = inputs_tgt[inputs_tgt["Bprime_mass"]>400] # take only BpM>400. pred cut made later.
  ##inputs_mnr = inputs_mnr[inputs_mnr["Bprime_mass"]>400]

  inputs_region = { region: None for region in [ "X", "Y", "A", "B", "C", "D" ] }
  Bdecay_region = { region: None for region in [ "X", "Y", "A", "B", "C", "D" ] }

  for region in [ "X", "Y", "A", "B", "C", "D" ]:
    if config.regions["X"][region][0] == ">=":
      select = inputs[ config.regions["X"]["VARIABLE"] ] >= config.regions[ "X" ][ region ][1]
    elif config.regions["X"][region][0] == "<=":
      select = inputs[ config.regions["X"]["VARIABLE"] ] <= config.regions[ "X" ][ region ][1]
    else:
      select = inputs[ config.regions["X"]["VARIABLE"] ] == config.regions[ "X" ][ region ][1]

    if config.regions["Y"][region][0] == ">=":
      select &= inputs[ config.regions["Y"]["VARIABLE"] ] >= config.regions[ "Y" ][ region ][1]
    elif config.regions["Y"][region][0] == "<=":
      select &= inputs[ config.regions["Y"]["VARIABLE"] ] <= config.regions[ "Y" ][ region ][1]
    else:
      select &= inputs[ config.regions["Y"]["VARIABLE"] ] == config.regions[ "Y" ][ region ][1]

    inputs_region[region] = inputs.loc[select]
    Bdecay_region[region] = Bdecay.loc[select]

  if "Major" in fileName:
    print( ">> Encoding and normalizing source inputs" )
    inputs_enc_region = {}
    encoder = {}
    inputs_nrm_region = {}
    inputmeans = np.hstack( [ float( mean ) for mean in params[ "INPUTMEANS" ] ] )
    inputsigmas = np.hstack( [ float( sigma ) for sigma in params[ "INPUTSIGMAS" ] ] )

    for region in inputs_region:
      encoder[region] = abcdnn.OneHotEncoder_int( categorical, lowerlimit = lowerlimit, upperlimit = upperlimit )
      inputs_enc_region[ region ] = encoder[region].encode( inputs_region[ region ].to_numpy( dtype = np.float32 ) )
      inputs_nrm_region[ region ] = ( inputs_enc_region[ region ] - inputmeans ) / inputsigmas
      inputs_region[ region ][ variables[0] ] = inputs_region[ region ][ variables[0] ].clip(upper=config.variables[variables[0]]["LIMIT"][1])
      inputs_region[ region ][ variables[1] ] = inputs_region[ region ][ variables[1] ].clip(upper=config.variables[variables[1]]["LIMIT"][1])
  
    #get predictions
    predictions = { region: [] for region in [ "X", "Y", "A", "B", "C", "D" ] }

    NAF = abcdnn.NAF( 
      inputdim = params["INPUTDIM"],
      conddim = params["CONDDIM"],
      activation = params["ACTIVATION"], 
      regularizer = params["REGULARIZER"],
      initializer = params["INITIALIZER"],
      nodes_cond = params["NODES_COND"],
      hidden_cond = params["HIDDEN_COND"],
      nodes_trans = params["NODES_TRANS"],
      depth = params["DEPTH"],
      permute = bool( params["PERMUTE"] )
    )
    NAF.load_weights( os.path.join( folder, args.tag ) )

    for region in tqdm.tqdm(predictions):
      NAF_predict = np.asarray( NAF.predict( np.asarray( inputs_nrm_region[ region ] ) ) )
      predictions[ region ] = NAF_predict * inputsigmas[0:2] + inputmeans[0:2]
      ##Bdecay_src_region[region] = Bdecay_src_region[region][predictions[ region ][:,0]>400]
      ##predictions[ region ] = predictions[region][predictions[ region ][:,0]>400] # take only BpM>400
    del NAF

  if "Major" in fileName:
    return predictions, Bdecay_region
  else:
    return inputs_region, Bdecay_region
  

def makeV(fileName, case, inputs_array, weight_array, pNet_array, pNetUp_array, pNetDn_array):
  ##predict_array = predictions["D"]
  ##inputs_tgt_array = inputs_tgt_region["D"].to_numpy(dtype='d')
  ##inputs_mnr_array = inputs_mnr_region["D"].to_numpy(dtype='d')
  ##weight_mnr_array = inputs_mnr_region["D"]["xsecWeight"].to_numpy(dtype='d')

  if "Major" in fileName:
    hist        = ROOT.TH1D(f'Bprime_mass_pre_V', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
    hist_pNetUp = ROOT.TH1D(f'Bprime_mass_pre_V_pNetUp', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
    hist_pNetDn = ROOT.TH1D(f'Bprime_mass_pre_V_pNetDn', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
  elif "Data" in fileName:
    hist        = ROOT.TH1D(f'Bprime_mass_dat_V', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
  elif "Minor" in fileName:
    hist        = ROOT.TH1D(f'Bprime_mass_mnr_V', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
    
  for i in range(len(inputs_array)):
    if inputs_array[i][1]<validationCut and inputs_array[i][0]>bin_lo:
      hist.Fill(inputs_array[i][0], weight_array[i] * pNet_array[i])
  hist.Write()
    
  if ("Major" in fileName) and (case=="case1" or case=="case2"):
    for i in range(len(inputs_array)):
      if inputs_array[i][1]<validationCut and inputs_array[i][0]>bin_lo:
        hist_pNetUp.Fill(inputs_array[i][0], pNetUp_array[i]) # pNetSF_Up
        hist_pNetDn.Fill(inputs_array[i][0], pNetDn_array[i]) # pNetSF_Down
    hist_pNetUp.Write()
    hist_pNetDn.Write()

    
def makeD2(fileName, case, inputs_array, weight_array, pNet_array, pNetUp_array, pNetDn_array):
  if "Major" in fileName:
    hist        = ROOT.TH1D(f'Bprime_mass_pre_D2', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
    hist_pNetUp = ROOT.TH1D(f'Bprime_mass_pre_D2_pNetUp', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
    hist_pNetDn = ROOT.TH1D(f'Bprime_mass_pre_D2_pNetDn', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
  elif "Data" in fileName:
    hist        = ROOT.TH1D(f'Bprime_mass_dat_D2', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
  elif "Minor" in fileName:
    hist        = ROOT.TH1D(f'Bprime_mass_mnr_D2', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)

  for i in range(len(inputs_array)):
    if inputs_array[i][1]>validationCut and inputs_array[i][0]>bin_lo:
      hist.Fill(inputs_array[i][0], weight_array[i] * pNet_array[i])
  hist.Write()

  if ("Major" in fileName) and (case=="case1" or case=="case2"):
    for i in range(len(inputs_array)):
      if inputs_array[i][1]<validationCut and inputs_array[i][0]>bin_lo:
        hist_pNetUp.Fill(inputs_array[i][0], pNetUp_array[i]) # pNetSF_Up
        hist_pNetDn.Fill(inputs_array[i][0], pNetDn_array[i]) # pNetSF_Down
    hist_pNetUp.Write()
    hist_pNetDn.Write()

    
def makeHists_fit(fileName, inputs_region, Bdecay_region, region, case):
  print(fileName)
  caseValue = int(case[-1])
  
  if "Major" in fileName:
    inputs_array = inputs_region[region][ Bdecay_region[region]["Bdecay_obs"]==caseValue ] # name prediction_array as inputs_array
    weight_array = np.ones(len(inputs_array))
  elif "Minor" in fileName:
    inputs_array = inputs_region[region][ Bdecay_region[region]["Bdecay_obs"]==caseValue ].to_numpy(dtype='d')[:,:2]
    weight_array = inputs_region[region]["xsecWeight"][ Bdecay_region[region]["Bdecay_obs"]==caseValue ].to_numpy(dtype='d')
  else: # data
    inputs_array = inputs_region[region][ Bdecay_region[region]["Bdecay_obs"]==caseValue ].to_numpy(dtype='d')[:,:2]
    weight_array = np.ones(len(inputs_array))
  
  if case=="case1" or case=="case2":
    if case=="case1":
      pNetTag = "pNetTtagWeight"
    else:
      pNetTag = "pNetWtagWeight"
    pNet_array = Bdecay_region[region][pNetTag][ Bdecay_region[region]["Bdecay_obs"]==caseValue ].to_numpy(dtype='d')
    pNetUp_array = Bdecay_region[region][f'{pNetTag}Up'][ Bdecay_region[region]["Bdecay_obs"]==caseValue ].to_numpy(dtype='d')
    pNetDn_array = Bdecay_region[region][f'{pNetTag}Dn'][ Bdecay_region[region]["Bdecay_obs"]==caseValue ].to_numpy(dtype='d')
  else: # case3,4,14,23
    # dummy pNet_array: pNet tag should be implemented for case14 and 23, but we do not need the combined cases for now
    pNet_array   = np.ones(len(inputs_array))
    pNetUp_array = np.ones(len(inputs_array))
    pNetDn_array = np.ones(len(inputs_array))
  
  if not log:
    inputs_array = np.exp(inputs_array)
    inputs_array = np.clip(inputs_array, 0, 2500)

  if "Data" in fileName:
    hist = ROOT.TH1D(f'Bprime_mass_dat_{region}', "Bprime_mass", Nbins, bin_lo, bin_hi)
  elif "Minor" in fileName:
    hist = ROOT.TH1D(f'Bprime_mass_mnr_{region}', "Bprime_mass", Nbins, bin_lo, bin_hi)
  elif "Major" in fileName:
    hist = ROOT.TH1D(f'Bprime_mass_pre_{region}', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
    if case=="case1" or case=="case2":
      hist_pNetUp = ROOT.TH1D(f'Bprime_mass_pre_{region}_pNetUp', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
      hist_pNetDn = ROOT.TH1D(f'Bprime_mass_pre_{region}_pNetDn', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
  
  # fill target hist. Same for all cases. No pNetSF needed
  for i in range(len(inputs_array)):
    if inputs_array[i][0]>bin_lo:
      hist.Fill(inputs_array[i][0], weight_array[i] * pNet_array[i])
  hist.Write()

  if ("Major" in fileName) and (case=="case1" or case=="case2"):
    for i in range(len(inputs_array)):
      if inputs_array[i][0]>bin_lo:
        hist_pNetUp.Fill(inputs_array[i][0], pNetUp_array[i])
        hist_pNetDn.Fill(inputs_array[i][0], pNetDn_array[i])
    hist_pNetUp.Write()
    hist_pNetDn.Write()

  return inputs_array, weight_array, pNet_array, pNetUp_array, pNetDn_array

if 'case14' in args.tag:
  ##case_list = ["case14", "case1", "case4"]
  case_list = ["case1", "case4"]
elif 'case23' in args.tag:
  ##case_list = ["case23", "case2", "case3"]
  case_list = ["case2", "case3"]

def main(fileName, tfileOption):
  inputs_region, Bdecay_region = processInput(fileName)

  for case in case_list:
    if log:
      histFile = ROOT.TFile.Open(f'{testDir}hists_ABCDnn_{case}_{bin_lo}to{bin_hi}_{Nbins}_log_pNet.root', tfileOption)
    else:
      histFile = ROOT.TFile.Open(f'{testDir}hists_ABCDnn_{case}_{bin_lo}to{bin_hi}_{Nbins}_pNet.root', tfileOption)

    makeHists_fit(fileName, inputs_region, Bdecay_region, "A", case)
    makeHists_fit(fileName, inputs_region, Bdecay_region, "B", case)
    makeHists_fit(fileName, inputs_region, Bdecay_region, "C", case)
    inputs_array , weight_array, pNet_array, pNetUp_array, pNetDn_array = makeHists_fit(fileName, inputs_region, Bdecay_region, "D", case)
    #makeHists_fit(fileName, inputs_region, Bdecay_region, "X", case)
    #makeHists_fit(fileName, inputs_region, Bdecay_region, "Y", case)

    makeV(fileName, case, inputs_array , weight_array, pNet_array, pNetUp_array, pNetDn_array)
    makeD2(fileName, case, inputs_array , weight_array, pNet_array, pNetUp_array, pNetDn_array)
    histFile.Close()


main(args.target, "recreate")
main(args.minor, "update")
main(args.source, "update")
