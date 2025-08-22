# this script is run on the condor node for applying gthe trained ABCDnn model to ttbar samples
# last updated 11/15/2021 by Daniel Li

import numpy as np
import os
#import imageio
import uproot
import abcdnn
import tqdm
from argparse import ArgumentParser
from json import loads as load_json
from array import array
import ROOT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import config

parser = ArgumentParser()
parser.add_argument( "-s", "--source", required = True )
parser.add_argument( "-t", "--target", required = True )
parser.add_argument( "-b", "--minor", required = True, help = "Minor background to subtract from data. Include with --useMinor" )
parser.add_argument( "--unblind", action = "store_true" )
parser.add_argument( "--useMinor", action = "store_true" )
parser.add_argument( "--transfer", action = "store_true" )
parser.add_argument( "-m", "--tag", required = True )

args = parser.parse_args()

highST = True
plotBest = True

folder = config.params[ "MODEL" ][ "SAVEDIR" ]
folder_contents = os.listdir( folder )

if args.transfer:    
  os.system( "eos root://cmseos.fnal.gov rm -rf /{}/{}/".format( config.targetDir[ "LPC" ].split( "//" )[2], args.tag ) )

# load in json file
print( ">> Reading in {}.json for hyper parameters...".format( args.tag ) )
with open( os.path.join( folder, args.tag + ".json" ), "r" ) as f:
  params = load_json( f.read() )

print( ">> Setting up NAF model..." )

print( ">> Loading checkpoint weights from {} with tag: {}".format( folder, args.tag ) )
checkpoints = [ name.split( ".index" )[0] for name in folder_contents if ( "EPOCH" in name and args.tag in name and name.endswith( "index" ) ) ][-2:] # TEMP

#print(checkpoints[-14:])

print( ">> Load the data" )
sFile = uproot.open( args.source )
tFile = uproot.open( args.target )
mFile = uproot.open( args.minor )
sTree = sFile[ "Events" ]
tTree = tFile[ "Events" ]
mTree = mFile[ "Events" ]

variables = [ str( key ) for key in sorted( config.variables.keys() ) if config.variables[key]["TRANSFORM"] ]
variables_transform = [ str( key ) for key in sorted( config.variables.keys() ) if config.variables[key]["TRANSFORM"] ]
if config.regions["Y"]["VARIABLE"] in config.variables and config.regions["X"]["VARIABLE"] in config.variables:
  variables.append( config.regions["Y"]["VARIABLE"] )
  variables.append( config.regions["X"]["VARIABLE"] )
else:
  sys.exit( "[ERROR] Control variables not listed in config.variables, please add. Exiting..." )

categorical = [ config.variables[ vName ][ "CATEGORICAL" ] for vName in variables ]
lowerlimit = [ config.variables[ vName ][ "LIMIT" ][0] for vName in variables ] # MIGHT NEED TO BE FIXED
upperlimit = [ config.variables[ vName ][ "LIMIT" ][1] for vName in variables ]

print( ">> Found {} variables: ".format( len( variables ) ) )
for i, variable in enumerate( variables ):
  print( "  + {}: [{},{}], Categorical = {}".format( variable, lowerlimit[i], upperlimit[i], categorical[i] ) )

#inputs_src = sTree.pandas.df( variables ) # uproot3
#inputs_src = sTree.arrays( variables + [ "Bprime_chi2" ], library="pd" ) # uproot4
#inputs_src = sTree.arrays( variables + [ "gcLeadingOSFatJet_pNetJ" ], library="pd" )
inputs_src = sTree.arrays( variables, library="pd" )

X_MIN = inputs_src[ variables[-1] ].min()
X_MAX = inputs_src[ variables[-1] ].max()
Y_MIN = inputs_src[ variables[-2] ].min()
Y_MAX = inputs_src[ variables[-2] ].max()

x_region = np.linspace( X_MIN, X_MAX, X_MAX - X_MIN + 1 )
y_region = np.linspace( Y_MIN, Y_MAX, Y_MAX - Y_MIN + 1 )

inputs_src_region = { region: None for region in [ "X", "Y", "A", "B", "C", "D" ] }
print( ">> Found {} total source entries".format( inputs_src.shape[0] ) )
#inputs_tgt = tTree.pandas.df( variables ) # uproot3
inputs_tgt = tTree.arrays( variables, library="pd" ) # uproot4 
inputs_tgt_region = { region: None for region in [ "X", "Y", "A", "B", "C", "D" ] }
print( ">> Found {} total target entries".format( inputs_tgt.shape[0] ) )

#inputs_mnr = mTree.pandas.df( variables + [ "xsecWeight" ] ) # uproot3 
inputs_mnr = mTree.arrays( variables + [ "xsecWeight" ], library="pd" ) # uproot4
inputs_mnr_region = { region: None for region in [ "X", "Y", "A", "B", "C", "D" ] }
print( ">> Found {} total minor background entries".format( inputs_mnr.shape[0] ) )

# inputs_src = (inputs_src.loc[inputs_src["Bprime_chi2"]>25]).drop(columns=["Bprime_chi2"])
# inputs_tgt = (inputs_tgt.loc[inputs_tgt["Bprime_chi2"]>25]).drop(columns=["Bprime_chi2"])
# inputs_mnr = (inputs_mnr.loc[inputs_mnr["Bprime_chi2"]>25]).drop(columns=["Bprime_chi2"])

#inputs_src = inputs_src.loc[inputs_src["gcLeadingOSFatJet_pNetJ"]>0.5]
#inputs_tgt = inputs_tgt.loc[inputs_tgt["OS1FatJetProbJ"]>0.9]
#inputs_mnr = inputs_mnr.loc[inputs_mnr["OS1FatJetProbJ"]>0.9]
#inputs_tgt = inputs_tgt.loc[inputs_tgt["gcJet_ST"]<850]
#inputs_mnr = inputs_mnr.loc[inputs_mnr["gcJet_ST"]<850]
#inputs_tgt = inputs_tgt.loc[inputs_tgt["Bprime_ptbal"]<0.85]
#inputs_mnr = inputs_mnr.loc[inputs_mnr["Bprime_ptbal"]<0.85]

for variable in variables:
  #if variable=="gcJet_ST": continue # TEMP. for validation only
  inputs_tgt[variable] = inputs_tgt[variable].clip(upper = config.variables[variable]["LIMIT"][1])
  inputs_mnr[variable] = inputs_mnr[variable].clip(upper = config.variables[variable]["LIMIT"][1])
#print(inputs_mnr["Bprime_mass"].min()
  
for region in [ "X", "Y", "A", "B", "C", "D" ]:
  if config.regions["X"][region][0] == ">=":
    select_src = inputs_src[ config.regions["X"]["VARIABLE"] ] >= config.regions[ "X" ][ region ][1]
    select_tgt = inputs_tgt[ config.regions["X"]["VARIABLE"] ] >= config.regions[ "X" ][ region ][1]
    select_mnr = inputs_mnr[ config.regions["X"]["VARIABLE"] ] >= config.regions[ "X" ][ region ][1]
  elif config.regions["X"][region][0] == "<=":
    select_src = inputs_src[ config.regions["X"]["VARIABLE"] ] <= config.regions[ "X" ][ region ][1]
    select_tgt = inputs_tgt[ config.regions["X"]["VARIABLE"] ] <= config.regions[ "X" ][ region ][1]
    select_mnr = inputs_mnr[ config.regions["X"]["VARIABLE"] ] <= config.regions[ "X" ][ region ][1]
  else:
    select_src = inputs_src[ config.regions["X"]["VARIABLE"] ] == config.regions[ "X" ][ region ][1]
    select_tgt = inputs_tgt[ config.regions["X"]["VARIABLE"] ] == config.regions[ "X" ][ region ][1]
    select_mnr = inputs_mnr[ config.regions["X"]["VARIABLE"] ] == config.regions[ "X" ][ region ][1]

  if config.regions["Y"][region][0] == ">=":
    select_src &= inputs_src[ config.regions["Y"]["VARIABLE"] ] >= config.regions[ "Y" ][ region ][1]
    select_tgt &= inputs_tgt[ config.regions["Y"]["VARIABLE"] ] >= config.regions[ "Y" ][ region ][1]
    select_mnr &= inputs_mnr[ config.regions["Y"]["VARIABLE"] ] >= config.regions[ "Y" ][ region ][1]
  elif config.regions["Y"][region][0] == "<=":
    select_src &= inputs_src[ config.regions["Y"]["VARIABLE"] ] <= config.regions[ "Y" ][ region ][1]
    select_tgt &= inputs_tgt[ config.regions["Y"]["VARIABLE"] ] <= config.regions[ "Y" ][ region ][1]
    select_mnr &= inputs_mnr[ config.regions["Y"]["VARIABLE"] ] <= config.regions[ "Y" ][ region ][1]
  else:
    select_src &= inputs_src[ config.regions["Y"]["VARIABLE"] ] == config.regions[ "Y" ][ region ][1]
    select_tgt &= inputs_tgt[ config.regions["Y"]["VARIABLE"] ] == config.regions[ "Y" ][ region ][1]
    select_mnr &= inputs_mnr[ config.regions["Y"]["VARIABLE"] ] == config.regions[ "Y" ][ region ][1]

  inputs_src_region[region] = inputs_src.loc[select_src]
  inputs_tgt_region[region] = inputs_tgt.loc[select_tgt]
  inputs_mnr_region[region] = inputs_mnr.loc[select_mnr]

# inputs_tgt_region["D"] = inputs_tgt_region["D"].where(inputs_tgt_region["D"] <= 1000, 0)
# inputs_mnr_region["D"] = inputs_mnr_region["D"].where(inputs_tgt_region["D"] <= 1000, 0)

print( ">> Yields in each region:" )
for region in inputs_src_region:
  print( "  + Region {}: Source = {}, Target = {}".format( region, inputs_src_region[region].shape[0], inputs_tgt_region[region].shape[0] ) )

# region_key = { # the row and column of ABCDXY
#   0: {
#     0: "X",
#     1: "Y"
#   },
#   1: {
#     0: "A",
#     1: "C"
#   },
#   2:{
#     0: "B",
#     1: "D"
#   }
# }
# bins = np.linspace( config.variables[ "Bprime_mass" ][ "LIMIT" ][0], config.variables[ "Bprime_mass" ][ "LIMIT" ][1], config.params[ "PLOT" ][ "NBINS" ] )
# for x in range(6):
#       for y in range(2):
#         blind = False
#         if ( not args.unblind and y==1 ):
#           if ( x == 4 or x == 5 ):
#             blind = True
#         if x % 2 == 0:
#           data_hist = np.histogram( inputs_tgt_region[ region_key[int(x/2)][y] ][ variables_transform ].to_numpy()[:,0], bins = bins, density = False )
#           print(data_hist)
# exit()
  
print( ">> Encoding and normalizing source inputs" )
inputs_enc_region = {}
encoder = {}
inputs_nrm_region = {}
inputmeans = np.hstack( [ float( mean ) for mean in params[ "INPUTMEANS" ] ] )
inputsigmas = np.hstack( [ float( sigma ) for sigma in params[ "INPUTSIGMAS" ] ] )

for region in inputs_src_region:
  encoder[region] = abcdnn.OneHotEncoder_int( categorical, lowerlimit = lowerlimit, upperlimit = upperlimit )
  inputs_enc_region[ region ] = encoder[region].encode( inputs_src_region[ region ].to_numpy( dtype = np.float32 ) )
  inputs_nrm_region[ region ] = ( inputs_enc_region[ region ] - inputmeans ) / inputsigmas 
  inputs_src_region[ region ][ variables[0] ] = inputs_src_region[ region ][ variables[0] ].clip(upper=config.variables[variables[0]]["LIMIT"][1])
  inputs_src_region[ region ][ variables[1] ] = inputs_src_region[ region ][ variables[1] ].clip(upper=config.variables[variables[1]]["LIMIT"][1])

print( ">> Processing checkpoints" )
predictions = {}
predictions_best = { region: [] for region in [ "X", "Y", "A", "B", "C", "D" ] }

if plotBest:
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

  for region in tqdm.tqdm( predictions_best ):
    NAF_predict = np.asarray( NAF.predict( np.asarray( inputs_nrm_region[ region ] )[::2] ) )
    print("NAF_predict shape: {}".format(NAF_predict.shape))
    predictions_best[ region ] = NAF_predict * inputsigmas[0:2] + inputmeans[0:2]
    #predictions_best[ region ] = predictions_best[ region ][predictions_best[ region ][:,1]<850] # TEMP validation cut

  del NAF
  
else:
  for i, checkpoint in enumerate( sorted( checkpoints ) ):
    epoch = checkpoint.split( "EPOCH" )[1]
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
    NAF.load_weights( os.path.join( folder, checkpoint ) )
  
    predictions[ int( epoch ) ] = { region: [] for region in [ "X", "Y", "A", "B", "C", "D" ] }

    for region in predictions[ int( epoch ) ]:
      NAF_predict = np.asarray( NAF.predict( np.asarray( inputs_nrm_region[ region ] )[::2] ) )
      predictions[ int( epoch ) ][ region ] = NAF_predict * inputsigmas[0:2] + inputmeans[0:2]
    x1_mean, x1_std = np.mean( predictions[ int( epoch ) ][ "D" ][:,0] ), np.std( predictions[ int( epoch ) ][ "D" ][:,0] )
    try:
      x2_mean, x2_std = np.mean( predictions[ int( epoch ) ][ "D" ][:,1] ), np.std( predictions[ int( epoch ) ][ "D" ][:,1] )
      print( "  + {}: {} = {:.3f} pm {:.3f}, {} = {:.3f} pm {:.3f}".format( 
        checkpoint, variables_transform[0], x1_mean, x1_std, variables_transform[1], x2_mean, x2_std ) )
    except:
      print( "  + {}: {} = {:.3f} pm {:.3f}".format(checkpoint, variables_transform[0], x1_mean, x1_std ) )
    del NAF


print( ">> Generating images of plots" )
region_key = { # the row and column of ABCDXY
  0: {
    0: "X",
    1: "Y"
  },
  1: {
    0: "A",
    1: "C" 
  },
  2:{
    0: "B",
    1: "D"
  }
}
images = { variable: [] for variable in variables_transform }

def ratio_err( x, xerr, y, yerr ):
  return np.sqrt( ( yerr * x / y**2 )**2 + ( xerr / y )**2 )

def plot_hist( ax, variable, x, y, epoch, mc_pred, mc_true, mc_minor, weights_minor, data, bins, useMinor, blind, i ):
  # log to original
  mc_pred  = np.exp(mc_pred)
  mc_true  = np.exp(mc_true)
  mc_minor = np.exp(mc_minor)
  data     = np.exp(data)
  # validation cut
  if highST:
    mc_pred       = mc_pred[mc_pred[:,1]>850]
    mc_true       = mc_true[mc_true[:,1]>850]
    weights_minor = weights_minor[mc_minor[:,1]>850] # has to go before making changes to mc_minor
    mc_minor      = mc_minor[mc_minor[:,1]>850]
    data          = data[data[:,1]>850]
  else:
    mc_pred       = mc_pred[mc_pred[:,1]<850]
    mc_true       = mc_true[mc_true[:,1]<850]
    weights_minor = weights_minor[mc_minor[:,1]<850]
    mc_minor      = mc_minor[mc_minor[:,1]<850]
    data          = data[data[:,1]<850]

  # select variable to plot
  mc_pred       = mc_pred[:,i]
  mc_true       = mc_true[:,i]
  mc_minor      = mc_minor[:,i]
  data          = data[:,i]

  # put tail in one bin
  mc_minor = np.clip(mc_minor, bins[0], bins[-1])
  data     = np.clip(data, bins[0], bins[-1])

  mc_pred_hist = np.histogram( np.clip( mc_pred, bins[0], bins[-1] ), bins = bins, density = False )
  mc_pred_scale = len(mc_pred)

  mc_true_hist = np.histogram( mc_true, bins = bins, density = False )
  mc_true_scale = len(mc_true)
  
  data_hist = np.histogram( data, bins = bins, density = False )

  mc_minor_hist = np.histogram( mc_minor, bins = bins, weights = weights_minor, density = False )
  mc_minor_scale = len(mc_minor)
  
  if useMinor:
    data_mod = data_hist[0] - mc_minor_hist[0]
  else:
    data_mod = data_hist[0]
  for i in range( len( data_mod ) ):
    if data_mod[i] < 0: data_mod[i] = 0
  data_mod_scale = float( np.sum(data_mod) )

  # plot the data first
  print("region: {}, blind: {}".format(region_key[x][y], blind))
  if not blind:
    ax.errorbar(
      0.5 * ( data_hist[1][1:] + data_hist[1][:-1] ),
      data_mod / data_mod_scale, yerr = np.sqrt( data_mod ) / data_mod_scale,
      label = "Target - Minor" if args.useMinor else "Target",
      marker = "o", markersize = 3, markerfacecolor = "black", markeredgecolor = "black",
      elinewidth = 1, ecolor = "black" , capsize = 2, lw = 0 
    )
  # plot the mc
  ax.errorbar(
    0.5 * ( mc_true_hist[1][1:] + mc_true_hist[1][:-1] ),
    mc_true_hist[0] / mc_true_scale, yerr = np.sqrt( mc_true_hist[0] ) / mc_true_scale,
    label = "Source",
    marker = ",", drawstyle = "steps-mid", lw = 2, alpha = 0.7, color = "green"
  )
  # plot the predicted
  ax.fill_between(
    0.5 * ( mc_pred_hist[1][1:] + mc_pred_hist[1][:-1] ),
    y2 = np.zeros( len( mc_pred_hist[0] ) ),
    y1 = mc_pred_hist[0] / mc_pred_scale, step = "mid",
    label = "ABCDnn",
    color = "red", alpha = 0.5,
  )
  ax.fill_between(
    0.5 * ( mc_pred_hist[1][1:] + mc_pred_hist[1][:-1] ),
    y1 = ( mc_pred_hist[0] + np.sqrt( mc_pred_hist[0] ) ) / mc_pred_scale,
    y2 = ( mc_pred_hist[0] - np.sqrt( mc_pred_hist[0] ) ) / mc_pred_scale,
    interpolate = False, step = "mid",
    color = "red", alpha = 0.2
  )
  
  ax.set_xlim( config.variables[ variable ][ "LIMIT_plot" ][0], config.variables[ variable ][ "LIMIT_plot" ][1] )
  y_max = max( [ max( data_mod ) / data_mod_scale, max( mc_true_hist[0] ) / mc_true_scale ] )
  ax.set_ylim( 0, 1.4 * y_max )
  ax.set_yscale( config.params[ "PLOT" ][ "YSCALE" ] )
  ax.axes.yaxis.set_visible(False)
  ax.axes.xaxis.set_visible(False)

  region = region_key[x][y]
  title_text = "$"
  if config.regions[ "X" ][ region ][0] == ">=":
    title_text += "{}\geq {}, ".format( config.variables[ config.regions[ "X" ][ "VARIABLE" ] ][ "LATEX" ], int(config.regions[ "X" ][ region ][1]) )
  elif config.regions[ "X" ][ region ][0] == "<=":
    title_text += "{}\leq {}, ".format( config.variables[ config.regions[ "X" ][ "VARIABLE" ] ][ "LATEX" ], int(config.regions[ "X" ][ region ][1]) )
  else:
    title_text += "{}={}, ".format( config.variables[ config.regions[ "X" ][ "VARIABLE" ] ][ "LATEX" ], int(config.regions[ "X" ][ region ][1]) )
    
  if config.regions[ "Y" ][ region ][0] == ">=":
    title_text += "{}\geq {}".format( config.variables[ config.regions[ "Y" ][ "VARIABLE" ] ][ "LATEX" ], int(config.regions[ "Y" ][ region ][1]) )
  elif config.regions[ "Y" ][ region ][0] == "<=":
    title_text += "{}\leq {}".format( config.variables[ config.regions[ "Y" ][ "VARIABLE" ] ][ "LATEX" ], int(config.regions[ "Y" ][ region ][1]) )
  else:
    title_text += "{}={}".format( config.variables[ config.regions[ "Y" ][ "VARIABLE" ] ][ "LATEX" ], int(config.regions[ "Y" ][ region ][1]) )
  title_text += "$"
  #ax.set_title( "Region {}: {}".format( region, title_text ), ha = "right", x = 1.0, fontsize = 10 )
  #print("Region {}: {}".format( region, title_text ))
  ax.text(
    0.02, 0.95, f'Region {region}',
    ha = "left", va = "top", transform = ax.transAxes, fontsize = 12
  )
  ax.legend( loc = "upper right", ncol = 2, fontsize = 10 )

def plot_ratio( ax, variable, x, y, mc_pred, mc_true, mc_minor, weights_minor, data, bins, useMinor, blind, i ):
  # log to original
  mc_pred  = np.exp(mc_pred)
  mc_true  = np.exp(mc_true)
  mc_minor = np.exp(mc_minor)
  data     = np.exp(data)
  # validation cut
  if highST:
    mc_pred       = mc_pred[mc_pred[:,1]>850]
    mc_true       = mc_true[mc_true[:,1]>850]
    weights_minor = weights_minor[mc_minor[:,1]>850]
    mc_minor      = mc_minor[mc_minor[:,1]>850]
    data          = data[data[:,1]>850]
  else:
    mc_pred       = mc_pred[mc_pred[:,1]<850]
    mc_true       = mc_true[mc_true[:,1]<850]
    weights_minor = weights_minor[mc_minor[:,1]<850]
    mc_minor      = mc_minor[mc_minor[:,1]<850]
    data          = data[data[:,1]<850]
  # select variable to plot
  mc_pred       = mc_pred[:,i]
  mc_true       = mc_true[:,i]
  mc_minor      = mc_minor[:,i]
  data          = data[:,i]
  # put tail in one bin
  mc_minor = np.clip(mc_minor, bins[0], bins[-1])
  data     = np.clip(data, bins[0], bins[-1])
  
  mc_pred_hist = np.histogram( np.clip( mc_pred, bins[0], bins[-1] ), bins = bins, density = False  )
  mc_pred_scale = float( len(mc_pred) )
  
  mc_true_hist = np.histogram( mc_true, bins = bins, density = False )
  mc_true_scale = float( len(mc_true) )

  mc_minor_hist = np.histogram( mc_minor, bins = bins, weights = weights_minor, density = False )
  mc_minor_hist = np.histogram( mc_minor, bins = bins, density = False )
  mc_minor_scale = float( len(mc_minor) )

  data_hist = np.histogram( data, bins = bins, density = False )
  if useMinor:
    data_mod = data_hist[0] - mc_minor_hist[0]
  else:
    data_mod = data_hist[0]
  data_mod_scale = np.sum(data_mod)
  
  ratio = []
  ratio_std = []
  for i in range( len( data_hist[0] ) ):
    if data_mod[i] == 0 or mc_pred_hist[0][i] == 0: 
      ratio.append(0)
      ratio_std.append(0)
    else:
      ratio.append( ( data_mod[i] / float( data_mod_scale ) /  ( mc_pred_hist[0][i] / float( mc_pred_scale ) ) ) )
      ratio_std.append( ratio_err(
        mc_pred_hist[0][i],
        np.sqrt( mc_pred_hist[0][i] ),
        data_mod[i],
        np.sqrt( data_mod[i] )
      ) * ( data_mod_scale / mc_pred_scale ) )
  if not blind:
    ax.scatter(
        0.5 * ( data_hist[1][1:] + data_hist[1][:-1] ),
      ratio, 
      linewidth = 0, marker = "o", 
      color = "black", zorder = 3
    )
    ax.fill_between(
      0.5 * ( data_hist[1][1:] + data_hist[1][:-1] ),
      y1 = np.array( ratio ) + np.array( ratio_std ),
      y2 = np.array( ratio ) - np.array( ratio_std ),
      interpolate = False, step = "mid",
      color = "gray", alpha = 0.2
    )
  ax.axhline(
    y = 1.0, color = "r", linestyle = "-", zorder = 1
  )

  ax.set_xlabel( "${}$".format( config.variables[ variable ][ "LATEX" ] ), ha = "right", x = 1.0, fontsize = 10 )
  ax.set_xlim( config.variables[ variable ][ "LIMIT_plot" ][0], config.variables[ variable ][ "LIMIT_plot" ][1] )
  ax.set_ylabel( "Target/ABCDnn", ha = "right", y = 1.0, fontsize = 8 )
  ax.set_ylim( config.params[ "PLOT" ][ "RATIO" ][0], config.params[ "PLOT" ][ "RATIO" ][1] )
  ax.set_yticks( [ 0.80, 0.90, 1.0, 1.10, 1.20 ] )
  ax.tick_params( axis = "both", labelsize = 8 )
  if x != 2: ax.axes.xaxis.set_visible(False)

  # print(data_mod[i] / float( data_mod_scale ))
  # print( mc_pred_hist[0][i] / float( mc_pred_scale ) )
  print(ratio)

if plotBest:
  os.system( "mkdir -vp {}/{}".format( folder, args.tag ) )
  print( "Plotting best trained model" )
  for i, variable in enumerate( variables_transform ):
    #bins = np.linspace( config.variables[ variable ][ "LIMIT" ][0], config.variables[ variable ][ "LIMIT" ][1], config.params[ "PLOT" ][ "NBINS" ] )
    if variable == "OS1FatJetProbJ":
      #bins = np.linspace( config.variables[ variable ][ "LIMIT" ][0], config.variables[ variable ][ "LIMIT" ][1], 3 ) # bug suspected
      bins = np.array([0,0.9,1])
    elif variable == "gcJet_ST":
      bins = np.array([config.variables[ variable ][ "LIMIT" ][0],850,config.variables[ variable ][ "LIMIT" ][1]])
    elif variable == "Bprime_ptbal":
      bins = np.array([config.variables[ variable ][ "LIMIT" ][0],0.85,config.variables[ variable ][ "LIMIT" ][1]])
    else:
      bins = np.linspace( config.variables[ variable ][ "LIMIT_plot" ][0], config.variables[ variable ][ "LIMIT_plot" ][1], 26)
    #  bins = np.linspace( config.variables[ variable ][ "LIMIT" ][0], 5000, config.params[ "PLOT" ][ "NBINS" ] )
    #  bins = np.concatenate((bins[bins<2500], np.array([2750, 3750, 5000])), axis=0)
    #else:
    #  bins = np.linspace( config.variables[ variable ][ "LIMIT" ][0], config.variables[ variable ][ "LIMIT" ][1], config.params[ "PLOT" ][ "NBINS" ] )
    fig, axs = plt.subplots( 6, 2, figsize = (9,12), gridspec_kw = { "height_ratios": [3,1,3,1,3,1] } )
  
    for x in range(6):
      for y in range(2):
        blind = False
        if not args.unblind:
          if region_key[int(x/2)][y]=="D" and highST:
            blind = True
        if x % 2 == 0:
          plot_hist(
            ax = axs[x,y],
            variable = variable,
            x = int( x / 2 ), y = y,
            epoch = "BEST",
            mc_pred = np.asarray( predictions_best[ region_key[ int(x/2) ][y] ] ),
            mc_true = inputs_src_region[ region_key[int(x/2)][y] ][ variables_transform ].to_numpy(),
            mc_minor = inputs_mnr_region[ region_key[int(x/2)][y] ][ variables_transform ].to_numpy(),
            weights_minor = inputs_mnr_region[ region_key[int(x/2)][y] ][ [ "xsecWeight" ] ].to_numpy()[:,0],
            data = inputs_tgt_region[ region_key[int(x/2)][y] ][ variables_transform ].to_numpy(),
            bins = bins,
            useMinor = args.useMinor,
            blind = blind,
            i = i
          )
        else:
          plot_ratio(
            ax = axs[x,y],
            variable = variable,
            x = int((x-1)/2), y = y,
            mc_pred = np.asarray( predictions_best[ region_key[ int((x-1)/2) ][y] ] ),
            mc_true = inputs_src_region[ region_key[int((x-1)/2)][y] ][ variables_transform ].to_numpy(),
            mc_minor = inputs_mnr_region[ region_key[int((x-1)/2)][y] ][ variables_transform ].to_numpy(),
            weights_minor = inputs_mnr_region[ region_key[int((x-1)/2)][y] ][ [ "xsecWeight" ] ].to_numpy()[:,0],
            data = inputs_tgt_region[ region_key[int((x-1)/2)][y] ][ variables_transform ].to_numpy(),
            bins = bins,
            useMinor = args.useMinor,
            blind = blind,
            i = i
          )
        if(x!=0):
          position_old = axs[x,y].get_position()
          position_new = axs[x-1,y].get_position()
          points_old = position_old.get_points()
          points_new = position_new.get_points()
          points_old[1][1] = points_new[0][1]
          position_old.set_points( points_old )
          axs[x,y].set_position( position_old )
      
    plt.savefig( "{}/{}/{}_{}.png".format( folder, args.tag, args.tag, variable ) )  
    plt.close()
else:
  print( "Plotting models per epoch:" )
  for epoch in sorted( predictions.keys() ):
    print( "  + Generating image for epoch {}".format( epoch ) )
    for i, variable in enumerate( variables_transform ): 
      bins = np.linspace( config.variables[ variable ][ "LIMIT_plot" ][0], config.variables[ variable ][ "LIMIT_plot" ][1], config.params[ "PLOT" ][ "NBINS" ] )
      fig, axs = plt.subplots( 6, 2, figsize = (9,12), gridspec_kw = { "height_ratios": [3,1,3,1,3,1] } )
      for x in range( 6 ):
        for y in range( 2 ):
          blind = False
          if ( not args.unblind and y==1 ):
            if ( x == 4 or x == 5 ):
              blind = False
          if x % 2 == 0: 
            plot_hist( 
              ax = axs[x,y], 
              variable = variable,
              x = int( x / 2 ), y = y,
              epoch = epoch,
              mc_pred = np.asarray( predictions[ epoch ][ region_key[int(x/2)][y] ] ),
              mc_true = inputs_src_region[ region_key[int(x/2)][y] ][ variables_transform ].to_numpy(),
              mc_minor = inputs_mnr_region[ region_key[int(x/2)][y] ][ variables_transform ].to_numpy(),
              weights_minor = inputs_mnr_region[ region_key[int((x)/2)][y] ][ [ "xsecWeight" ] ].to_numpy()[:,0],
              data = inputs_tgt_region[ region_key[int(x/2)][y] ][ variables_transform ].to_numpy(),
              bins = bins,
              useMinor = args.useMinor,
              blind = blind,
              i = i
            )
          else: 
            plot_ratio( 
              ax = axs[x,y],
              variable = variable,
              x = int((x-1)/2), y = y, 
              mc_pred = np.asarray( predictions[ epoch ][ region_key[int((x-1)/2)][y] ] ),
              mc_true = inputs_src_region[ region_key[int((x-1)/2)][y] ][ variables_transform ].to_numpy(),
              mc_minor = inputs_mnr_region[ region_key[int((x-1)/2)][y] ][ variables_transform ].to_numpy(),
              weights_minor = inputs_mnr_region[ region_key[int((x-1)/2)][y] ][ [ "xsecWeight" ] ].to_numpy()[:,0],
              data = inputs_tgt_region[ region_key[int((x-1)/2)][y] ][ variables_transform ].to_numpy(),
              bins = bins,
              useMinor = args.useMinor,
              blind = blind,
              i = i
            )
          if(x!=0):
            position_old = axs[x,y].get_position()
            position_new = axs[x-1,y].get_position()
            points_old = position_old.get_points()
            points_new = position_new.get_points()
            points_old[1][1] = points_new[0][1]
            position_old.set_points( points_old )
            axs[x,y].set_position( position_old )

    plt.savefig( "{}/{}/{}_{}_EPOCH{}.png".format( folder, args.tag, args.tag, variable, epoch ) )
    #images[ variable ].append( imageio.imread( "{}/{}/{}_{}_EPOCH{}.png".format( folder, args.tag, args.tag, variable, epoch ) ) )
    plt.close()
exit()

try:
  print( "[DONE] {} training GIF completed: {}_{}.gif, {}_{}.gif".format( args.tag, args.tag, variables_transform[0], args.tag, variables_transform[1] ) )
except:
  print( "[DONE] {} training GIF completed: {}_{}.gif".format( args.tag, args.tag, variables_transform[0] ) )
for variable in variables_transform:
  imageio.mimsave( "{}/{}/{}_{}.gif".format( folder, args.tag, args.tag, variable ), images[ variable ], duration = 1 )
  if args.transfer:
    os.system( "xrdcp -rfp {}/{}/ {}/{}/".format( folder, args.tag, config.targetDir[ "LPC" ], args.tag ) )
