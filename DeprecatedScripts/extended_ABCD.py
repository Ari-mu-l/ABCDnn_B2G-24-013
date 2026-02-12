import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from json import loads as load_json
from json import dumps as dump_json
from argparse import ArgumentParser

import config
import abcdnn

parser = ArgumentParser()
parser.add_argument( "-s", "--source", required = True )
parser.add_argument( "-b", "--minor",  required = True )
parser.add_argument( "-t", "--target", required = True )
parser.add_argument( "--hpo", action = "store_true" )
parser.add_argument( "--randomize", action = "store_true" )
parser.add_argument( "--verbose", action = "store_true" )
parser.add_argument( "--closure", action = "store_true" )
parser.add_argument( "-m", "--modeltag", default = "best_model", help = "Name of model saved to Results directory" )
parser.add_argument( "-d", "--disc_tag", default = "ABCDnn", help = "Postfix appended to original branch names of transformed variables" )
parser.add_argument( "-f", "--fraction", default = "1.0")
args = parser.parse_args()

if args.randomize: config.params["MODEL"]["SEED"] = np.random.randint( 100000 )

abcdnn_ = abcdnn.ABCDnn_training()
abcdnn_.setup_events(
  rSource = args.source, 
  rMinor  = args.minor,
  rTarget = args.target,
  selection = config.selection,
  variables = config.variables,
  regions = config.regions,
  closure = args.closure,
  frac = float(args.fraction)
)

abcdnn_.evaluate_regions()
abcdnn_.extended_ABCD()

with open( os.path.join( "Results/", args.modeltag + ".json" ), "r+" ) as f:
    params = load_json( f.read() )

#print(params)
#params[ "TRANSFER" ] = abcdnn_.transfer
#print(params)

#with open( os.path.join( "Results/", args.modeltag + ".json" ), "w" ) as f:
#    f.write( dump_json( params, indent=2 ) )


