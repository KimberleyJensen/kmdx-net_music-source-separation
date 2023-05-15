## How to write your own models?

We recommend that you place the code for all your agents in the `my_submisison` directory (though it is not mandatory). We have added sample code that just returns the original soundfile in `identity_music_separation_model.py`

**Add your model name in** [`user_config.py`](user_config.py)

# What instruments need to be separated:

Your model needs to output 4 separate sound arrays corresponding to 'bass', 'drums', 'other', and 'vocals'

## Music Separation model format
You will have to implement a class containing the function `separate_music_file`. This will recieve sound array,  which will be the mixed music. You need to dictionary with the instruments as keys and the correponding demixed arrays as values.

The evaluator uses `MySeparationModel` from `user_config.py` as its entrypoint. Specify the class name for your model here.

## What's AIcrowd Wrapper

Don't change this file, it is used to save the outputs you predict and read the input sound files. We run it on the client side for efficiency. The AIcrowdWrapper is the actual class that is called by the evaluator, it also contains some checks to see your predictions are formatted correctly.
