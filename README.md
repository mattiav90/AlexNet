<!-- README  -->

## train the floating point alexnet ##
to train the floating point AlexNet, make sure that in the config file, the pruning variable is false, 
set the number of epochs and run.

python3 main.py 

to test the dense model that is in the google folder, type:

python3 main.py --test models/fp/0/name.pt 

## introduce sparsity in the dense alexnet model ##
once the dense model is traned we can introduce sparsity in it. 
to introduce sparsity in the dense alexnet model, set the configuration file to pruning=True,
set the desired final sparsity. let pruning_every be 2. Set the desired number of epochs (maybe 15).
run:

python3 main.py --load model/fp/0/dense_alexnet_model.pt

This will load the previously trained alexnet dense model and progressively apply pruning to it. 

to test the pruned models you can type:

python3 main.py --test models/fp/0/qat_False_bits_8_pruning_False_sparsity_0_acc_87.pt
python3 main.py --test models/fp/25/qat_False_bits_8_pruning_True_sparsity_27_acc_88.pt
python3 main.py --test models/fp/50/qat_False_bits_8_pruning_True_sparsity_52_acc_87.pt 
python3 main.py --test models/fp/75/...
python3 main.py --test models/fp/90/...

the script will load the model, count the zeros in the layers to confirm that the model is properly 
sparsified, show the calculated sparsity.
then run a test with it on the testing dataset. 



## qat trainig ##
now that we have sparse models we can do qat on them. 
go into config.py and set the desired quantization bit width, symmetrical or asymmetrical, entropy or minmax stats calculation.
you can set 8,4 or 2 bits activations and weights. 
you can also set 2 bit weights and 8 bit activations by setting to 8 the variable: activation_bit.
almost all the models that I trained and saved in the repo have been trained using symmetrical and entropy. 
the 2 bit models I used asymmetrical. 

to perform quantization aware training on a model we can type:

python3 main.py --load models/fp/25/qat_False_bits_8_pruning_True_sparsity_27_acc_88.pt --qat

this command will load the model and start performing qat on it. 
the accuracy will initially drop depending on the quantization settings, then eventually it will increase again. 




## testing qat ##
currently there is something wrong with this step. 
my steps are: 
- I load the pruned fp model
- I look at the values that are 0, and apply a pruning mask on them, to avoid them to be used in the training
- perform the training, and get the results that I report in the pdf. 
- save the model. 
- when I load the model and test it again, the accuracy dropped. 

I am not sure what is the problem during the inference phase, I am still investigating. 

to run a test on the pruned and quantized models:

python3 main.py --test models/int8/50/... --qat

the accuracy unfortunally will not reflect the one obtained during traing, I am still figuring out why. 

# quantized models that works
currently the quantized models that works are the dense one. to test them use:

python3 main.py --test models/int8/0/model.pth --qat
python3 main.py --test models/int4/0/model.pth --qat

