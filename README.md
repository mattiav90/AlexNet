<!-- README  -->

## train the floating point alexnet ##
to train the floating point AlexNet, make sure that in the config file pruning is false, 
set the number of epochs and run.

python3 main.py 

## introduce sparsity in the dense alexnet model ##
to introduce sparsity in the dense alexnet model, set the configuration file to pruning=True,
set the desired final sparsity. let pruning_every be 2. 
run:

python3 main.py --load model/dense_alexnet_model.pt

This will load the previously trained alexnet dense model and progressively apply pruning to it. 



## testing pruning ##
all the fp dense and unstructured sparsity models are in the zip file.
they are testable with 

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
as mentioned in the report, unfortunately there is something wrong in my forward pass for quantization deployment. 
during training I see good accuracy, even with both weights and activations quantized, but then, when I save the model and try to load it and test it, 
the accuracy drops significantly. 
I think there might be something inconsistent in the file quantizer_test.py. 

to run a test on the pruned and quantized models:

python3 main.py --test models/int8/50/... --qat

the accuracy unfortunally will not reflect the one obtained during traing, I am still figuring out why. 