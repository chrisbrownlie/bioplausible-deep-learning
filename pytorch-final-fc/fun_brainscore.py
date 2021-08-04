import functools
import torch
import datetime
from fun_cifar10 import test_CIFAR10
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment

def test_score(model_def, model_name, layers_to_consider = ['fc1', 'fc2'], fun_trained = True):
    """
    Test a trained model on CIFAR10 data, and then try to obtain a score on the public brainscore IT benchmark

    model_name = the identifier which was used to train the model
    empty_model = an initialised model object e.g. myModel()
    layers_to_consider = list indicating names of layers in the model to consider when scoring
    fun_trained -- whether it was trained using the train_CIFAR10 function, if this is false, the model supplied to model_def will be pretrained
    """
    if fun_trained:
        test_accuracy = test_CIFAR10(model_name = model_name, model = model_def)
        
        # Construct model path from name
        model_def.load_state_dict(torch.load('./models/' + model_name + '.pth'))

    # Define the preprocessing to be used with image size 32x32 (as this is what cifar10 is)
    preprocessing = functools.partial(load_preprocess_images,image_size=32)

    # Convert model to an activations model with PytorchWrapper
    activations_model = PytorchWrapper(
        identifier=model_name, 
        model=model_def, 
        preprocessing=preprocessing
    )


    # Convert activations model to a brain interface model that can be scored by brain score
    brain_model = ModelCommitment(
        identifier = model_name, 
        activations_model = activations_model,
        layers = layers_to_consider
    )

    # Try to obtain test score               
    public_IT_score = score_model(
        model_identifier = brain_model.identifier,
        model = brain_model,
        benchmark_identifier='dicarlo.MajajHong2015public.IT-pls'
    )

    final_score = round(public_IT_score[0].item(), 5)

    # Log result of scoring
    with open('model_test_scores.csv','a') as log:
        log.write('\n' + str(datetime.datetime.now()) + ',' + model_name + ',' + str(test_accuracy) + ',' + str(final_score))

    print("Brain score for public benchmark: " + str(final_score))

    # Return result
    return final_score