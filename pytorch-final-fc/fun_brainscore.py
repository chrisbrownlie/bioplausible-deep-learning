import functools
import torch
import datetime
from fun_cifar10 import test_CIFAR10
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment

def test_scores(model_def, model_name, layers_to_consider = ['fc1', 'fc2'], fun_trained = True, benchmarks = ['v1', 'v2', 'v4', 'it', 'bhv'], debug=False, meulemans=False):
    """
    Test a trained model on CIFAR10 data, and then try to obtain a score on the public brainscore IT benchmark

    model_name = the identifier which was used to train the model
    empty_model = an initialised model object e.g. myModel()
    layers_to_consider = list indicating names of layers in the model to consider when scoring
    fun_trained -- whether it was trained using the train_CIFAR10 function, if this is false, the model supplied to model_def will be pretrained
    """
    if fun_trained:
        test_accuracy = test_CIFAR10(model_name = model_name, model = model_def, meulemans=meulemans)
        
        # Construct model path from name
        model_def.load_state_dict(torch.load('./models/' + model_name + '.pth'))

        print(model_def)
        print("Trained")

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

    # Obtain as many test scores as possible
    # IT Score
    if 'it' in benchmarks:
        if debug:
            public_IT_score = score_model(
                model_identifier = brain_model.identifier,
                model = brain_model,
                benchmark_identifier='dicarlo.MajajHong2015public.IT-pls'
            )
            final_IT_score = round(public_IT_score[0].item(), 5)
        else:
            try:             
                public_IT_score = score_model(
                    model_identifier = brain_model.identifier,
                    model = brain_model,
                    benchmark_identifier='dicarlo.MajajHong2015public.IT-pls'
                )
                final_IT_score = round(public_IT_score[0].item(), 5)
            except:
                final_IT_score = 'error'
    else:
        final_IT_score = 'skipped'
        print('Skipping IT benchmark...')

    # V4 score
    if 'v4' in benchmarks:
        if debug:
            public_V4_score = score_model(
                model_identifier = brain_model.identifier,
                model = brain_model,
                benchmark_identifier='dicarlo.MajajHong2015public.V4-pls'
            )
            final_V4_score = round(public_V4_score[0].item(), 5)
        else:
            try:             
                public_V4_score = score_model(
                    model_identifier = brain_model.identifier,
                    model = brain_model,
                    benchmark_identifier='dicarlo.MajajHong2015public.V4-pls'
                )
                final_V4_score = round(public_V4_score[0].item(), 5)
            except:
                final_V4_score = 'error'
    else:
        final_V4_score = 'skipped'
        print('Skipping V4 benchmark...')

    if 'v2' in benchmarks:
        if debug:
            public_V2_score = score_model(
                model_identifier = brain_model.identifier,
                model = brain_model,
                benchmark_identifier='movshon.FreemanZiemba2013public.V2-pls'
            )
            final_V2_score = round(public_V2_score[0].item(), 5)
            print("Success - final V2 score is " + str(final_V2_score))
        else:
            try:             
                public_V2_score = score_model(
                    model_identifier = brain_model.identifier,
                    model = brain_model,
                    benchmark_identifier='movshon.FreemanZiemba2013public.V2-pls'
                )
                final_V2_score = round(public_V2_score[0].item(), 5)
            except:
                final_V2_score = 'error'
    else:
        final_V2_score = 'skipped'
        print('Skipping V2 benchmarks...')

    if 'v1' in benchmarks:
        if debug:
            public_V1_score = score_model(
                model_identifier = brain_model.identifier,
                model = brain_model,
                benchmark_identifier='movshon.FreemanZiemba2013public.V1-pls'
            )
            final_V1_score = round(public_V1_score[0].item(), 5)
            print("Success - final V1 score is " + str(final_V1_score))
        else:
            try:             
                public_V1_score = score_model(
                    model_identifier = brain_model.identifier,
                    model = brain_model,
                    benchmark_identifier='movshon.FreemanZiemba2013public.V1-pls'
                )
                final_V1_score = round(public_V1_score[0].item(), 5)
            except:
                final_V1_score = 'error'
    else:
        final_V1_score = 'skipped'
        print('Skipping V1 benchmark...')

    if 'bhv' in benchmarks:
        if debug:
            public_behaviour_score = score_model(
                model_identifier = brain_model.identifier,
                model = brain_model,
                benchmark_identifier='dicarlo.Rajalingham2018public-i2n'
            )
            final_behaviour_score = round(public_behaviour_score[0].item(), 5)
        else:
            try:             
                public_behaviour_score = score_model(
                    model_identifier = brain_model.identifier,
                    model = brain_model,
                    benchmark_identifier='dicarlo.Rajalingham2018public-i2n'
                )
                final_behaviour_score = round(public_behaviour_score[0].item(), 5)
            except:
                final_behaviour_score = 'error'
    else:
        final_behaviour_score = 'skipped'
        print('Skipping behavioural benchmark...')

    # Log result of scoring
    with open('model_test_scores_v2.csv','a') as log:
        log.write('\n' + str(datetime.datetime.now()) + ',' + model_name + ',' + str(test_accuracy) + ',' + str(final_V1_score) + ',' + str(final_V2_score) + ',' + str(final_V4_score) + ',' + str(final_IT_score) + ',' + str(final_behaviour_score))

    errors = [final_V1_score, final_V2_score, final_V4_score, final_IT_score, final_behaviour_score].count('error')
    skipped = [final_V1_score, final_V2_score, final_V4_score, final_IT_score, final_behaviour_score].count('skipped')
    attempted = 5-skipped
    successful = 5-(skipped+errors)
    print("Successful benchmarks: " + str(successful) + "/" + str(attempted))

    # Return result
    return test_accuracy