from ..tresnet import TResnetM


def create_model(args, num_classes):
    """
    Create a model, with model_name and num_classes
    """
    model_params = {'args': args, 'num_classes': num_classes}
    model_name = args.model_name.lower()

    assert model_name=='tresnet_m'
    model = TResnetM(model_params)

    return model
