import imageclassification.training.models as mdls


def get_model(my_net, num_classes):
    if 'vggbndrop' in my_net:
        net = mdls.VGGBNDrop(num_classes=num_classes, init_weights=True)
    elif 'vgg' in my_net:
        net = mdls.VGG(num_classes=num_classes, init_weights=True)
    else:
        raise NotImplementedError
    return net
