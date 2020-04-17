import keras.applications as classifiers

class VGG19:
    def build_network(self, input_shape, num_classes):
        model = classifiers.vgg16.VGG16(include_top=True,
                                        weights=None,
                                        input_tensor=None,
                                        input_shape=input_shape,
                                        pooling='max',
                                        classes=num_classes)
        return model