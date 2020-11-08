import logging
import math

import matplotlib.pyplot as plt
import numpy as np

import src.utils.config as cnf

logger = logging.getLogger(__name__)


class Predicter:
    def __init__(self, model_to_predict_on, data):
        logger.info(f"Creating {type(self).__name__} class")

        self.model = model_to_predict_on
        self.data = data
        self.batch_size = cnf.config.post_train.batch_size
        self.classes = cnf.config.post_train.classes
        # following method -> (img + 1) * 127.5 is the reverse process
        # of preprocess_input func. used in DataLoader. This is needed,
        # because data is in [-1, 1] range. pyplot.imshow would clip the
        # range to [0, 1] which would impact pictures quality.
        # preprocess_input func. source:
        # https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
        self.invert_tf_preprocess_input_rescale = lambda img: (
            (img + 1) * 127.5
        ).astype(np.uint8)

    def predict_some_files(self, number_of_pictures_to_predict):
        assert self.batch_size >= number_of_pictures_to_predict, (
            "You can only predict as much files as a single "
            "batch has to offer. This is a design restriction."
        )

        def generate_sublots_size():
            sqrt = math.sqrt(number_of_pictures_to_predict)
            if isinstance(sqrt, int):
                return sqrt, sqrt
            elif round(sqrt) == math.ceil(sqrt):
                return math.ceil(sqrt), math.ceil(sqrt)
            else:
                return math.ceil(sqrt), round(sqrt)

        size_1, size_2 = generate_sublots_size()
        plt.figure(figsize=(15, 15))
        for (img, label) in self.data:
            batch_prediction = np.argmax(self.model.predict(img), axis=-1)
            for i in range(number_of_pictures_to_predict):
                title_background = None
                truth_label = self.classes[np.argmax(label[i])]
                prediction_label = self.classes[batch_prediction[i]]
                plt.subplot(size_1, size_2, i + 1)
                if truth_label != prediction_label:
                    title_background = {"facecolor": "red", "alpha": 0.5, "pad": 5}
                plt.title(
                    "truth: " + truth_label + "\nprediction: " + prediction_label,
                    bbox=title_background,
                )
                plt.axis("off")
                plt.imshow(self.invert_tf_preprocess_input_rescale(img[0 + i, :, :, :]))
            break
        plt.tight_layout()
        plt.show()

    def predict_single_file(self):
        prediction = np.argmax(self.model.predict(self.data))
        plt.figure()
        prediction_label = self.classes[prediction]
        plt.title("prediction: " + prediction_label)
        plt.axis("off")
        plt.imshow(self.invert_tf_preprocess_input_rescale(self.data[0]))
        plt.show()

    def evaluate_model(self):
        logger.info("Starting evaluation")
        results = self.model.evaluate(self.data)
        logger.info(f"test loss, test acc: {results}")
