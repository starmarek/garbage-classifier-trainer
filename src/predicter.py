import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class Predicter:
    def __init__(self, model_to_predict_on, data_generator, classes):
        self.model = model_to_predict_on
        self.data = data_generator
        self.classes = classes

    def predict_some_files(self):
        plt.figure(figsize=(10, 10))
        for (img, label) in self.data:
            batch_prediction = np.argmax(self.model.predict(img), axis=-1)
            for i in range(9):
                title_background = None
                truth_label = self.classes[np.argmax(label[i])]
                prediction_label = self.classes[batch_prediction[i]]
                plt.subplot(3, 3, i + 1)
                if truth_label != prediction_label:
                    title_background = {"facecolor": "red", "alpha": 0.5, "pad": 5}
                plt.title(
                    "truth: " + truth_label + "\nprediction: " + prediction_label,
                    bbox=title_background,
                )
                plt.axis("off")
                plt.imshow(img[0 + i, :, :, ::])
            break
        plt.tight_layout()
        plt.show()

    def evaluate_model(self):
        results = self.model.evaluate(self.data)
        logger.info(f"test loss, test acc: {results}")
