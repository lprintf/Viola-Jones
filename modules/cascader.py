import numpy as np
from config import WINDOW_SIZE
from .adaboost import (
    strong_classifier,
    build_weak_classifiers,
)
from .feature import generate_features
from .integral_image import to_integral
from .dataloader import Dataloader, gleam
from .feature import *


class Cascader:
    def __init__(
        self, strong_classifier_num: int = 3,
    ) -> None:
        self.weak_classifiers=None
        self.weak_classifiers_2=None
        self.weak_classifiers_3=None
        pass

    def train(
        self,
        dataloader: Dataloader, # todo应是抽象类
        if_3v: bool = True,
        window_size: list = [15],
    ):
        xs, ys = dataloader.load(500,1000)
        xis = np.array(
            [to_integral(x) for x in xs]
        )
        features = generate_features(if_3v)
        self.weak_classifiers,_ = build_weak_classifiers(
            "1st", 2, xis, ys, features
        )
        xs, ys = dataloader.load(500,1000)
        self.weak_classifiers_2,_ = build_weak_classifiers(
            "1st", 10, xis, ys, features
        )
        xs, ys = dataloader.load(500,1000)
        self.weak_classifiers_3,_ = build_weak_classifiers(
            "1st", 20, xis, ys, features
        )

    def load_classifiers(self):
        pass

    def test(self, target_image: np.ndarray):
        weak_classifiers = self.weak_classifiers
        weak_classifiers_2 = (
            self.weak_classifiers_2
        )
        weak_classifiers_3 = (
            self.weak_classifiers_3
        )
        grayscale = gleam(target_image)
        integral = to_integral(grayscale)
        rows, cols = integral.shape[0:2]
        HALF_WINDOW = WINDOW_SIZE // 2
        face_positions_1 = []
        face_positions_2 = []
        face_positions_3 = []
        # todo 待讨论，补偿关照差异
        normalized_integral = integral
        # normalized_integral = to_integral(
        #     normalize(grayscale)
        # )

        for row in range(
            HALF_WINDOW + 1, rows - HALF_WINDOW
        ):
            for col in range(
                HALF_WINDOW + 1,
                cols - HALF_WINDOW,
            ):
                window = normalized_integral[
                    row
                    - HALF_WINDOW
                    - 1 : row
                    + HALF_WINDOW
                    + 1,
                    col
                    - HALF_WINDOW
                    - 1 : col
                    + HALF_WINDOW
                    + 1,
                ]

                # First cascade stage
                probably_face = strong_classifier(
                    window, weak_classifiers
                )
                if probably_face < 0.5:
                    continue
                face_positions_1.append(
                    (row, col)
                )

                # Second cascade stage
                probably_face = strong_classifier(
                    window, weak_classifiers_2
                )
                if probably_face < 0.5:
                    continue
                face_positions_2.append(
                    (row, col)
                )

                # Third cascade stage
                probably_face = strong_classifier(
                    window, weak_classifiers_3
                )
                if probably_face < 0.5:
                    continue
                face_positions_3.append(
                    (row, col)
                )

        print(
            f"Found {len(face_positions_1)} candidates at stage 1, {len(face_positions_2)} at stage 2 and {len(face_positions_3)} at stage 3."
        )
