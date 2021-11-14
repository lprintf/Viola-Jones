from typing import Iterable, NamedTuple
from .feature import Feature
from .feature2h import Feature2h
from .feature2v import Feature2v
from .feature3h import Feature3h
from .feature3v import Feature3v
from .feature4 import Feature4
from config import WINDOW_SIZE

Size = NamedTuple(
    "Size", [("height", int), ("width", int)]
)
Location = NamedTuple(
    "Location", [("top", int), ("left", int)]
)


def possible_position(
    size: int, window_size: int
) -> Iterable[int]:
    return range(0, window_size - size + 1)


def possible_locations(
    base_shape: Size, window_size: int,
) -> Iterable[Location]:
    return (
        Location(left=x, top=y)
        for x in possible_position(
            base_shape.width, window_size
        )
        for y in possible_position(
            base_shape.height, window_size
        )
    )


def possible_shapes(
    base_shape: Size, window_size: int,
) -> Iterable[Size]:
    base_height = base_shape.height
    base_width = base_shape.width
    return (
        Size(height=height, width=width)
        for width in range(
            base_width,
            window_size + 1,
            base_width,
        )
        for height in range(
            base_height,
            window_size + 1,
            base_height,
        )
    )


def generate_features(
    if_3v=True, size=WINDOW_SIZE
) -> list:
    WINDOW_SIZE = size
    feature2h = list(
        Feature2h(
            location.left,
            location.top,
            shape.width,
            shape.height,
        )
        for shape in possible_shapes(
            Size(height=1, width=2), WINDOW_SIZE
        )
        for location in possible_locations(
            shape, WINDOW_SIZE
        )
    )

    feature2v = list(
        Feature2v(
            location.left,
            location.top,
            shape.width,
            shape.height,
        )
        for shape in possible_shapes(
            Size(height=2, width=1), WINDOW_SIZE
        )
        for location in possible_locations(
            shape, WINDOW_SIZE
        )
    )

    feature3h = list(
        Feature3h(
            location.left,
            location.top,
            shape.width,
            shape.height,
        )
        for shape in possible_shapes(
            Size(height=1, width=3), WINDOW_SIZE
        )
        for location in possible_locations(
            shape, WINDOW_SIZE
        )
    )
    if if_3v:
        feature3v = list(
            Feature3v(
                location.left,
                location.top,
                shape.width,
                shape.height,
            )
            for shape in possible_shapes(
                Size(height=3, width=1), WINDOW_SIZE
            )
            for location in possible_locations(
                shape, WINDOW_SIZE
            )
        )
    else:
        feature3v=[]

    feature4 = list(
        Feature4(
            location.left,
            location.top,
            shape.width,
            shape.height,
        )
        for shape in possible_shapes(
            Size(height=2, width=2), WINDOW_SIZE
        )
        for location in possible_locations(
            shape, WINDOW_SIZE
        )
    )

    features = (
        feature2h
        + feature2v
        + feature3h
        + feature3v
        + feature4
    )

    return features


__all__ = [
    "Feature",
    "Feature2h",
    "Feature2v",
    "Feature3h",
    "Feature3v",
    "Feature4",
    "generate_features",
]
