from pathlib import Path

from PIL import Image

from medchange.data.nih.image_resolver import (
    NIHImageResolver,
)


def test_nih_image_resolver(
    tmp_path: Path,
):
    folder = (
        tmp_path
        / "images"
        / "batch01"
    )

    folder.mkdir(
        parents=True
    )

    image_path = (
        folder
        / "00000001_000.png"
    )

    Image.new(
        "L",
        (224, 224),
        128,
    ).save(
        image_path
    )

    resolver = NIHImageResolver(
        tmp_path
    )

    resolver.build_index()

    assert (
        resolver.num_images
        == 1
    )

    resolved = resolver.resolve(
        "00000001_000.png"
    )

    assert (
        resolved
        == image_path
    )


def test_resolver_contains(
    tmp_path: Path,
):
    image_path = (
        tmp_path
        / "test.png"
    )

    Image.new(
        "RGB",
        (32, 32),
    ).save(
        image_path
    )

    resolver = NIHImageResolver(
        tmp_path
    )

    assert resolver.contains(
        "test.png"
    )

    assert not resolver.contains(
        "missing.png"
    )