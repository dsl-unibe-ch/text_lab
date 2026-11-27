"""Writing a result's files into a folder an interface serves them from.

The features' ``save_*`` functions write a finished run's downloads, page
images and tables into a folder, such as a web job's outputs, and return
what they wrote (:class:`SavedFile`), so an interface can offer the
downloads and show or link the rest.
"""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class SavedFile:
    """A file a feature wrote for an interface.

    Attributes:
        name: Its file name, unique in the folder.
        label: What it is, for a download list.
        media_type: Its media type.
        path: Where it is.
        listed: Whether it is a download of its own; page images, figures
            and single tables are shown or linked in the result instead.
    """

    name: str
    label: str
    media_type: str
    path: Path
    listed: bool = True


class OutputFolder:
    """A folder that notes every file written into it, in order.

    Attributes:
        folder: The folder; it must exist.
        files: The files written so far.
    """

    def __init__(self, folder: Path) -> None:
        """Write into ``folder``, which must exist."""
        self.folder = folder
        self.files: list[SavedFile] = []

    def write(
        self,
        name: str,
        data: bytes,
        label: str,
        media_type: str,
        *,
        listed: bool = True,
    ) -> str:
        """Write a file and note it.

        Args:
            name: Its file name.
            data: Its content.
            label: What it is, for a download list.
            media_type: Its media type.
            listed: Whether it is a download of its own.

        Returns:
            The file name, to refer to the file in a result.
        """
        path = self.folder / name
        path.write_bytes(data)
        self.files.append(SavedFile(name, label, media_type, path, listed))
        return name


def web_image(
    data: bytes, max_side: int | None = None
) -> tuple[bytes | None, str, tuple[int, int]]:
    """Return an image as a browser shows it: PNG or JPEG data.

    Args:
        data: Image file data, in any format PIL reads.
        max_side: Scale the image down to this longest side, if it is
            larger; scaled photos are saved as JPEG.

    Returns:
        The data (as it is if it is PNG or JPEG and small enough, else
        converted), its file extension, and the size of the original image
        as ``(width, height)``, so outlines in its pixels still fit; ``None``
        data if it cannot be read.
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as picture:
            size = picture.size
            scale = 1.0
            if max_side and max(size) > max_side:
                scale = max_side / max(size)
            if scale == 1.0 and picture.format in ("PNG", "JPEG"):
                return data, "png" if picture.format == "PNG" else "jpg", size
            image = picture.convert("RGB")
        buffer = io.BytesIO()
        if scale < 1.0:
            image = image.resize(
                (round(size[0] * scale), round(size[1] * scale)),
                Image.Resampling.LANCZOS,
            )
            image.save(buffer, format="JPEG", quality=85)
            return buffer.getvalue(), "jpg", size
        image.save(buffer, format="PNG")
        return buffer.getvalue(), "png", size
    except Exception:
        return None, "png", (0, 0)


def image_media_type(extension: str) -> str:
    """Return the media type of an image file extension (``png``, ``jpg``)."""
    return f"image/{'jpeg' if extension in ('jpg', 'jpeg') else extension}"
