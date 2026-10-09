import numpy as np

def row_to_image(row, img_size=14):

    row_str = ",".join([str(x) for x in row])

    ascii_vals = [ord(c) for c in row_str]

    total_pixels = img_size * img_size

    ascii_vals = (
        ascii_vals[:total_pixels]
        + [0] * max(0, total_pixels - len(ascii_vals))
    )

    return np.array(
        ascii_vals,
        dtype=np.uint8
    ).reshape(img_size, img_size)