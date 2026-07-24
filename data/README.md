# Dataset

`input/` contains 103 images converted to PNG from the
[USC-SIPI Image Database](https://sipi.usc.edu/database/), maintained by the
Signal and Image Processing Institute at the University of Southern
California. The database is published for research and educational use, and
the images are redistributed here on that basis so the project runs without a
download step.

Volumes included:

| Volume | Files | Sizes |
|--------|-------|-------|
| Miscellaneous (`misc_*`) | 39 | 256x256, 512x512, 1024x1024 |
| Textures (`textures_*`) | 64 | 512x512, 1024x1024 |

Totals: 103 files, 46.27 megapixels, 89 grayscale and 14 colour.

File names encode their origin, so `misc_4_2_03.png` is `4.2.03.tiff` from the
Miscellaneous volume. The only change made to the originals is the TIFF to PNG
container conversion; pixel data is unmodified.

Reproduce this directory from the upstream source with:

```bash
./scripts/fetch_sipi_data.sh            # misc + textures, as shipped
./scripts/fetch_sipi_data.sh aerials    # optionally add 38 larger images
```

`output/` is where the pipeline writes its edge maps; it starts empty.
