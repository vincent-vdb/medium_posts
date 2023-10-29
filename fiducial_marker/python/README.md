## Fiducial Marker

## How to generate random tags

To generate a random tag with the 2 circles and 16 dots per circle, 
just run the following command and store the output image in `image_tag.jpg`:

```bash
python tag_generation.py --num_layers 2 --num_dots_per_layer 16 --output_path image_tag.jpg
```
## How to generate synthetic images

To generate 200 synthetic images for object detection training in the folder `synthetic_data` 
(to create beforehand), run the following command:

```bash
python synthetic_images_generation.py --number 200
```

You may specify the number of circles and dots per circle, as well as
the folder for background images and the output folder with the following arguments:
- `--output_path`: path to save the images, defaults to `synthetic_data`
- `--background_path`: path to background images, defaults to `background_images`
- `--num_layers`: number of layers in the tag, defaults to `2`
- `--num_dots_per_layer`: number of dots per layers in the tag, defaults to `24`
