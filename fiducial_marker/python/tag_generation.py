import argparse

from rune_tag import Runetag

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate rune tag')
    parser.add_argument('--pixel_size', type=int, help='pixel size of the square side of generated image', default=128)
    parser.add_argument('--num_layers', type=int, help='number of layers in the tag', default=2)
    parser.add_argument('--num_dots_per_layer', type=int, help='number of dots per layers in the tag', default=24)
    parser.add_argument('--output_path', type=str, help='path to save the image', default=None)
    args = parser.parse_args()

    rune_tag = Runetag(num_layers=args.num_layers, num_dots_per_layer=args.num_dots_per_layer)
    rune_tag.generate_random_tag(
        pixel_size=args.pixel_size,
        output_path=args.output_path,
    )
