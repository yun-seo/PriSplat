import os
import argparse
import mediapy as media
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--factor", type=int, default=8)
    args = parser.parse_args()

    for r in {1, 4}:
        dir_suffix = f"_{args.factor*r}"
        input_image_dir = os.path.join(args.dataset, "images")
        image_dir = os.path.join(args.dataset, "images" + dir_suffix)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            im_files = os.listdir(input_image_dir)
            for im_file in tqdm(im_files, desc=f"Downsample factor {args.factor*r}"):
                im = media.read_image(os.path.join(input_image_dir, im_file))

                resized_im = media.resize_image(
                    im, (int(im.shape[0]//(args.factor*r)), int(im.shape[1]//(args.factor*r)))
                )
                media.write_image(os.path.join(image_dir, im_file), resized_im)
        else:
            print(f"{image_dir} already exists! no downsampling.")
