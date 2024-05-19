# PhenoBench version 1.1.0 (December 2023)

We provide here the images for train, validation (val), and test sets of the PhenoBench datasets. For train and validation, the dataset provides also pixel-wise annotations for semantics and instances of plants and leaves.

See https://www.phenobench.org for more information on the dataset and updates.

## Change Log

Version 1.0.0: initial release.
Version 1.1.0: temporally unique instance ids.

## Dataset format

For each image of the training and validation set, we provide the following annotations in the corresponding sub-folders:

 - images: Field images as RGB images of size 1024 x 1024. The beginning of the filename in train, validation set corresponds to the the date (05-15, 05-26, and 06-05) in 2020, when the images was taken from the field.
 - semantics: Pixel-wise semantic masks, where label ids correspond to: background (0), crop (1), weed(2), partial-crop (3), partial-crop (4). 
   Partial crops and weeds have less than 50% visible pixels.
 - plant_instances: Pixel-wise instance masks for crops and weed, where ids > 0 correspond to distinct instances.
 - leaf_instances: Pixel-wise instance masks for leaves, where ids > 0 correspond to distinct instances.
 - plant_visibility: Pixel-wise visibility in range [0,255], where 255 means fully visible.
 - leaf_visibility: Pixel-wise visibility in range [0,255], where 255 means fully visible.

We store all annotations as 16-bit png files as these provide a decent lossless compression and can be easily read with off-the-shelf pillow or OpenCV, minimizing the dependencies. All images and corresponding annotations have an image size of of size 1024 by 1024 pixels. The size was chosen such that even in later growth stages multiple plants are completely inside the image.

More information is available on https://www.phenobench.org.

## License

We distribute the data under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) 
(see https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

This means that you must attribute the work in the manner specified by the authors, you may not use this work for 
commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work 
only under the same license.

Specifically you should cite our work:

@article{weyler2023dataset,
  author = {Jan Weyler and Federico Magistri and Elias Marks and Yue Linn Chong and Matteo Sodano and Gianmarco Roggiolani
  and Nived Chebrolu and Cyrill Stachniss and Jens Behley},
  title = {{PhenoBench --- A Large Dataset and Benchmarks for Semantic Image Interpretation
    in the Agricultural Domain}},
  journal = {arXiv preprint},
  year = {2023}
}
