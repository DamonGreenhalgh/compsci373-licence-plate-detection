# Brief

## Task (10 Points) 

Aim is to detect bounding box around the license plate in an image of a car.

## Topics

- Conversion to Greyscale
- Contrast Stretching
- Filtering to detect high contrast regions
- Thresholding for Segmentation
- Morphological operations
- Connected Component analysis

## Algorithm

- [x] Read the input image, convert RGB data to greyscale, and strech the values to lie between 0 and 255 (Greyscale conversion and Contrast Stretching).
- [x] Find structs with high contrast in the image by computing the standard deviation in the pixel neighborhood (Filtering).
- [x] Perform a thresholding operation to get the high contrast regions as a binary image (Thresholding for Segmentation). Hint: a good threshold value is around 150.
- [x] Perform several 3x3 dilation steps followed by several 3x3 erosion steps to get a blob region for the license plate (Morphological operations).
- [ ] Perform a connected component analysis to find the largest connected object (Connected Component analysis).
- [ ] Extract the final bounding box around this region, by looping over the image and looking for the minimum and maximum x and y coordinates of the pixels of the previously determined connected component.

## Extension (5 Points)

Extend your algorithm in a unique way. Write a reflective report on the challenges you faced and how you have altered the program.

# 2022_S1_CS373_AssignmentSkeleton

This repository provides a Python 3 code skeleton for the image processing assignment of CompSci 373 in Semester 1, 2022.

This assignment will require you to use what we have studied in the image processing lectures to generate a software that detects license plates in images of cars - a technology that is used routinely for example on toll roads for automatic toll pricing.

You will receive 10 marks for solving the license plate detection problem, and there will be an additional component for 5 marks, where you will extend upon the license plate detection, and write a short reflective report about your extension.

