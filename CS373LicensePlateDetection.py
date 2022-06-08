'''
COMPSCI 373 Assignment - Licence Plate Detection
Name: Damon Greenhalgh
UPI: dgre615
'''

import math
import sys
from pathlib import Path
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# Extension import
from CS373Extension import computeVignette


# Queue data structure used for connected component analysis
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


def convertToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b):
    '''
    This function converts an RGB image to GREYSCALE image.
    '''

    image = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            # Each color has specific coefficients based on their natural 'intensity'.
            image[i][j] = 0.299 * px_array_r[i][j] + 0.587 * px_array_g[i][j] + 0.114 * px_array_b[i][j]

    return image


def computeStandardDeviation(pixel_array, image_width, image_height):
    '''
    This function computes a 5x5 Standard Deviation filter across the argument
    pixel_array to find high contrast structures within the image.
    '''

    image = createInitializedGreyscalePixelArray(image_width, image_height, 0.0)

    for i in range(2, image_height - 2):
        for j in range(2, image_width - 2):

            values = []

            for k in [-2, -1, 0, 1, 2]:
                for l in [-2, -1, 0, 1, 2]:

                    # Add all pixels in a 5x5 neighbourhood
                    values.append(pixel_array[i + k][j + l])

            # Compute mean and standard deviation
            mean = sum(values) / 25
            sd = pow(sum([pow(value - mean, 2) for value in values]) / 25, 0.5)
            image[i][j] = sd

    return image


def computeThresholdSegmentation(pixel_array, image_width, image_height, threshold):
    '''
    This function converts an image to a binary black and white image. Pixel values
    are determined by the threshold argument.
    '''

    image = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):

            if pixel_array[i][j] < threshold:
                image[i][j] = 0
            else:
                image[i][j] = 1

    return image


def computeMinAndMaxValues(pixel_array, image_width, image_height):
    '''
    Helper function for the scale() function. This functions finds the min
    and max values of the pixel_array argument.
    '''

    # Default min and max values, at the boundary
    max_value = 0
    min_value = 255

    # Iterate over every pixel in the image
    for i in range(image_height):
        for j in range(image_width):

            value = pixel_array[i][j]

            # Update max if pixel value is larger then current max_value
            if value > max_value:
                max_value = value

            # Update min if pixel value is smaller then current min_value
            if value < min_value:
                min_value = value

    return (min_value, max_value)


def computeScale(pixel_array, image_width, image_height):
    '''
    This function scales the intensities of an image to its max range (0, 255).
    '''

    # Create new blank image that we will update and return
    image = createInitializedGreyscalePixelArray(image_width, image_height)

    # Find min and max values of pixel_array
    f_min, f_max = computeMinAndMaxValues(pixel_array, image_width, image_height)

    # Default bounds
    g_min, g_max = 0, 255

    # In the case min and max of the image are the same return blank image
    if f_min == f_max:
        return image

    # Iterate over every pixel
    for i in range(image_height):
        for j in range(image_width):

            # Compute gain and bias coefficients
            a = (g_max - g_min) / (f_max - f_min)
            b = g_min - f_min * a

            # Generate transformed value
            new_value = round(a * pixel_array[i][j] + b)

            # Clamp to keep new_value between the bounds
            # Keep new_value <= g_max (255)
            if new_value > g_max:
                new_value = g_max

            # Keep new_value >= g_min (0)
            if new_value < g_min:
                new_value = g_min

            # Update pixel in image
            image[i][j] = new_value

    return image


def computeDilation(pixel_array, image_width, image_height):
    '''
    This function performs a dilation operation on the argument pixel_array.
    '''

    image = createInitializedGreyscalePixelArray(image_width, image_height)

    # Structuring element
    se = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]

    # For each pixel in the pixel_array
    for i in range(image_height):
        for j in range(image_width):

            # Hit flag, assume false and flag true if at least
            # one pixel in se matches.
            hit = False

            # Iterate over a 3x3 neighbourhood
            for k in [-1, 0, 1]:
                for l in [-1, 0, 1]:

                    y = i + k
                    x = j + l

                    # Keep index within the bounds of the image.
                    if -1 < y < image_height and -1 < x < image_width:

                        if pixel_array[y][x] != 0 and se[k + 1][l + 1] == 1:
                            hit = True

            if hit:
                image[i][j] = 1

    return image


def computeErosion(pixel_array, image_width, image_height):
    '''
    This function performs a erosion operation on the argument pixel_array.
    '''

    image = createInitializedGreyscalePixelArray(image_width, image_height)

    # Structuring element
    se = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]

    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):

            # Fit flag, assume true, flag false if we find one
            # neighbouring pixel that does not match se if se = 1
            fit = True

            for k in [-1, 0, 1]:
                for l in [-1, 0, 1]:

                    value = pixel_array[i + k][j + l]

                    if value == 0 and value != se[k + 1][l + 1]:
                        fit = False

            if fit:
                image[i][j] = 1

    return image


def computeConnectedComponent(pixel_array, image_width, image_height):
    '''
    This function computes and finds the connected components within the pixel_array.
    Each connected component is assigned a label starting from 1 onwards. This uses a
    4-neighbourhood scheme to determine 'connectedness'. This function follows the standard
    connected component algorithm.
    '''

    image = createInitializedGreyscalePixelArray(image_width, image_height)

    # This array keeps track of the pixels that have been visited (1) or not visited (0).
    visited = createInitializedGreyscalePixelArray(image_width, image_height)

    label = 1
    info = {}

    # Iterate over each pixel
    for i in range(image_height):
        for j in range(image_width):

            # If the pixel is 'active' and has not been visited
            if pixel_array[i][j] != 0 and visited[i][j] != 1:

                q = Queue()          # Initialize new queue
                q.enqueue((i, j))    # Enqueue the pixel coordinates
                visited[i][j] = 1    # Update visited array
                count = 0            # Initialize a count for the number of pixel within this connected component

                while not q.isEmpty():

                    (y, x) = q.dequeue()    # Dequeue pixel
                    image[y][x] = label     # Label the image with current connected component label
                    count += 1              # Increment count

                    # Denotes the 4-neighbourhood region to check around
                    neighbours = [
                        (y + 1, x),
                        (y - 1, x),
                        (y, x + 1),
                        (y, x - 1)
                    ]

                    for px in neighbours:

                        if -1 < px[0] < image_height and -1 < px[1] < image_width:

                            # If the pixel is 'active' and has not been visited
                            if pixel_array[px[0]][px[1]] != 0 and visited[px[0]][px[1]] != 1:
                                q.enqueue((px[0], px[1]))
                                visited[px[0]][px[1]] = 1

                info[label] = count
                label += 1

    return image, info


def computeBoundingBox(pixel_array, image_width, image_height, cc_dict):
    '''
    This function computes the bounding box region for the licence plate.
    '''

    aspect_ratio = 0
    cc_index = -1

    # Keep iterating through each connected component until it meets the following
    # requirements.
    while not (1.5 < aspect_ratio < 5) and cc_index < len(cc_dict) - 1:

        cc_index += 1
        label = cc_dict[cc_index][0]

        min_x = image_width - 1
        max_x = 0
        min_y = image_height - 1
        max_y = 0

        for i in range(image_height):
            for j in range(image_width):

                if pixel_array[i][j] == label:

                    if j < min_x:
                        min_x = j

                    if j > max_x:
                        max_x = j

                    if i < min_y:
                        min_y = i

                    if i > max_y:
                        max_y = i

        # Avoid division by zero
        if max_y != min_y:
            aspect_ratio = (max_x - min_x) / (max_y - min_y)

    return min_x, max_x, min_y, max_y


# This is our code skeleton that performs the licence plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect licence plates!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate1.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])


    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(4, 2)
    fig1.set_figwidth(15)
    fig1.set_figheight(20)


    # STUDENT IMPLEMENTATION

    # Convert RGB image to Greyscale
    grey_array = convertToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b)
    px_array = grey_array
    axs1[0, 0].set_title('Greyscale')
    axs1[0, 0].imshow(grey_array, cmap='gray')

    ''' Uncomment below to generate extension filter '''
    # EXTENSION - Vignette
    vignette_strength = 0.45
    px_array = computeVignette(px_array, image_width, image_height, vignette_strength)
    axs1[0, 1].set_title('EXTENSION Vignette (strength={})'.format(vignette_strength))
    axs1[0, 1].imshow(px_array, cmap='gray')

    # Standard Deviation and Scale
    px_array = computeStandardDeviation(px_array, image_width, image_height)
    axs1[1, 0].set_title('Standard Deviation')
    axs1[1, 0].imshow(px_array, cmap='gray')

    # Scale Intensities
    px_array = computeScale(px_array, image_width, image_height)
    axs1[1, 1].set_title('Scale')
    axs1[1, 1].imshow(px_array, cmap='gray')

    # Threshold Segmentation
    threshold = 140
    px_array = computeThresholdSegmentation(px_array, image_width, image_height, threshold)
    axs1[2, 0].set_title('Threshold Segmentation (threshold={})'.format(threshold))
    axs1[2, 0].imshow(px_array, cmap='gray')

    # Dilation operations
    num_dilation = 3
    num_erosion = 1
    for i in range(num_dilation):
        px_array = computeDilation(px_array, image_width, image_height)

    # Erosion operations
    for i in range(num_erosion):
        px_array = computeErosion(px_array, image_width, image_height)

    axs1[2, 1].set_title('Morphological Operations (dilation={0}, erosion={1})'.format(num_dilation, num_erosion))
    axs1[2, 1].imshow(px_array, cmap='gray')

    # Connected Component Analysis
    px_array, cc_dict = computeConnectedComponent(px_array, image_width, image_height)
    axs1[3, 0].set_title('Connected Component Detection')
    axs1[3, 0].imshow(px_array, cmap='gray')

    # Sort to find largest connect component
    cc_dict = sorted(cc_dict.items(), key=lambda item: item[1], reverse=True)

    # Compute bounding box
    bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y = computeBoundingBox(px_array, image_width, image_height, cc_dict)


    # Draw a bounding box as a rectangle into the input image
    axs1[3, 1].set_title('Final image of detection')
    axs1[3, 1].imshow(grey_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=3,
                     edgecolor='g', facecolor='none')
    axs1[3, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[3, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()
