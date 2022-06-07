import math
import sys
from pathlib import Path
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png


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

# This function computes the standard deviation of the input image.
def standardDeviation(pixel_array, image_width, image_height):
    image = createInitializedGreyscalePixelArray(image_width, image_height, 0.0)
    region_size = 2
    region = [(i - region_size) for i in range(region_size * 2 + 1)]
    norm = pow(region_size * 2 + 1, 2)
    print(region, norm)
    for i in range(region_size, image_height - region_size):
        for j in range(region_size, image_width - region_size):

            values = []

            for k in region:
                for l in region:
                    values.append(pixel_array[i + k][j + l])

            mean = sum(values) / norm
            image[i][j] = pow(sum([pow(value - mean, 2) for value in values]) / norm, 0.5)

    return image


# Helper function to compute the min and max values of a image
def computeMinAndMaxValues(pixel_array, image_width, image_height):
    
    # Default min and max values. At the boundary.
    max_value = 0
    min_value = 255
    
    # Iterate over every pixel in the image
    for i in range(image_height):
        for j in range(image_width):
            
            value = pixel_array[i][j]
            
            # Update max
            if value > max_value:
                max_value = value
            
            # Update min
            if value < min_value:
                min_value = value
                
    return (min_value, max_value)

            
# This function scales the intensities of an image
def scale(pixel_array, image_width, image_height):
    
    # Create identical dimension blank image to be used as a updated version of 
    # the parameter image after transformation.
    image = createInitializedGreyscalePixelArray(image_width, image_height)
    
    # Find min and max values of the image with the use of a helper function below
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
            a = (g_max - g_min)/(f_max - f_min)
            b = g_min - f_min*a
            
            # Generate transformed value
            new_value = round(a*pixel_array[i][j] + b)
            
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


# This function converts a rgb image to a greyscale image
def convertToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b):

    greyscale_image = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            greyscale_image[i][j] = 0.299*px_array_r[i][j] + 0.587*px_array_g[i][j] + 0.114*px_array_b[i][j]
    
    return greyscale_image

def thresholdSegmentation(pixel_array, image_width, image_height, threshold):

    image = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):

            if pixel_array[i][j] < threshold:
                image[i][j] = 0
            else:
                image[i][j] = 1

    return image


def dilation(pixel_array, image_width, image_height):
    image = createInitializedGreyscalePixelArray(image_width, image_height)

    se = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]

    for i in range(image_height):
        for j in range(image_width):

            hit = False

            for k in [-1, 0, 1]:
                for l in [-1, 0, 1]:

                    y = i + k
                    x = j + l

                    if -1 < y < image_height and -1 < x < image_width:

                        active = False
                        if pixel_array[y][x] == 1 or pixel_array[y][x]:
                            active = True

                        if active and 1 == se[k + 1][l + 1]:
                            hit = True

            if hit:
                image[i][j] = 1

    return image


def erosion(pixel_array, image_width, image_height):

    image = createInitializedGreyscalePixelArray(image_width, image_height)

    se = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]

    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):

            fit = True

            for k in [-1, 0, 1]:
                for l in [-1, 0, 1]:

                    value = pixel_array[i + k][j + l]

                    if value == 0 and value != se[k + 1][l + 1]:
                        fit = False

            if fit:
                image[i][j] = 1

    return image


def connectedComponent(pixel_array, image_width, image_height):
    image = createInitializedGreyscalePixelArray(image_width, image_height)
    visited = createInitializedGreyscalePixelArray(image_width, image_height)

    label = 1
    info = {}

    for i in range(image_height):
        for j in range(image_width):

            if pixel_array[i][j] != 0 and visited[i][j] != 1:

                q = Queue()
                q.enqueue((i, j))
                visited[i][j] = 1
                count = 0

                while not q.isEmpty():

                    (y, x) = q.dequeue()
                    image[y][x] = label
                    count += 1

                    neighbours = [
                        (y + 1, x),
                        (y - 1, x),
                        (y, x + 1),
                        (y, x - 1)
                    ]

                    for px in neighbours:

                        if -1 < px[0] < image_height and -1 < px[1] < image_width:

                            if pixel_array[px[0]][px[1]] != 0 and visited[px[0]][px[1]] != 1:
                                q.enqueue((px[0], px[1]))
                                visited[px[0]][px[1]] = 1

                info[label] = count
                label += 1

    return image, info


# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate4.png"

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
    fig1, axs1 = pyplot.subplots(7, 1)
    # axs1[0, 0].set_title('Input red channel of image')
    # axs1[0, 0].imshow(px_array_r, cmap='gray')
    # axs1[0, 1].set_title('Input green channel of image')
    # axs1[0, 1].imshow(px_array_g, cmap='gray')
    # axs1[1, 0].set_title('Input blue channel of image')
    # axs1[1, 0].imshow(px_array_b, cmap='gray')

    fig1.set_figwidth(5)
    fig1.set_figheight(21)


    # STUDENT IMPLEMENTATION here

    # Convert RGB image to Greyscale
    grey_array = convertToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b)
    axs1[0].set_title('Greyscale')
    axs1[0].imshow(grey_array, cmap='gray')

    # Standard Deviation and Scale
    px_array = standardDeviation(grey_array, image_width, image_height)
    px_array = scale(px_array, image_width, image_height)
    axs1[1].set_title('Standard Deviation and Scale')
    axs1[1].imshow(px_array, cmap='gray')

    # Threshold Segmentation
    px_array = thresholdSegmentation(px_array, image_width, image_height, 140)
    axs1[2].set_title('Threshold Segmentation')
    axs1[2].imshow(px_array, cmap='g    ray')

    # Dilation operations
    for i in range(5):
        px_array = dilation(px_array, image_width, image_height)

    axs1[3].set_title('Dilation Operations')
    axs1[3].imshow(px_array, cmap='gray')

    # Erosion operations
    for i in range(5):
        px_array = erosion(px_array, image_width, image_height)

    axs1[4].set_title('Erosion Operations')
    axs1[4].imshow(px_array, cmap='gray')

    # Open operation
    px_array = erosion(px_array, image_width, image_height)
    px_array = dilation(px_array, image_width, image_height)

    # Connected Component Analysis
    px_array, cc_dict = connectedComponent(px_array, image_width, image_height)
    axs1[5].set_title('Connected Component Detection')
    axs1[5].imshow(px_array, cmap='gray')

    # Sort to find largest connect component
    cc_dict = sorted(cc_dict.items(), key=lambda item: item[1], reverse=True)

    # Compute bounding box
    aspect_ratio = 0
    cc_index = -1

    while not (1.5 < aspect_ratio < 5):
        cc_index += 1
        label = cc_dict[cc_index][0]
        bbox_min_x = image_width - 1
        bbox_max_x = 0
        bbox_min_y = image_height - 1
        bbox_max_y = 0

        for i in range(image_height):
            for j in range(image_width):

                if px_array[i][j] == label:

                    if j < bbox_min_x:
                        bbox_min_x = j

                    if j > bbox_max_x:
                        bbox_max_x = j

                    if i < bbox_min_y:
                        bbox_min_y = i

                    if i > bbox_max_y:
                        bbox_max_y = i

        aspect_ratio = (bbox_max_x - bbox_min_x) / (bbox_max_y - bbox_min_y)


    # Draw a bounding box as a rectangle into the input image
    axs1[6].set_title('Final image of detection')
    axs1[6].imshow(grey_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[6].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[6].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()
