import math
import sys
from pathlib import Path
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png


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

# This function computes the standard deviation of the input image in a 3x3 window.
def computeStandardDeviationImage3x3(pixel_array, image_width, image_height):
    image = createInitializedGreyscalePixelArray(image_width, image_height, 0.0)

    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):

            values = []

            for k in [-1, 0, 1]:
                for l in [-1, 0, 1]:
                    values.append(pixel_array[i + k][j + l])

            mean = sum(values) / 9
            image[i][j] = pow(sum([pow(value - mean, 2) for value in values]) / 9, 0.5)

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
def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    
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


# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
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
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')


    # STUDENT IMPLEMENTATION here
    # Convert RGB image to Greyscale
    px_array = convertToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b)

    # Scale and Quantize the image
    px_array = scaleTo0And255AndQuantize(px_array, image_width, image_height)

    # Compute Standard Deviation
    px_array = computeStandardDeviationImage3x3(px_array, image_width, image_height)


    # compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    bbox_min_x = center_x - image_width / 4.0
    bbox_max_x = center_x + image_width / 4.0
    bbox_min_y = center_y - image_height / 4.0
    bbox_max_y = center_y + image_height / 4.0

    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()
