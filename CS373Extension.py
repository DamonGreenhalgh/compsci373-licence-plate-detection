'''
COMPSCI 373 Assignment EXTENSION
Name: Damon Greenhalgh
UPI: dgre615
'''


# Helper function for the Vignette filter
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


def computeVignette(pixel_array, image_width, image_height, strength):
    '''
    EXTENSION computeVignette
    -------------------------
    This function applies a 'Vignette' to the input image i.e. Darkens the pixel
    intensities relative to the proximity away from the center of the image. This
    algorithm uses the Euclidean distance metric to compute the weight coefficient
    of each pixel.

    float: [0, 1] strength - this argument is the strength of the applied filter.
        The lower the strength the weaker the darkening of the edges.
    '''

    image = createInitializedGreyscalePixelArray(image_width, image_height)

    # Compute center point of image
    mid_x = image_width // 2
    mid_y = image_height // 2

    # Compute the maximum Euclidean distance from the center
    max_dist = pow(pow(mid_x, 2) + pow(mid_y, 2), 0.5)

    # Iterate over each pixel
    for i in range(image_height):
        for j in range(image_width):

            # Compute the weight as the Euclidean distance (j, i) from center
            # divided by the maximum distance.
            weight = pow(pow(j - mid_x, 2) + pow(i - mid_y, 2), 0.5) / max_dist * strength
            image[i][j] = (1 - weight) * pixel_array[i][j]

    return image

