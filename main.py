
IMAGE_PATH = R"image.png"
BLOCK_SIZE = 4 # for display only, not in gd
LIMIT = 9998



# ignore bad code
import cv2
import numpy as np
import colorsys

def lerp(_a, _b, t):
    a = float(_a)
    b = float(_b)
    return a + (b - a) * t

# thanks chatgpt
def resize_under_limit(original_shape, limit):
    height, width = original_shape
    aspect_ratio = width / height
    
    # Ensure the dimensions are within bounds
    new_height = int((limit / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)
    
    return (new_height, new_width)


def gen_image(img):
    image = img
    if image is None:
        raise Exception("Image not found or unable to load.")


    y_size, x_size = image.shape[:2]
    y_size_grad, x_size_grad = resize_under_limit((y_size, x_size), LIMIT)
    image = cv2.cvtColor(
        cv2.resize(image, (x_size_grad * BLOCK_SIZE, y_size_grad * BLOCK_SIZE), 
                   interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    y_size, x_size = image.shape[:2]

    new_image = np.full((y_size_grad, x_size_grad, 2, 3), 
                        np.array(np.full((2, 3), (0, 0, 0), dtype=np.float32)))

    for i in range(0, y_size_grad):
        for j in range(0, x_size_grad):
            x, y = j*BLOCK_SIZE, i*BLOCK_SIZE
            
            first_column = image[y:y + BLOCK_SIZE, x]
            last_column = image[y:y + BLOCK_SIZE, x + BLOCK_SIZE-1]

            new_image[i][j][0] = colorsys.rgb_to_hsv(*(np.mean(first_column, 0) / 255))
            new_image[i][j][1] = colorsys.rgb_to_hsv(*(np.mean(last_column, 0) / 255))
            # print(new_image[i][j][0])

    return new_image

def show_image(image):
    y_size_grad, x_size_grad = image.shape[:2]
    y_size, x_size = y_size_grad * BLOCK_SIZE, x_size_grad * BLOCK_SIZE

    a = np.full((y_size, x_size, 3), np.array([0, 0, 0], np.uint8))
    for i in range(y_size_grad):
        for j in range(x_size_grad):
            for x in range(BLOCK_SIZE):
                for y in range(BLOCK_SIZE):
                    gr = image[i][j]
                    gr0 = colorsys.hsv_to_rgb(*gr[0])
                    gr1 = colorsys.hsv_to_rgb(*gr[1])

                    for k in range(3):
                        val = lerp(gr0[k] * 255, gr1[k] * 255, x / BLOCK_SIZE)
                        a[i * BLOCK_SIZE + y][j * BLOCK_SIZE + x][k] = val
    
    cv2.imshow('Image', cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image = gen_image(cv2.imread(IMAGE_PATH))
print(image.shape[:2], image.shape[0] * image.shape[1])
with open("out.json", "w") as f:
    f.write(str(np.multiply(image, 255).astype(np.uint8).tolist()).replace(" ", ""))
show_image(image)