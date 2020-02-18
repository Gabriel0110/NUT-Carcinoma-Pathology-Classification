from skimage import transform, util, exposure
from scipy import ndarray, ndimage
import random
import numpy as np
import skimage as sk
import skimage.io as skio

def random_rotation(image_array: ndarray):
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1]
    
def vertical_flip(image_array: ndarray):
    return image_array[::-1, :]

def invert(image_array: ndarray):
    return np.invert(image_array)
    
def intensity(image_array: ndarray):
    v_min, v_max = np.percentile(image_array, (0.2, 99.8))
    better_contrast = exposure.rescale_intensity(image_array, in_range=(v_min, v_max))
    return better_contrast
    
def gamma(image_array: ndarray):
    adjustment = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    adjusted_gamma_image = exposure.adjust_gamma(image_array, gamma=random.choice(adjustment), gain=random.choice(adjustment))
    return adjusted_gamma_image
    
def blur(image_array: ndarray):
    blurred_image = ndimage.uniform_filter(image_array, size=(11, 11, 1))
    return blurred_image

def augment_images(target, amt):
    transforms = {
        'rotate': random_rotation,
        'noise': random_noise,
        'horizontal_flip': horizontal_flip,
        'vertical_flip': vertical_flip,
        'invert': invert,
        'intensity': intensity,
        'gamma': gamma,
        'blur': blur
    }

    folder_path = 'C:\\Users\\gabri\\Desktop\\NMC Pathology\\' + target
    files_to_make = amt

    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    num_generated_files = 0
    while num_generated_files < files_to_make:
        image_to_transform = skio.imread(random.choice(images))
        num_transforms = random.randint(2, len(transforms))
        transform_count = 0
        transformed_image = None
        while transform_count <= num_transforms:
            key = random.choice(list(transforms))
            transformed_image = transforms[key](image_to_transform)
            transform_count += 1

        new_file_path = '%s/augmented_image_%s.png' % (folder_path, num_generated_files)

        skio.imsave(new_file_path, transformed_image)
        num_generated_files += 1