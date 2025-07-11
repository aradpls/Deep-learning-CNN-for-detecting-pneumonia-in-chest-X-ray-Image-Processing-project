import os
import numpy as np
import cv2

os.chdir("C:\\image_processing_course")
# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #set_random_seed(seed_value) # Keras uses its source of randomness regardless Theano or TensorFlow.In addition, TensorFlow has its own random number generator that must also be seeded by calling the set_random_seed() function immediately after the NumPy random number generator:

exec(open(os.path.abspath('image_common_utils.py')).read())


def selective_sharpening(image):
    
    #Applies selective sharpening to the image.
    
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def crop_bottom(image, pixels):
    # Crops the bottom 'pixels' rows from the image
    return image[:-pixels]

#function to Process the image
def process_and_save_images(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    #creating Clahe 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.jpeg', '.jpg')):
            input_file_path = os.path.join(input_directory, filename)
            image = cv2.imread(input_file_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                # Normalize the image
                image = crop_bottom(image, 100)#croping part of the image in the bootom
                image = cv2.resize(image, (180, 180))  # Resize image first to speed up processing
                image = (image - np.mean(image)) / np.std(image)
                image = np.clip(image, 0, 1)  # Ensure normalized values are within [0, 1]
                
                clahe_image = clahe.apply((image * 255).astype(np.uint8))
                
                #image for workflow - Clahe
                #cv2.imwrite(r"C:\Users\97254\Desktop\workflow\output\Clahe.jpeg", clahe_image)
                
                # Applying a morphological close operation with a different kernel size
                closing = cv2.morphologyEx(clahe_image, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                
                #image for workflow - closnig
                #cv2.imwrite(r"C:\Users\97254\Desktop\workflow\output\closing.jpeg", closing)
                
                # Applying bilateral filtering for noise reduction while preserving edges
                filtered = cv2.bilateralFilter(closing, 9, 75, 75)
                
                #image for workflow - bilateralFilter
                #cv2.imwrite(r"C:\Users\97254\Desktop\workflow\output\bilateralFilter.jpeg", filtered)
                
                # Enhanceing edges using the Laplacian of Gaussian (LoG)
                log = cv2.Laplacian(filtered, cv2.CV_16S, ksize=3)
                log = cv2.convertScaleAbs(log)
                
                #image for workflow - Enhanceing edges
                #cv2.imwrite(r"C:\Users\97254\Desktop\workflow\output\Enhanceingedges.jpeg", log)
                
                # Improved segmentation (e.g., Otsu's thresholding for foreground-background separation)
                ret, thresh = cv2.threshold(log, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                #image for workflow - Enhanceing edges
                #cv2.imwrite(r"C:\Users\97254\Desktop\workflow\output\otsu.jpeg", thresh)
                
                
                # Applying the watershed algorithm for segmentation on the preprocessed image
                # First, finding sure foreground area
                dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
                ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
                
                # Finding unknown region
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(closing, sure_fg)
                
                # Labeling the sure foreground
                ret, markers = cv2.connectedComponents(sure_fg)
                
                # Adding one to all labels so that sure background is not 0, but 1
                markers = markers+1
                
                # mark the region of unknown with zero
                markers[unknown==255] = 0
                
                markers = cv2.watershed(cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR), markers)
                clahe_image[markers == -1] = 255  # Marking the boundaries in the original image
                segmnted = cv2.applyColorMap(clahe_image, cv2.COLORMAP_BONE)
                #image for workflow - segmnted image
                #cv2.imwrite(r"C:\Users\97254\Desktop\workflow\output\cl2.jpeg", clahe_image)
                
                #image for workflow - after manupulation on orignal image
                #cv2.imwrite(r"C:\Users\97254\Desktop\workflow\output\cl32.jpeg", segmnted)
                
                sharpened_image = selective_sharpening(segmnted)
                #outputing images to the directory
                output_file_path = os.path.join(output_directory, filename)
                cv2.imwrite(output_file_path, sharpened_image)
            else:
                print(f"Failed to read the image: {filename}")
        else:
            print(f"{filename} is not a JPEG file and was skipped.")

            
#Noraml train images
input_directory1 = r"./X-Rray/ChestXRay2017/chest_xray/train/NORMAL"
output_directory1 = r"C:\Users\97254\Desktop\ChestXrayoutput\train\NORMAL"
process_and_save_images(input_directory1, output_directory1)

#Noraml test images
input_directory2 = r"./X-Rray/ChestXRay2017/chest_xray/test/NORMAL"
output_directory2 = r"C:\Users\97254\Desktop\ChestXrayoutput\test\NORMAL"
process_and_save_images(input_directory2, output_directory2)

#PNEUMONIA train images
input_directory3 = r"./X-Rray/ChestXRay2017/chest_xray/train/PNEUMONIA"
output_directory3 = r"C:\Users\97254\Desktop\ChestXrayoutput\train\PNEUMONIA"
process_and_save_images(input_directory3, output_directory3)

#PNEUMONIA test images
input_directory4 = r"./X-Rray/ChestXRay2017/chest_xray/test/PNEUMONIA"
output_directory4 = r"C:\Users\97254\Desktop\ChestXrayoutput\test\PNEUMONIA"
process_and_save_images(input_directory4, output_directory4)

#showing images for work flow
input_directory5 = r"C:\Users\97254\Desktop\workflow\input"
output_directory5 = r"C:\Users\97254\Desktop\workflow\output"
process_and_save_images(input_directory5, output_directory5)
