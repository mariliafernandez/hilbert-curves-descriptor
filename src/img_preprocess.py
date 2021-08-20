from hilbert_curve import *
from PIL import Image, ImageDraw
from scipy.io import loadmat
import matplotlib.pyplot as plt
import argparse


def segment(img_path, annotation_path):

    dataset_path = Path("../caltech-101")

    img_full = Image.open(img_path).convert("L")
    annotation_array = loadmat(annotation_path)

    # Cut image outside box
    y1,y2,x1,x2 = annotation_array['box_coord'][0]

    if y1!=y2 and x1!=x2:
        img_cut = img_full.crop((x1-1,y1-1,x2+1,y2+1))

        contour = []
        for y, x in zip(annotation_array['obj_contour'][0], annotation_array['obj_contour'][1]):
            contour.append(y)
            contour.append(x) 

        array_shape = np.asarray(img_cut).shape
        bounds = Image.fromarray(np.zeros(array_shape))

        blank = ImageDraw.Draw(bounds)
        blank.line(contour, fill=255)

        bounds_array = np.asarray(bounds).astype('uint8')

        se = np.ones((7,7), dtype='uint8')
        image_close = cv.morphologyEx(bounds_array, cv.MORPH_CLOSE, se)

        cnt = cv.findContours(image_close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        cnt_mask = np.zeros(image_close.shape[:2], np.uint8)
        filled_mask = cv.drawContours(cnt_mask, cnt, -1, 255, -1)

        mask = filled_mask//255
        final = np.asarray(img_cut) * mask

        return final, filled_mask
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images', type=Path)
    parser.add_argument('annotations', type=Path)

    args = parser.parse_args()
    object = args.images.name

    img_output_dir = Path('caltech-101/segmented_images') / str(object)
    mask_output_dir = Path('caltech-101/masks') / str(object)

    for img_path in args.images.iterdir():

        file_index = img_path.name.split('.')[0].split('_')[-1]  
        annotation_filename = f'annotation_{file_index}.mat'
        annotation_path = args.annotations / annotation_filename
        
        segmented = segment(img_path, annotation_path)
        
        if segmented is not None:
            
            img_segmented, mask = segmented

            img_result = Image.fromarray(img_segmented)
            mask_result = Image.fromarray(mask)

            if not img_output_dir.exists():
                img_output_dir.mkdir()
            if not mask_output_dir.exists():
                mask_output_dir.mkdir()

            output_filename = f'{file_index}.jpg'

            output_image_path = img_output_dir / output_filename
            output_mask_path = mask_output_dir / output_filename
            
            img_result.save(str(output_image_path))
            mask_result.save(str(output_mask_path))
