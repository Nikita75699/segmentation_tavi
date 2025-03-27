import os
import json
import numpy as np
from PIL import Image, ImageDraw


def create_mask_from_annotation(image_size, annotation):
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)

    for obj in annotation['objects']:
        points = [(point['x'], point['y']) for point in obj['points']['exterior']]
        draw.polygon(points, outline=1, fill=1)

    return np.array(mask)


def process_data(images_dir, annotations_dir, output_masks_dir):
    if not os.path.exists(output_masks_dir):
        os.makedirs(output_masks_dir)

    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(images_dir, filename)
            annotation_path = os.path.join(annotations_dir, filename.replace('.jpg', '.json').replace('.png', '.json'))

            with open(annotation_path, 'r') as f:
                annotation = json.load(f)

            image = Image.open(image_path)
            mask = create_mask_from_annotation(image.size, annotation)

            mask_image = Image.fromarray(mask * 255).convert('L')
            mask_image.save(os.path.join(output_masks_dir, filename))


if __name__ == '__main__':
    images_dir = 'path/to/images'
    annotations_dir = 'path/to/annotations'
    output_masks_dir = 'path/to/output/masks'

    process_data(images_dir, annotations_dir, output_masks_dir)
