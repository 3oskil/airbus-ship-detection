import shutil
import numpy as np
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from keras.utils import Sequence
from sklearn.model_selection import train_test_split


class SegmentationDataGenerator(Sequence):
    def __init__(self, directory, batch_size=32, shuffle=True, augmentation=True, subset='train', img_size=(384, 384)):
        self.directory = Path(directory)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.subset = subset
        self.img_size = img_size

        self.image_dir = self.directory / subset / 'images'
        self.mask_dir = self.directory / subset / 'masks'

        self.images = [img.name for img in self.image_dir.glob('*')]
        self.masks = [mask.name for mask in self.mask_dir.glob('*')]

        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)

        self.augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.RandomShadow(p=0.1),
            A.OneOf([
                A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
            ], p=0.8),
        ])

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size

        batch_images = self.images[start:end]
        batch_masks = self.masks[start:end]

        x_batch = np.array(
            [np.array(Image.open(self.image_dir / img).resize(self.img_size)) for img in batch_images])
        y_batch = np.array([np.expand_dims(
            np.array(Image.open(self.mask_dir / msk).convert('L').resize(self.img_size)), axis=-1) for
            msk in batch_masks])

        if self.augmentation:
            for i in range(len(x_batch)):
                while x_batch[i].shape[0] <= 0 or x_batch[i].shape[1] <= 0:
                    x_batch[i] = np.array(Image.open(self.image_dir / batch_images[i]).resize(self.img_size))

                while y_batch[i].shape[0] <= 0 or y_batch[i].shape[1] <= 0:
                    y_batch[i] = np.expand_dims(np.array(
                        Image.open(self.mask_dir / batch_masks[i]).convert('L').resize(self.img_size)), axis=-1)

                augmented = self.augmentations(image=x_batch[i], mask=y_batch[i])
                x_batch[i] = augmented['image']
                y_batch[i] = augmented['mask']

        return x_batch / 255.0, y_batch / 255.0  # Normalize pixel values to [0, 1]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def show_batch(self, index=0, num_examples=None):
        x_batch, y_batch = self.__getitem__(index)

        if num_examples is None:
            num_examples = len(x_batch)

        for i in range(num_examples):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(x_batch[i])
            axes[0].set_title('Input Image')
            axes[0].axis('off')

            axes[1].imshow(np.squeeze(y_batch[i]), cmap='gray')
            axes[1].set_title('Mask')
            axes[1].axis('off')

            plt.show()

    def show_comparison(self, num_examples=3):
        indexes = np.random.choice(len(self.images), num_examples, replace=False)

        for i in indexes:
            original_image = np.array(Image.open(self.image_dir / self.images[i]).resize(self.img_size))
            original_mask = np.expand_dims(
                np.array(Image.open(self.mask_dir / self.masks[i]).convert('L').resize(self.img_size)), axis=-1)

            augmented = self.augmentations(image=original_image, mask=original_mask)
            augmented_image = augmented['image']
            augmented_mask = augmented['mask']

            fig, axes = plt.subplots(1, 4, figsize=(18, 6))

            axes[0].imshow(original_image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(np.squeeze(original_mask), cmap='gray')
            axes[1].set_title('Original Mask')
            axes[1].axis('off')

            axes[2].imshow(augmented_image)
            axes[2].set_title('Augmented Image')
            axes[2].axis('off')

            axes[3].imshow(np.squeeze(augmented_mask), cmap='gray')
            axes[3].set_title('Augmented Mask')
            axes[3].axis('off')

            plt.show()


def get_mask(img_id, df, dataset_img_size=(768, 768)):
    img = np.zeros(dataset_img_size[0] * dataset_img_size[1], dtype=np.uint8)
    masks = df[df['ImageId'] == img_id]['EncodedPixels']
    if pd.isnull(masks).any():
        return img.reshape(dataset_img_size).T
    if isinstance(masks, pd.Series):
        masks = masks.dropna().tolist()
    elif isinstance(masks, str):
        masks = [masks]
    for mask in masks:
        s = mask.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start:start + length] = 1
    return img.reshape(dataset_img_size).T


def validate_dataset(dataset_dir):
    success = True
    splits = ['train', 'val', 'test']

    for split in splits:
        split_dir = Path(dataset_dir) / split
        images_dir = split_dir / 'images'
        masks_dir = split_dir / 'masks'

        # Check and create directory structure
        if not images_dir.exists():
            images_dir.mkdir(parents=True, exist_ok=True)
            print(f'Created directory: {images_dir}')
            success = False

        if not masks_dir.exists():
            masks_dir.mkdir(parents=True, exist_ok=True)
            print(f'Created directory: {masks_dir}')
            success = False

        image_files = [f for f in images_dir.glob('*.jpg')]
        mask_files = [f for f in masks_dir.glob('*.jpg')]

        if len(image_files) == 0:
            success = False

        if split != 'test':
            if len(mask_files) == 0:
                success = False

        # Check file formats in images and masks folders
        if len(image_files) != len(list(images_dir.iterdir())):
            raise ValueError(f'Invalid file format in {images_dir}. Only JPG files are allowed.')

        if len(mask_files) != len(list(masks_dir.iterdir())):
            raise ValueError(f'Invalid file format in {masks_dir}. Only JPG files are allowed.')

        # Check if the length of images and masks is equal
        if split != 'test':
            if len(image_files) != len(mask_files):
                raise ValueError(f'Mismatch in the number of images and masks in {split}.')

    return success


# def combine_objects(img_id_, selected_):
#     image_path = f'airbus-ship-detection/train_v2/{img_id_}'
#     new_image_ = cv2.imread(image_path)
#     new_image_ = cv2.cvtColor(new_image_, cv2.COLOR_BGR2RGB)
#
#     if len(selected_) < 0:
#         return None
#
#     for image_id in selected_['ImageId']:
#         mask_ = get_mask(image_id, selected_)
#         original_image_path = f'airbus-ship-detection/train_v2/{image_id}'
#         original_image = cv2.imread(original_image_path)
#         original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
#         _, binary_mask = cv2.threshold(mask_, 127, 255, cv2.THRESH_BINARY)
#         object_img = get_object_by_mask(original_image, mask_)
#         new_image_ += object_img
#
#     return new_image_
#
#
# def get_object_by_mask(img, mask_):
#     # Ensure the mask and image have the same shape
#     if img.shape[:2] != mask_.shape:
#         raise ValueError("Image and mask shapes do not match")
#
#     object_img = cv2.bitwise_and(img, img, mask=mask_)
#
#     return object_img
#
#
# def generate_random_ship_masks(img_id, df_, num_masks=None):
#     if num_masks is None:
#         num_masks = random.choice(range(1, 8))
#
#     selected = df_.sample(n=num_masks)
#     selected_masks = selected['EncodedPixels'].tolist()
#     masks_valid = check_mask_intersection(selected_masks)
#
#     if masks_valid:
#         new_rows_ = pd.DataFrame([{'ImageId': img_id, 'EncodedPixels': m} for m in selected_masks])
#         new_image_ = combine_objects(img_id, selected)
#         return new_rows_, new_image_
#     else:
#         return generate_random_ship_masks(img_id, df_, num_masks)
#
#
# def check_mask_intersection(masks):
#     mask_arrays = [rle_to_mask(m) for m in masks]
#     sum_masks = np.sum(mask_arrays, axis=0)
#     return not np.any(sum_masks > 1)
#
#
# def rle_to_mask(rle):
#     if pd.notna(rle):
#         s = rle.split()
#         starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
#         starts -= 1
#         ends = starts + lengths
#         mask_ = np.zeros(768 * 768)
#         for lo, hi in zip(starts, ends):
#             mask_[lo:hi] = 1
#         return mask_.reshape(768, 768).T
#     else:
#         return np.zeros((768, 768))
#
#
# def add_boats(masks_df_, train_path_, old_train_path_):
#     clear_masks_df_ = masks_df_.dropna()
#     images_to_apply = masks_df_[pd.isna(masks_df_['EncodedPixels'])]['ImageId']
#     print('Generating fake images...')
#     for image_id in tqdm(images_to_apply):
#         new_rows, new_image = generate_random_ship_masks(image_id, clear_masks_df_)
#         clear_masks_df_ = pd.concat([clear_masks_df_, new_rows], ignore_index=True)
#         dest_path = train_path_ / image_id
#         new_image = Image.fromarray(new_image)
#         new_image.save(dest_path)
#     print('Fake images generated...')
#
#     print('Copying basic training data...')
#     for image_id in tqdm(clear_masks_df_['ImageId'].unique()):
#         source_path = old_train_path_ / image_id
#         dest_path = train_path_ / image_id
#         if not dest_path.exists():
#             shutil.copy2(source_path, dest_path)
#     print('Basic training data copied...')
#
#     return clear_masks_df_


def get_data(config):
    destination_dataset_path = Path(config['destination_dataset_path'])
    img_size = (config['img_size'], config['img_size'])
    batch_size = config['batch_size']

    is_validated = validate_dataset(destination_dataset_path)

    if not is_validated:
        dataset_img_size = (config['dataset_img_size'], config['dataset_img_size'])
        train_path = Path(config['source_dataset_path']) / 'train_v2'
        test_path = Path(config['source_dataset_path']) / 'test_v2'
        mask_path = Path(config['source_dataset_path']) / 'train_ship_segmentations_v2.csv'
        masks_df = pd.read_csv(mask_path)

        custom_augmentation = config['custom_augmentation']

        if custom_augmentation:
            raise ValueError('Custom augmentation is not implemented yet')
            # print('Custom augmentation applies...')
            # old_train_path = train_path
            # train_path = Path(config['source_dataset_path']) / 'train'
            # train_path.mkdir(parents=True, exist_ok=True)
            # masks_df = add_boats(masks_df, train_path, old_train_path)
        else:
            has_ships = len(masks_df.dropna()['ImageId'].unique())
            no_ships = int(has_ships / 0.9 - has_ships)
            masks_df = pd.concat([masks_df.dropna(), masks_df[masks_df['EncodedPixels'].isna()].sample(no_ships)],
                                 ignore_index=True)

        images = masks_df['ImageId'].unique()
        test_images = test_path.glob('*.jpg')

        random_state = 42
        train_images, val_images = train_test_split(images, test_size=0.05, random_state=random_state)

        print('Generating a valid training dataset...')
        for image_name in tqdm(train_images):
            image_path = train_path / image_name
            image_dest_path = destination_dataset_path / 'train' / 'images' / image_name
            shutil.copy2(image_path, image_dest_path)

            mask = get_mask(image_name, masks_df, dataset_img_size)
            mask_path = destination_dataset_path / 'train' / 'masks' / image_name
            mask = Image.fromarray((mask * 255).astype(np.uint8))
            mask.save(mask_path)
        print('Valid training dataset generated...')

        print('Generating a valid validation dataset...')
        for image_name in tqdm(val_images):
            image_path = train_path / image_name
            image_dest_path = destination_dataset_path / 'val' / 'images' / image_name
            shutil.copy2(image_path, image_dest_path)

            mask = get_mask(image_name, masks_df, dataset_img_size)
            mask_path = destination_dataset_path / 'val' / 'masks' / image_name
            mask = Image.fromarray((mask * 255).astype(np.uint8))
            mask.save(mask_path)
        print('Valid validation dataset generated...')

        print('Generating a valid test dataset...')
        for image_path in tqdm(test_images):
            image_name = image_path.name
            image_dest_path = destination_dataset_path / 'test' / 'images' / image_name
            shutil.copy2(image_path, image_dest_path)
        print('Valid test dataset generated...')

        is_validated_2 = validate_dataset(destination_dataset_path)

        if not is_validated_2:
            raise 'The dataset has not been validated twice. Please investigate.'

    train_gen = SegmentationDataGenerator(directory=destination_dataset_path,
                                          batch_size=batch_size,
                                          subset='train',
                                          img_size=img_size)
    val_gen = SegmentationDataGenerator(directory=destination_dataset_path,
                                        batch_size=batch_size,
                                        subset='val',
                                        augmentation=False,
                                        img_size=img_size)
    test_gen = SegmentationDataGenerator(directory=destination_dataset_path,
                                         batch_size=batch_size,
                                         subset='test',
                                         augmentation=False,
                                         img_size=img_size)

    return train_gen, val_gen, test_gen
