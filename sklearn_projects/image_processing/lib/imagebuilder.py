import os
import numpy as np
import ndjson
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skimage import draw, io
from tqdm import tqdm_notebook as tqdm
import warnings


class ImageBuilder:
    """
    Class for processing ndjson input files and converting to .png data files.
    Assumes that input data has the form of the preprocessed quickdraw data.
    See: https://github.com/googlecreativelab/quickdraw-dataset#preprocessed-dataset

    Data will be extracted to output_path/images/.

    Metadata will be extracted to output_path/meta/.

    USAGE:

    [1] ib = ImageBuilder(input_path, output_path)

    [2] ib.build_images()
    """

    def __init__(self,
                 input_path: str,
                 output_path: str,
                 save_metadata: bool = True,
                 output_format: str = None,
                 max_images: int = None,
                 ignore_unrecognized_images: bool = True):
        """
        :param input_path: path to ndjson data file.
        :param output_path: target path for output data images. This will be created if it does not exist.
        :param save_metadata: If true, will save metadata as output_path/meta/meta.csv.
        :param output_format: Output image format. Defaults to .png.
        :param max_images: If specified, only the first max_images images will be saved.
        :param ignore_unrecognized_images: If True, any images that were not recognized in the source data
        will be dropped.
        """
        if output_format is None:
            output_format = '.png'

        self.input_path = input_path
        "Path to ndjson data file."
        self.output_path = output_path
        "Target path for output data images. This will be created if it does not exist."
        self.save_metadata = save_metadata
        "If true, will save metadata as output_path/meta/meta.csv."
        self.output_format = output_format
        "Output image format. Defaults to .png."
        self.max_images = max_images
        "Max. number of images to extract."
        self.ignore_unrecognized_images = ignore_unrecognized_images
        "If True, any images that were not recognized in the source data will be dropped."
        self.input_data = None
        "The input data read from disk. Data is read in self.read_data()."
        self.input_metadata = None
        "Metadata for data read from disk. Data is read in self.read_data()."
        self.total_images_extracted = 0
        "Total number of images extracted."

        assert os.path.isfile(self.input_path)
        if os.path.isdir(self.output_path):
            print('WARNING! Output path {} already exists. '.format(self.output_path)
                  + '\n .... Data in this directory will be overwritten.')

    def build_images(self):
        """
        Build and save images and metadata from input data.

        This is the main public method for this class.

        :return: None
        """
        # read in data and get metadata
        self.read_data()
        # build and save output data images
        self.get_outputs()

        return

    def get_outputs(self):
        """
        Convert ndjson data to binary images and save to target output directory.

        :return: None
        """
        assert self.input_data is not None
        assert os.path.isdir(self.output_path)
        warnings.simplefilter("ignore", category=UserWarning)

        if not os.path.isdir(self.output_path + '/images'):
            os.makedirs(self.output_path + '/images')

        # only take top images if specified:
        if self.max_images is not None:
            self.input_data = self.input_data[:self.max_images]

        # if we are ignoring unrecognized images, remind the user:
        print('Ignoring unrecognized images: {}'.format(self.ignore_unrecognized_images))

        # convert each row of data to image and save
        for sample_id in tqdm(range(len(self.input_data)),
                              desc='Extracting and saving sample data...'):

            # check if sample was recognized or not
            was_recognized = self.input_metadata[self.input_metadata['sample_id']
                                                 == sample_id]['recognized'].values.tolist()[0]
            # if it was not recognized and we are ignoring unrecognized images, skip this one.
            if not ((not was_recognized) and self.ignore_unrecognized_images):
                sample = self.input_data[sample_id]
                drawing_sample = sample['drawing']
                drawing_image = np.zeros([256, 256])  # blank image
                for stroke_num, stroke in enumerate(drawing_sample):
                    x = stroke[0]
                    y = stroke[1]
                    assert len(x) == len(y)
                    for pixel in range(len(x) - 1):
                        rr, cc = draw.line(y[pixel], x[pixel],
                                           y[pixel + 1], x[pixel + 1])
                        drawing_image[rr, cc] = 255
                # save output data
                io.imsave(self.output_path + '/images/{}{}'.format(sample_id, self.output_format),
                          drawing_image.astype(np.uint8))
                self.total_images_extracted += 1
        print('Saved output data to {}'.format(self.output_path))
        print('Output data format: {}'.format(self.output_format))
        print('Total images extracted: {}.'.format(self.total_images_extracted))

        return

    def read_data(self):
        """
        Create output directory if needed, read in data, and compute metadata table.

        :return: None
        """
        # create output dir if needed
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)

        # read in data
        with open(self.input_path) as f:
            self.input_data = ndjson.load(f)

        # create metadata objects
        metadata_dict = {
            'sample_id': [],
        }
        for key in ['key_id', 'word', 'countrycode', 'recognized', 'num_strokes']:
            metadata_dict[key] = []
        for sample_id in range(len(self.input_data)):
            sample = self.input_data[sample_id]
            metadata_dict['sample_id'] += [sample_id]
            for key in ['key_id', 'word', 'countrycode', 'recognized']:
                metadata_dict[key] += [sample[key]]
            metadata_dict['num_strokes'] += [len(sample['drawing'])]
        self.input_metadata = pd.DataFrame.from_dict(metadata_dict)
        if self.max_images is not None:
            print('Image samples: {}. Extracting the first {}.'.format(len(self.input_metadata),
                                                                       self.max_images))
        else:
            print('Image samples: {}.'.format(len(self.input_metadata)))

        # save metadata if asked
        if self.save_metadata:
            if not os.path.isdir(self.output_path + '/meta'):
                os.makedirs(self.output_path + '/meta')
            self.input_metadata.to_csv(self.output_path + '/meta/meta.csv')
            print('Saved image metadata to {}.'.format(self.output_path + '/meta/meta.csv'))

        return
