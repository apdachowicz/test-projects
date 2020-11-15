import numpy as np
import ndjson
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skimage import draw


class image_builder:
    """
    Class for processing ndjson input files and converting to .png data files.
    TODO: convert ndjson data to binary images
    TODO: pipeline for saving images to a new image data directory
    """

    def __init__(self, input_path, output_path):
        """
        Init.
        :param input_path: path to ndjson data file.
        :param output_path: target path for output data images.
        """
        self.input_path = input_path
        self.output_path = output_path

    def get_outputs(self):
        """
        Convert ndjson data to binary .pngs and save to target output directory.
        :return: None
        """

        # TODO: confirm input exists and create output dir if needed

        # TODO: read in data

        # TODO: convert each row of data to image

        # TODO: save output data

        return
