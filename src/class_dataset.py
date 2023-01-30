"""
Data generator
"""
import glob
import os
import cv2 as cv

AVAILABLE_EXTENSIONS = ["*.jpg", "*.png", "*.jpeg"]


class Dataset:
    """
    Get images for detection
    """

    def __init__(self, dataSource, **kwargs):
        """
        :param dataSource:
            'folder' : extract data from folder
                kwargs['folder'] - folder with images
                kwargs['qnty'] - qnty of images from specified folder. If == 0 or not specify - all images
            'file_list' : extract data from file list
                kwargs['file_list'] - list with file pathes
                kwargs['qnty'] - qnty of images from specified folder. If == 0 or not specify - all images
        :param kwargs:
        """
        self.dataSource = dataSource
        if dataSource == "folder":
            self.imgFolder = kwargs["folder"]
            self.imgList = []
            for extension in AVAILABLE_EXTENSIONS:
                self.imgList += glob.glob(os.path.join(self.imgFolder, extension))
            if (
                kwargs.get("qnty", -1) != -1
                and (not kwargs["qnty"] == None)
                and kwargs["qnty"] > 0
            ):
                self.imgList = self.imgList[: kwargs["qnty"]]
            self.imgList = sorted(self.imgList)
        elif dataSource == "file_list":
            self.imgList = kwargs["file_list"]
            if (
                kwargs.get("qnty", -1) != -1
                and (not kwargs["qnty"] == None)
                and kwargs["qnty"] > 0
            ):
                self.imgList = self.imgList[: kwargs["qnty"]]
            self.imgList = sorted(self.imgList)
        else:
            raise ValueError
        return

    def __getitem__(self, index):
        """
        Return np.array with next image
        :param item:
        :return:
        """
        imgFile = self.imgList[index]
        image = cv.imread(imgFile)
        if self.dataSource == "folder" or self.dataSource == "file_list":
            imgSource = os.path.basename(imgFile)
        else:
            imgSource = None
        return image, imgSource

    def __len__(self):
        return len(self.imgList)
