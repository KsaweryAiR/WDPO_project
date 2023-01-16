import json
from pathlib import Path
from typing import Dict
import numpy as np
import click
import cv2
from tqdm import tqdm

def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #print(img_path)
    #imGMIN = cv2.resize(img, (960, 540))
    #cv2.imshow('oryginal', imGMIN)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #define red
    def red():
        #FILTR
        median = cv2.medianBlur(img_hsv, 23)
        #ZAKRES
        lower_red1 = np.array([172, 80, 80])#([172, 80, 110])
        upper_red1 = np.array([180, 255, 255])
        lower_red2 = np.array([0, 190, 45])#([0, 50, 45])
        upper_red2 = np.array([9, 255, 255])

        mask1 = cv2.inRange(median, lower_red1, upper_red1)
        mask2 = cv2.inRange(median, lower_red2, upper_red2)
        # Połączenie dwóch zakresów
        mask3 = cv2.bitwise_or(mask1,mask2)

        # filtry
        kernel = np.ones((30, 30), np.uint8)
        kernel2 = np.ones((15, 15), np.uint8)
        opening = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
        # szukanie konturów
        countours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # szukanie konturów

        return len(countours)
    def green():
        # FILTR
        median = cv2.medianBlur(img_hsv, 23)
        # ZAKRES
        lower_green1 = np.array([27, 115, 42])#([27, 115, 25])
        upper_green1 = np.array([86, 255, 200])
        mask1 = cv2.inRange(median, lower_green1, upper_green1)

        # filtry
        kernel = np.ones((30, 30), np.uint8)
        kernel2 = np.ones((15, 15), np.uint8)
        opening = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)

        # szukanie konturów
        countours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # szukanie konturów

        return len(countours)
    def yellow():
        # FILTR
        median = cv2.medianBlur(img_hsv, 23)
        # ZAKRES
        lower_yellow1 = np.array([23, 110, 170])
        upper_yellow1 = np.array([30, 255, 255])
        mask1 = cv2.inRange(median, lower_yellow1, upper_yellow1)

        #filtry
        kernel = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((15, 15), np.uint8)
        opening = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)

        # szukanie konturów
        countours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # szukanie konturów

        return len(countours)
    def purple():
        # FILTR
        median = cv2.medianBlur(img_hsv, 23)
        # ZAKRES
        lower_purple1 = np.array([144, 38, 0]) #38
        upper_purple1 = np.array([173, 255, 177])

        mask1 = cv2.inRange(median, lower_purple1, upper_purple1)


        #filtry
        kernel = np.ones((23, 23), np.uint8)
        kernel2 = np.ones((15, 15), np.uint8)
        opening = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)

        # szukanie konturów
        countours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # szukanie konturów

        return len(countours)

    #TODO: Implement detection method.
    
    red = red()
    yellow = yellow()
    green = green()
    purple = purple()
    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
