import os
import glob
import datetime

import utils  
from utils import Image, date_diff_in_Seconds

def main():
    '''OG Function'''
    for path in glob.glob(os.getcwd()+'/Dense_Haze_NTIRE19/hazy/*.png'):
        my_img = Image(path)
        print('Reconstructing image...')
        start_time = datetime.datetime.now()
        my_img.reconstructImage()
        end_time = datetime.datetime.now()
        print('Image reconstructed in {} seconds'.format(date_diff_in_Seconds(end_time, start_time)) )
        break
    return
    
    '''Current Testing'''
    # path = os.getcwd()+'/Dense_Haze_NTIRE19/hazy/01_hazy.png'
    # my_img = Image(path)
    # print(my_img.transmissionRefinement([5,7]))

if __name__ == "__main__":
    main()