import pkg_resources
from astropy.io.fits import getdata
from .image_process import fits2df
from .features import Stamping


class CNN():
    def __init__(self, model):
        # initialize CNN model
        self.model = model

    @classmethod
    def load(cls):
        from keras.models import load_model
        # define parameters
        params = {
            'datapath': pkg_resources.resource_filename('gTranRec', 'data'),
        }
        # define model path
        cnn_path = '/'.join([params['datapath'], 'cnn.m'])
        # load model
        print("Loading CNN...")
        model = load_model(cnn_path)
        # load model with class method
        print("Done!")  
        return cls(model)
    
    def image_predict(self, filename):
        parameters = {
            'diffphoto': 'PHOTOMETRY_DIFF',
            'diffimage': 'DIFFERENCE',
        }
        # load DIFFERENCE image
        self.image_data = getdata(filename, parameters['diffimage'])
        # load PHOTOMETRY_DIFFERENCE
        self.photo_df = fits2df(filename, parameters['diffphoto'])
        # get stamps from the PHOTOMETRY_DIFFERENCE
        stamps_obj = Stamping.create_stamps(self.image_data, self.photo_df)
        # clean data
        stamps_obj.clean_stamps()
        # normalize data
        stamps_obj.norm_stamps()
        # display progress if verbose=True
        print("Making prediction...")
        # CNN predicts
        self.cnn_score = self.model.predict(stamps_obj.norm_stamps.reshape(-1,21,21,1))
        # get the score of being real
        self.cnn_score = self.cnn_score[:,1]
        # add score column to the given pd.DataFrame
        self.photo_df['gtr_cnn'] = self.cnn_score
        print("Done!")  




