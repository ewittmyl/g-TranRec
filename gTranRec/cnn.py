import pkg_resources
from astropy.io.fits import getdata
from .image_process import fits2df
from .data_process import Stamping


class CNN():
    """
    CNN model class object.
    """
    def __init__(self, model):
        # define CNN model
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
        model = load_model(cnn_path)

        return cls(model)
    
    def image_predict(self, filename):
        parameters = {
            'diffphoto': 'PHOTOMETRY_DIFF',
            'diffimage': 'DIFFERENCE',
        }
        # load image
        self.image_data = getdata(filename, parameters['diffimage'])
        # load photometry table
        self.photo_df = fits2df(filename, parameters['diffphoto'])
        # get stamps
        stamps_obj = Stamping.create_stamps(self.image_data, self.photo_df)
        stamps_obj.clean_stamps()
        stamps_obj.norm_stamps()
        # calculate score
        self.cnn_score = self.model.predict(stamps_obj.norm_stamps.reshape(-1,21,21,1))
        self.cnn_score = self.cnn_score[:,1]
        self.photo_df['gtr_cnn'] = self.cnn_score



