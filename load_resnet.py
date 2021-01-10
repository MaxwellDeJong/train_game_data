import pickle
import torchvision.models as models
from resnet import resnet50, resnet50_dropout

def load_resnet50(use_custom=False, dropout=False, dirname='ski-race'):
    
    balanced_dir = 'E:/steep_training/' + dirname + '/balanced/'
    balanced_one_hot_filename = balanced_dir + 'one_hot_dict.pkl'
    
    with open(balanced_one_hot_filename, 'rb') as handle:
        balanced_one_hot = pickle.load(handle)
        
    num_classes = len(balanced_one_hot)
    print('number of classes: ', num_classes)
    kwarg = {'num_classes': num_classes}
    
    if use_custom:
        if not dropout:
            model = resnet50(num_classes)
        else:
            model = resnet50_dropout(num_classes)
    else:
        model = models.resnet50(kwarg)
    
    return model