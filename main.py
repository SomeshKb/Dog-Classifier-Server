import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import glob
import torch
import torchvision.models as models
import torch.nn as nn
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)
model = models.resnet50(pretrained = True)



num_classes = 133

model.fc = nn.Linear(2048, num_classes, bias = True)
model.load_state_dict(torch.load('classifier.pt', map_location=torch.device('cpu')))
model.eval()


def resnet50_predict(pil_image):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    # Image Resize to 256
    resnet50 = models.resnet50(pretrained=True)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    image_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean,std)])
    image_tensor = image_transforms(pil_image)
    image_tensor.unsqueeze_(0)
    resnet50.eval()
    output = resnet50(image_tensor)
    _,classes= torch.max(output,dim=1)
    return classes.item() # predicted class index

def dog_detector(pil_image):
    ## TODO: Complete the function.
    class_dog=resnet50_predict(pil_image)
    return class_dog >= 151 and class_dog <=268 # true/false

def transform_image(image):
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])
                            
    return transform(image)[:3,:,:].unsqueeze(0)

def predict_breed_transfer(image):
    # load the image and return the predicted breed
    transformed_image = transform_image(image)
    idx = torch.argmax(model(transformed_image))
    with open("dog_breeds.json") as f:
        class_names = json.loads(f.read())
        return class_names[idx]
    
@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        if (dog_detector(image)):
            class_name = predict_breed_transfer(image)
            return jsonify({'class_name': class_name})
        else:
            return jsonify({'class_name': "Not a dog image"})

@app.route('/health', methods=['GET'])
def ping():
    return jsonify({'heath': "healthy"})

