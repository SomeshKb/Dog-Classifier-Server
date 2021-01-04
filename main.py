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


app = Flask(__name__)
CORS(app)
model = models.resnet50(pretrained = True)



num_classes = 133

model.fc = nn.Linear(2048, num_classes, bias = True)
model.load_state_dict(torch.load('classifier.pt', map_location=torch.device('cpu')))
model.eval()

class_names = ['Affenpinscher',
 'Afghan hound',
 'Airedale terrier',
 'Akita',
 'Alaskan malamute',
 'American eskimo dog',
 'American foxhound',
 'American staffordshire terrier',
 'American water spaniel',
 'Anatolian shepherd dog',
 'Australian cattle dog',
 'Australian shepherd',
 'Australian terrier',
 'Basenji',
 'Basset hound',
 'Beagle',
 'Bearded collie',
 'Beauceron',
 'Bedlington terrier',
 'Belgian malinois',
 'Belgian sheepdog',
 'Belgian tervuren',
 'Bernese mountain dog',
 'Bichon frise',
 'Black and tan coonhound',
 'Black russian terrier',
 'Bloodhound',
 'Bluetick coonhound',
 'Border collie',
 'Border terrier',
 'Borzoi',
 'Boston terrier',
 'Bouvier des flandres',
 'Boxer',
 'Boykin spaniel',
 'Briard',
 'Brittany',
 'Brussels griffon',
 'Bull terrier',
 'Bulldog',
 'Bullmastiff',
 'Cairn terrier',
 'Canaan dog',
 'Cane corso',
 'Cardigan welsh corgi',
 'Cavalier king charles spaniel',
 'Chesapeake bay retriever',
 'Chihuahua',
 'Chinese crested',
 'Chinese shar-pei',
 'Chow chow',
 'Clumber spaniel',
 'Cocker spaniel',
 'Collie',
 'Curly-coated retriever',
 'Dachshund',
 'Dalmatian',
 'Dandie dinmont terrier',
 'Doberman pinscher',
 'Dogue de bordeaux',
 'English cocker spaniel',
 'English setter',
 'English springer spaniel',
 'English toy spaniel',
 'Entlebucher mountain dog',
 'Field spaniel',
 'Finnish spitz',
 'Flat-coated retriever',
 'French bulldog',
 'German pinscher',
 'German shepherd dog',
 'German shorthaired pointer',
 'German wirehaired pointer',
 'Giant schnauzer',
 'Glen of imaal terrier',
 'Golden retriever',
 'Gordon setter',
 'Great dane',
 'Great pyrenees',
 'Greater swiss mountain dog',
 'Greyhound',
 'Havanese',
 'Ibizan hound',
 'Icelandic sheepdog',
 'Irish red and white setter',
 'Irish setter',
 'Irish terrier',
 'Irish water spaniel',
 'Irish wolfhound',
 'Italian greyhound',
 'Japanese chin',
 'Keeshond',
 'Kerry blue terrier',
 'Komondor',
 'Kuvasz',
 'Labrador retriever',
 'Lakeland terrier',
 'Leonberger',
 'Lhasa apso',
 'Lowchen',
 'Maltese',
 'Manchester terrier',
 'Mastiff',
 'Miniature schnauzer',
 'Neapolitan mastiff',
 'Newfoundland',
 'Norfolk terrier',
 'Norwegian buhund',
 'Norwegian elkhound',
 'Norwegian lundehund',
 'Norwich terrier',
 'Nova scotia duck tolling retriever',
 'Old english sheepdog',
 'Otterhound',
 'Papillon',
 'Parson russell terrier',
 'Pekingese',
 'Pembroke welsh corgi',
 'Petit basset griffon vendeen',
 'Pharaoh hound',
 'Plott',
 'Pointer',
 'Pomeranian',
 'Poodle',
 'Portuguese water dog',
 'Saint bernard',
 'Silky terrier',
 'Smooth fox terrier',
 'Tibetan mastiff',
 'Welsh springer spaniel',
 'Wirehaired pointing griffon',
 'Xoloitzcuintli',
 'Yorkshire terrier']


def load_image(image_bytes):
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])
                            
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image)[:3,:,:].unsqueeze(0)


def predict_breed_transfer(image_bytes):
    # load the image and return the predicted breed
    image = load_image(image_bytes)
    idx = torch.argmax(model(image))
    print(class_names[idx])
    return class_names[idx]
    
@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        print(file)
        img_bytes = file.read()
        class_name = predict_breed_transfer(image_bytes=img_bytes)
        return jsonify({'class_name': class_name})

@app.route('/ping', methods=['GET'])
def ping():
    return "Pinging Model!!"

