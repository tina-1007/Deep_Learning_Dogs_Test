import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from PIL import Image

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torchvision.models as models

import numpy as np
#from tensorflow.keras.utils.data_utils import get_file
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles


#set url
model_file_name = './app/models/model.t7'

classes = ["Pug","French dog","Shar Pei"]
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))
device = torch.device("cpu")

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    model = models.resnet50(pretrained = False)
    model.fc = nn.Linear(model.fc.in_features,10)
    model = model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file_name,map_location='cpu'))

    return model


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
#     img = open_image(BytesIO(img_bytes))   
#     prediction = learn.predict(img)[0]
#     image = tf.keras.preprocessing.image.load_img( path, target_size=(img_size, img_size))
#     input_arr = keras.preprocessing.image.img_to_array(image)
#     input_arr = np.array([input_arr])  # Convert single image to a batch.
#     predictions = learn.predict(input_arr) 

    img = Image.open(BytesIO(img_bytes))
    img = img.convert('RGB')

    preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    process_img = preprocess(img)
    process_img = process_img[None,:,:]

    learn.eval()
    process_img = process_img.to(device)
    cls_prob = learn(process_img)
    cls_pred = cls_prob.max(dim=1)[1].numpy()
    return JSONResponse({'result': classes[cls_pred[0]]})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5001, log_level="info")
