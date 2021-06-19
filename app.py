import joblib
from util import replace_contractions

import torch
import torch.nn.functional as F
from model import CNN_Text

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from schema import SentimentResponse,SentimentRequest

import uvicorn
from fastapi import FastAPI


maxlen = 750
model_weight_path = 'src/resources/text_cnn_v1.pth'
tokenizer_path = 'src/resources/tokenizer.pkl'
label_enc_path = 'src/resources/label_enc.pkl'
embedding_matrix_path  = 'src/resources/embedding_mat.pkl'

le = joblib.load(label_enc_path)
tokenizer = joblib.load(tokenizer_path)
embedding_matrix = joblib.load(embedding_matrix_path)

# Create the app object
app = FastAPI()

@app.on_event("startup")
def startup_event():
    global model
    model = CNN_Text(embedding_matrix)
    model.load_state_dict(torch.load(model_weight_path,map_location='cpu'))
    model.eval()

@app.get('/')
def dummy():
    return {'message': 'Hello, World'}

@app.post("/predict",response_model=SentimentResponse)
async def predict_single(request: SentimentRequest):
    inp = request.dict()
    x = inp['text']
    x = replace_contractions(x.lower())
    x = tokenizer.texts_to_sequences([x])
    x = pad_sequences(x, maxlen=maxlen)
    x = torch.tensor(x, dtype=torch.long)#.cuda()

    #Predict
    pred = model(x).detach()
    pred = F.softmax(pred,dim=1).cpu().numpy()
    pred_class = pred.argmax(axis=1)

    predicted_sentiment = le.classes_[pred_class][0]
    
    confidence = pred.max(axis=1)
    
    return SentimentResponse(
        sentiment = predicted_sentiment,
        confidence= confidence
        )

#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
