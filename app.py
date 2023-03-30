from upcunet import *
import oneflow
import numpy as np
import gradio as gr
from time import time

class App:
    device = None
    models = {}

    def __init__(self):
        self.device = 'cuda' if oneflow.cuda.is_available() else 'cpu'
        print(f'Using device {self.device}')

        # read weights folder
        weights = os.listdir('weights')
        for weight in weights:
            scale = int(weight[2:3])
            self.models[weight] = RealWaifuUpScaler(scale, f'weights/{weight}', False, self.device)
            print(f'Loaded model {weight}')
    
    def get_models(self):
        return list(self.models.keys())
    
    def upscale(self, input, model, tile):
        if model not in self.models:
            return None
        
        input = np.array(input)
        print(f'Upscaling image with model {model} and tile size {tile}')
        t0 = time()
        result = self.models[model](input, tile)
        t1 = time()
        print(f'Upscaling complete. Completion time: {t1 - t0}. Upscaled: {input.shape} -> {result.shape}.')
        return result
    
    def run(self):
        input = gr.Image(type='pil', label='Original Image')

        model = gr.Dropdown(
            self.get_models(),
            label='Model',
            value=self.get_models()[0]
        )

        tile = gr.Slider(
            minimum=0,
            maximum=4,
            value=2,
            step=1,
            label='Tile Size'
        )

        inputs = [input, model, tile]
        outputs = 'image'

        interface = gr.Interface(
            self.upscale,
            inputs,
            outputs,
            allow_flagging='never'
        )

        interface.launch()

if __name__ == '__main__':
    app = App()
    app.run()