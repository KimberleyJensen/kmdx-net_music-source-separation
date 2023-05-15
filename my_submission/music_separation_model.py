import numpy as np
import torch
import time
from my_submission.models import get_models
from demucs.htdemucs import HTDemucs
from demucs.hdemucs import HDemucs
from demucs import pretrained
from demucs.apply import apply_model
import onnxruntime as ort

device = torch.device('cuda')

model_path = 'my_submission/model_weights'

# enable onnx acceleration 
use_onnx = True    

# model folder should contain onnx files if use_onnx==True, otherwise torch state_dicts
model_name = 'TFC_TDF_UNet_MDX21'   

# set to None if you only want to use TFC_TDF_UNet (without blending)
demucs_name = 'demucs_bass.ckpt', 'demucs_mmi.ckpt', 'demucs_other.ckpt', 'demucs_vocals.ckpt'

# in the order of MusicSeparationModel.instruments
blend_weights = [0.08, 0.08, 0.4, 0.88]     


class MusicSeparationModel:
    
    def __init__(self):     
        
        self.use_onnx = use_onnx
        self.blend = demucs_name is not None
        
        self.models = get_models(f'{model_path}/{model_name}', use_onnx)
                
        if self.blend:
            self.blend_weights = blend_weights
            self.demucs_instruments = ["drums", "bass", "other", "vocals"]
            self.demucs1 = HTDemucs(sources=self.demucs_instruments, bottom_channels=512, dconv_mode=3, segment=7.8).eval()
            self.demucs1.load_state_dict(torch.load(f'my_submission/model_weights/demucs_drums.ckpt')['state'], strict=False)  
            self.demucs2 = HTDemucs(sources=self.demucs_instruments, bottom_channels=512, dconv_mode=3, segment=7.8).eval()
            self.demucs2.load_state_dict(torch.load(f'my_submission/model_weights/demucs_bass.ckpt')['state'], strict=False)
            self.demucs3 = HTDemucs(sources=self.demucs_instruments, bottom_channels=512, dconv_mode=3, segment=7.8).eval()
            self.demucs3.load_state_dict(torch.load(f'my_submission/model_weights/demucs_other.ckpt')['state'], strict=False)
            self.demucs4 = HTDemucs(sources=self.demucs_instruments, bottom_channels=512, dconv_mode=3, segment=7.7).eval()
            self.demucs4.load_state_dict(torch.load(f'my_submission/model_weights/demucs_vocals.ckpt')['state'], strict=False) 
            self.demucs5 = HDemucs(sources=self.demucs_instruments, channels=48, segment=44).eval()
            self.demucs5.load_state_dict(torch.load(f'my_submission/model_weights/demucs_mmi.ckpt')['state'], strict=False)
            self.demucs6 = HDemucs(sources=self.demucs_instruments, channels=48, hybrid_old=True, cac=False).eval()
            self.demucs6.load_state_dict(torch.load(f'my_submission/model_weights/demucstracktwo.ckpt')['state'], strict=False)
     
        
    @property
    def instruments(self):
        """ DO NOT CHANGE """
        return ['bass', 'drums', 'other', 'vocals']

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def separate_music_file(self, mixed_sound_array, sample_rate):
        """
        Implements the sound separation for a single sound file
        Inputs: Outputs from soundfile.read('mixture.wav')
            mixed_sound_array
            sample_rate

        Outputs:
            separated_music_arrays: Dictionary numpy array of each separated instrument
            output_sample_rates: Dictionary of sample rates separated sequence
        """
        
        mixture = torch.tensor(mixed_sound_array.T, dtype=torch.float32)
        sources = self.demix(mixture)
        
        separated_music_arrays = {}
        output_sample_rates = {}
        for instrument in self.instruments:
            separated_music_arrays[instrument] = sources[instrument].numpy().T
            output_sample_rates[instrument] = sample_rate
            
        return separated_music_arrays, output_sample_rates
    
    
    def demix(self, mix):
        base_out = self.demix_base(mix)
        if self.blend:
            demucs_out = self.demix_demucs(mix)
            sources = {i: base_out[i] * b + demucs_out[i] * (1-b) for i, b in zip(self.instruments, self.blend_weights)} 
        else:
            sources = base_out
        return sources
    
    
    def demix_base(self, mix):          
        
        device = torch.device('cuda')
        sources = []
        n_sample = mix.shape[1]

        # Process vocals model separately and store its output
        vocals_output = None
        for model in self.models:
            if model.target_name == "vocals":
                overlap = model.overlap    
                chunk_size = model.chunk_size
                gen_size = chunk_size - 2*overlap
                pad_size = gen_size - n_sample%gen_size
                mix_padded = torch.cat([torch.zeros(2, overlap), mix, torch.zeros(2, pad_size+overlap)], 1)
                
                ort_session = ort.InferenceSession(f'{model_path}/{model_name}/{model.target_name}.onnx')
                
                # process one chunk at a time (batch_size=1)
                demixed_chunks = []           
                i = 0
                while i < n_sample + pad_size:                
                    chunk = mix_padded[:, i:i+chunk_size]
                    x = model.stft(chunk.unsqueeze(0).to(device))                
                    with torch.no_grad():
                        x = torch.tensor(ort_session.run(None, {'input': x.cpu().numpy()})[0]) 
                    x = model.stft.inverse(x).squeeze(0)
                    x = x[...,overlap:-overlap]
                    demixed_chunks.append(x)
                    i += gen_size

                vocals_output = torch.cat(demixed_chunks, -1)[...,:-pad_size].cpu()
                break

        # Subtract vocals output from the input mix for the remaining models
        mix_minus_vocals = mix - vocals_output

        for model in self.models:
            
            if model.target_name != "vocals":
                overlap = model.overlap    
                chunk_size = model.chunk_size
                gen_size = chunk_size - 2*overlap
                pad_size = gen_size - n_sample%gen_size
                mix_padded = torch.cat([torch.zeros(2, overlap), mix_minus_vocals, torch.zeros(2, pad_size+overlap)], 1)

                model.eval()
                model.to(device)
            
                # process one chunk at a time (batch_size=1)
                demixed_chunks = []           
                i = 0
                while i < n_sample + pad_size:                
                    chunk = mix_padded[:, i:i+chunk_size]
                    x = model.stft(chunk.unsqueeze(0).to(device))                
                    with torch.no_grad():
                        x = model(x)                       
                    x = model.stft.inverse(x).squeeze(0)
                    x = x[...,overlap:-overlap]
                    demixed_chunks.append(x)
                    i += gen_size

                demixed_wave = torch.cat(demixed_chunks, -1)[...,:-pad_size]                 
                sources.append(demixed_wave.cpu())

        # Add vocals output to the sources list
        sources.insert(0, vocals_output)

        return {model.target_name: x for model, x in zip(self.models, sources)}

    
    
    def demix_demucs(self, mix):
        b = time.time()
        mono = mix.mean(0)
        mean = mono.mean()
        std = mono.std()        
        mix = (mix - mean) / std
        
        with torch.no_grad():
            sources4 = apply_model(self.demucs4, mix[None], split=True, shifts=0, overlap=0.85, device=torch.device('cuda'))[0]
            vocals = sources4[self.demucs_instruments.index('vocals')]
            sources5 = apply_model(self.demucs5, mix[None], split=True, shifts=0, overlap=0.85, device=torch.device('cuda'))[0]
            sources6 = apply_model(self.demucs6, mix[None], split=True, shifts=0, overlap=0.8, device=torch.device('cuda'))[0]
            sources7 = apply_model(self.demucs2, mix[None], split=True, shifts=0, overlap=0.8, device=torch.device('cuda'))[0]
            vocalss = sources5[self.demucs_instruments.index('vocals')]
            mixture_1 = mix - (0.8 * vocals + 0.2 * vocalss)
            sources1 = apply_model(self.demucs1, mixture_1[None], split=True, shifts=0, overlap=0.85, device=torch.device('cuda'))[0]
            drums = sources1[self.demucs_instruments.index('drums')]
            drumss = sources5[self.demucs_instruments.index('drums')]
            mixture_3 = mixture_1 - (0.7 * drums + 0.3 * drumss)
            sources2 = apply_model(self.demucs3, mixture_3[None], split=True, shifts=0, overlap=0.85, device=torch.device('cuda'))[0]
            bass = sources2[self.demucs_instruments.index('bass')]
            mixture_4 = mixture_3 - bass
            sources = np.stack([sources1[0] * 0.60 + sources5[0] * 0.26 + sources6[0] * 0.14, sources2[1] * 0.89 + sources5[1] * 0.11, mixture_4 * 0.68 + sources5[2] * 0.04 + sources7[2] * 0.28, sources4[3] * 0.73 + sources5[3] * 0.21 + sources6[3] * 0.06])




        sources = torch.tensor(sources) 
        sources = sources * std + mean
            
        
        return {i: x for i, x in zip(self.demucs_instruments, sources)}
