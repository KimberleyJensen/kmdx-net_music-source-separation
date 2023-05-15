import os
import torch
import torch.nn as nn
import numpy as np

dim_s = 4
dim_c = 4

class STFT:
    def __init__(self, n_fft, hop_length, dim_f):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length=n_fft, periodic=True)        
        self.dim_f = dim_f
    
    def __call__(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0,3,1,2])
        x = x.reshape([*batch_dims,c,2,-1,x.shape[-1]]).reshape([*batch_dims,c*2,-1,x.shape[-1]])
        return x[...,:self.dim_f,:]

    def inverse(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c,f,t = x.shape[-3:]
        n = self.n_fft//2+1
        f_pad = torch.zeros([*batch_dims,c,n-f,t]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims,c//2,2,n,t]).reshape([-1,2,n,t])
        x = x.permute([0,2,3,1])
        x = x.contiguous()
        t_complex = torch.view_as_complex(x)
        x = torch.istft(t_complex, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True)
        x = x.reshape([*batch_dims,2,-1])
        return x


class Conv_TDF(nn.Module):
    def __init__(self, c, l, f, k, bn, bias=True):
        
        super(Conv_TDF, self).__init__()
        
        self.use_tdf = bn is not None
   
        self.H = nn.ModuleList()
        for i in range(l):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=k, stride=1, padding=k//2),
                    nn.GroupNorm(2, c),
                    nn.ReLU(),
                )
            )

        if self.use_tdf:
            if bn==0:
                self.tdf = nn.Sequential(
                    nn.Linear(f,f, bias=bias),
                    nn.GroupNorm(2, c),
                    nn.ReLU()
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f,f//bn, bias=bias),
                    nn.GroupNorm(2, c),
                    nn.ReLU(),
                    nn.Linear(f//bn,f, bias=bias),
                    nn.GroupNorm(2, c),
                    nn.ReLU()
                )
                       
    def forward(self, x):
        for h in self.H:
            x = h(x)
        
        return x + self.tdf(x) if self.use_tdf else x

class TFCC(nn.Module):
    def __init__(self, c, l, k):
        super(TFCC, self).__init__()

        self.H = nn.ModuleList()
        for i in range(l):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=k, stride=1, padding=k // 2),
                    nn.GroupNorm(4, c),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for h in self.H:
            x = h(x)
        return x


class DenseTFCC(nn.Module):
    def __init__(self, c, l, k):
        super(DenseTFCC, self).__init__()

        self.conv = nn.ModuleList()
        for i in range(l):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=k, stride=1, padding=k // 2),
                    nn.GroupNorm(4, c),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for layer in self.conv[:-1]:
            x = torch.cat([layer(x), x], 1)
        return self.conv[-1](x)


class TFC_TDFF(nn.Module):
    def __init__(self, c, l, f, k, bn, dense=False, bias=True):

        super(TFC_TDFF, self).__init__()

        self.use_tdf = bn is not None

        self.tfc = DenseTFCC(c, l, k) if dense else TFCC(c, l, k)

        if self.use_tdf:
            if bn == 0:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    nn.GroupNorm(4, c),
                    nn.ReLU()
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    nn.GroupNorm(4, c),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    nn.GroupNorm(4, c),
                    nn.ReLU()
                )

    def forward(self, x):
        x = self.tfc(x)
        return x + self.tdf(x) if self.use_tdf else x

class TFC(nn.Module):
    def __init__(self, c, l, k):
        super(TFC, self).__init__()

        self.H = nn.ModuleList()
        for i in range(l):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=k, stride=1, padding=k // 2),
                    nn.GroupNorm(2, c),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for h in self.H:
            x = h(x)
        return x


class DenseTFC(nn.Module):
    def __init__(self, c, l, k):
        super(DenseTFC, self).__init__()

        self.conv = nn.ModuleList()
        for i in range(l):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=k, stride=1, padding=k // 2),
                    nn.GroupNorm(2, c),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for layer in self.conv[:-1]:
            x = torch.cat([layer(x), x], 1)
        return self.conv[-1](x)


class TFC_TDF(nn.Module):
    def __init__(self, c, l, f, k, bn, dense=False, bias=True):

        super(TFC_TDF, self).__init__()

        self.use_tdf = bn is not None

        self.tfc = DenseTFC(c, l, k) if dense else TFC(c, l, k)

        if self.use_tdf:
            if bn == 0:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    nn.GroupNorm(2, c),
                    nn.ReLU()
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    nn.GroupNorm(2, c),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    nn.GroupNorm(2, c),
                    nn.ReLU()
                )

    def forward(self, x):
        x = self.tfc(x)
        return x + self.tdf(x) if self.use_tdf else x

class Conv_TDF_net_trimm(nn.Module):
    def __init__(self, model_path, use_onnx, target_name, 
                 L, l, g, dim_f, dim_t, k=3, hop=1024, bn=None, bias=True, overlap=1500):
        
        super(Conv_TDF_net_trimm, self).__init__()
        
        n_fft_scale = {'vocals':3, '*':2}
        
        out_c = in_c = 4
        self.n = L//2
        self.dim_f = 3072
        self.dim_t = 256
        self.n_fft = 7680
        self.hop = hop
        self.n_bins = self.n_fft//2+1
        self.chunk_size = hop * (self.dim_t-1)
        self.target_name = target_name
        self.overlap = overlap
               
        self.stft = STFT(self.n_fft, self.hop, self.dim_f)
                   
        if not use_onnx:
            
            self.first_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=g, kernel_size=1, stride=1),
                nn.BatchNorm2d(g),
                nn.ReLU(),
            )

            f = self.dim_f
            c = g
            self.ds_dense = nn.ModuleList()
            self.ds = nn.ModuleList()
            for i in range(self.n):
                self.ds_dense.append(Conv_TDF(c, l, f, k, bn, bias=bias))

                scale = (2,2)
                self.ds.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=c, out_channels=c+g, kernel_size=scale, stride=scale),
                        nn.BatchNorm2d(c+g),
                        nn.ReLU()
                    )
                )
                f = f//2
                c += g

            self.mid_dense = Conv_TDF(c, l, f, k, bn, bias=bias)
            if bn is None and mid_tdf:
                self.mid_dense = Conv_TDF(c, l, f, k, bn=0, bias=False)

            self.us_dense = nn.ModuleList()
            self.us = nn.ModuleList()
            for i in range(self.n):
                scale = (2,2)
                self.us.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels=c, out_channels=c-g, kernel_size=scale, stride=scale),
                        nn.BatchNorm2d(c-g),
                        nn.ReLU()
                    )
                )
                f = f*2
                c -= g

                self.us_dense.append(Conv_TDF(c, l, f, k, bn, bias=bias))

            
            self.final_conv = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=out_c, kernel_size=1, stride=1),
            )

            try:
                self.load_state_dict(
                    torch.load(f'{model_path}/{target_name}.pt', map_location=device)
                )
                print(f'Loading model ({target_name})')
            except FileNotFoundError:
                print(f'Random init ({target_name})') 
        
    
    def forward(self, x):
        
        x = self.first_conv(x)
        
        x = x.transpose(-1,-2)
        
        ds_outputs = []
        for i in range(self.n):
            x = self.ds_dense[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)
        
        x = self.mid_dense(x)
        
        for i in range(self.n):
            x = self.us[i](x)
            x *= ds_outputs[-i-1]
            x = self.us_dense[i](x)
        
        x = x.transpose(-1,-2)
        
        x = self.final_conv(x)
       
        return x

class Conv_TDF_net_trim(nn.Module):
    def __init__(self, model_path, use_onnx, target_name, 
                 L, l, g, dim_f, dim_t, k=3, hop=1024, bn=None, bias=False, overlap=1754):
        
        super(Conv_TDF_net_trim, self).__init__()
        
        n_fft_scale = {'drums', 'other'}
        
        out_c = in_c = 4
        self.n = L//2
        self.dim_f = 3072
        self.dim_t = 256
        self.n_fft = 7680
        self.hop = hop
        self.n_bins = self.n_fft//2+1
        self.chunk_size = hop * (self.dim_t-1)
        self.use_onnx = use_onnx
        self.target_name = target_name
        self.overlap = overlap
        self.window = nn.Parameter(torch.hann_window(window_length=self.n_fft, periodic=True), requires_grad=False)
        self.freq_pad = nn.Parameter(torch.zeros([1, dim_c, self.n_bins - self.dim_f, self.dim_t]), requires_grad=False)
               
        self.stft = STFT(self.n_fft, self.hop, self.dim_f)
                   
        if not use_onnx:
            
            scale = (2, 2)
            self.first_conv = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=g, kernel_size=(1, 1)),
                nn.GroupNorm(2, g),
                nn.ReLU(),
            )

            f = self.dim_f
            c = g
            self.encoding_blocks = nn.ModuleList()
            self.ds = nn.ModuleList()
            for i in range(self.n):
                self.encoding_blocks.append(TFC_TDF(c, l, f, k, bn, bias=bias))
                self.ds.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                        nn.GroupNorm(2, c + g),
                        nn.ReLU()
                    )
                )
                f = f // 2
                c += g

            self.bottleneck_block = TFC_TDF(c, l, f, k, bn, bias=bias)

            self.decoding_blocks = nn.ModuleList()
            self.us = nn.ModuleList()
            for i in range(self.n):
                self.us.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale),
                        nn.GroupNorm(2, c - g),
                        nn.ReLU()
                    )
                )
                f = f * 2
                c -= g

                self.decoding_blocks.append(TFC_TDF(c, l, f, k, bn, bias=bias))

            self.final_conv = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=4, kernel_size=(1, 1)),
            )

            try:
                self.load_state_dict(
                    torch.load(f'{model_path}/{target_name}.pt', map_location=lambda storage, loc: storage)
                )
                print(f'Loading model ({target_name})')
            except FileNotFoundError:
                print(f'Random init ({target_name})') 
        
    
    def forward(self, x):

        x = self.first_conv(x)

        x = x.transpose(-1, -2)

        ds_outputs = []
        for i in range(self.n):
            x = self.encoding_blocks[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        x = self.bottleneck_block(x)

        for i in range(self.n):
            x = self.us[i](x)
            x *= ds_outputs[-i - 1]
            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        return x

class Conv_TDF_net_trimmmm(nn.Module):
    def __init__(self, model_path, use_onnx, target_name, 
                 L, l, g, dim_f, dim_t, k=3, hop=1024, bn=None, bias=False, overlap=1350):
        
        super(Conv_TDF_net_trimmmm, self).__init__()
        
        n_fft_scale = {'bass'}
        
        out_c = in_c = 4
        self.n = L//2
        self.dim_f = 2048
        self.dim_t = 256
        self.n_fft = 16384
        self.hop = hop
        self.n_bins = self.n_fft//2+1
        self.chunk_size = hop * (self.dim_t-1)
        self.use_onnx = use_onnx
        self.target_name = target_name
        self.overlap = overlap
        self.window = nn.Parameter(torch.hann_window(window_length=self.n_fft, periodic=True), requires_grad=False)
        self.freq_pad = nn.Parameter(torch.zeros([1, dim_c, self.n_bins - self.dim_f, self.dim_t]), requires_grad=False)
               
        self.stft = STFT(self.n_fft, self.hop, self.dim_f)
                   
        if not use_onnx:
            
            scale = (2, 2)
            self.first_conv = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=g, kernel_size=(1, 1)),
                nn.GroupNorm(4, g),
                nn.ReLU(),
            )

            f = self.dim_f
            c = g
            self.encoding_blocks = nn.ModuleList()
            self.ds = nn.ModuleList()
            for i in range(self.n):
                self.encoding_blocks.append(TFC_TDFF(c, l, f, k, bn, bias=bias))
                self.ds.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                        nn.GroupNorm(4, c + g),
                        nn.ReLU()
                    )
                )
                f = f // 2
                c += g

            self.bottleneck_block = TFC_TDFF(c, l, f, k, bn, bias=bias)

            self.decoding_blocks = nn.ModuleList()
            self.us = nn.ModuleList()
            for i in range(self.n):
                self.us.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale),
                        nn.GroupNorm(4, c - g),
                        nn.ReLU()
                    )
                )
                f = f * 2
                c -= g

                self.decoding_blocks.append(TFC_TDFF(c, l, f, k, bn, bias=bias))

            self.final_conv = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=4, kernel_size=(1, 1)),
            )

            try:
                self.load_state_dict(
                    torch.load(f'{model_path}/{target_name}.pt', map_location=lambda storage, loc: storage)
                )
                print(f'Loading model ({target_name})')
            except FileNotFoundError:
                print(f'Random init ({target_name})') 
        
    
    def forward(self, x):

        x = self.first_conv(x)

        x = x.transpose(-1, -2)

        ds_outputs = []
        for i in range(self.n):
            x = self.encoding_blocks[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        x = self.bottleneck_block(x)

        for i in range(self.n):
            x = self.us[i](x)
            x *= ds_outputs[-i - 1]
            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        return x
    
  
def get_models(model_path, use_onnx):
    model_name = os.path.basename(model_path)
    
    if model_name=='TFC_TDF_UNet_MDX21':   
        return [
            Conv_TDF_net_trimm(   
                model_path, use_onnx, target_name='vocals', 
                L=11, l=3, g=48, bn=8, bias=False, 
                dim_f=11, dim_t=8
            ),
            Conv_TDF_net_trim(
                model_path, use_onnx=False, target_name='drums', 
                L=11, l=3, g=48, bn=8, bias=False,
                dim_f=11, dim_t=8
            ),
            Conv_TDF_net_trim( 
                model_path, use_onnx=False, target_name='other',  
                L=11, l=3, g=48, bn=8, bias=False, 
                dim_f=11, dim_t=8
            ),
            Conv_TDF_net_trimmmm(
                model_path, use_onnx=False, target_name='bass',                 
                L=15, l=4, g=40, bn=8, bias=False,
                dim_f=11, dim_t=8
            )
        ]
    
    else:
        print('Model undefined')
        return None
