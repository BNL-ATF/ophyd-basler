import numpy as np

def gaussian(x, y, a, cx, cy, sx, sy): 
    return a * np.exp(-0.5*(np.square((x-cx)/sx) + np.square((y-cy)/sy)))

def get_wandering_gaussian_beam(nf, nx, ny):

    # hard-coded for now

    length_scale = 64
    bounds = np.array([[0,256],[0,nx],[0,ny],[nx/64, nx/8],[ny/64, ny/8]])

    ps    = np.exp(-np.square(np.fft.fftfreq(nf)*length_scale))
    data  = np.real(np.fft.ifft(ps * np.fft.fft(np.random.standard_normal(size=(5, nf)))))
    data -= data.min(axis=1)[:,None]
    data *= (bounds.ptp(axis=1) / data.ptp(axis=1))[:,None]
    data += bounds.min(axis=1)[:,None]
    
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    
    return gaussian(X[None,:,:], Y[None,:,:], *data[:,:,None,None])

