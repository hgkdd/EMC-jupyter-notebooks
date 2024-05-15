import scipy as sp
import numpy as np

def get_freqs(Nt, timestep):
    # setup freq array
    freqs = sp.fft.fftfreq(Nt, d=timestep)
    # rearrange freq array
    freqs = sp.fft.fftshift(freqs)
    return freqs

def fourier(signal, t_min, t_max, Nt):
    # we assume times are generateted with np.linsoace(t_min, t_max, Nt)
    timestep = (t_max - t_min) / (Nt - 1)
    # setup freq array
    freqs = get_freqs(Nt, timestep)
    # Perform numerical Fourier Transform
    FT = sp.fft.fft(signal)
    # Rearrange array
    FT = sp.fft.fftshift(FT)
    # Normalize FFT
    FT *= (t_max - t_min) / Nt
    # shift phase
    FT = FT * np.exp(-1j*2*np.pi*freqs*t_min)
    return freqs, FT

def fourier_coefficients(signal, t_min, t_max, Nt):
    period = t_max - t_min
    omega0 = 2*np.pi / period
    freqs, FT = fourier(signal, t_min, t_max, Nt)
    cn = FT / Period
    omegas = 2*np.pi*freqs
    return omega0, period, omegas, cn

def rectangle(t, amplitude, width, middle):
    pls = lambda t: amplitude * ( np.heaviside(t-(middle-width/2), 0) - np.heaviside(t-(middle+width/2), 0))
    return pls(t)

def FT_rectangle(freqs, amplitude, width, t0=0):
    # calc omega from freq
    omega = 2*np.pi*freqs
    # analytical Fourier Transform
    FT = amplitude * width * np.sinc(omega*width/2 / np.pi) # numpy uses the normalized sinc(x) = sin(pi*x)/(pi*x) -> divide by np.pi 
    # phase shift due to time shift
    FT = FT * np.exp(-1j*omega*t0)
    return FT

def trapzoid(t, amplitude, width, risetime, t0=0):
    slope = amplitude/risetime
    pls = slope*width*sp.signal.sawtooth(2*np.pi*(t-t0-width/2)/width, width=0.5)/4.
    pls += slope*width/4.
    pls[pls>amplitude] = amplitude
    pls[np.where(t>t0+width/2)] = 0
    pls[np.where(t<t0-width/2)] = 0
    return pls

def FT_trapzoid(freqs, amplitude, width, risetime, t0=0):
    FT = amplitude * FT_rectangle(freqs, 1, width-2*risetime) * FT_rectangle(freqs, 1, risetime)
    # phase shift due to time shift
    FT = FT * np.exp(-1j*2*np.pi*freqs*t0)
    return FT
