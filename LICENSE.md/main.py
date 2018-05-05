import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.signal import argrelextrema

###############################################################################
""" 1) READING THE DATA IN """

np.set_printoptions(threshold=50)


def read(f):
    #Labview Motor VI data file is binary bigendian
    dat = np.array(np.fromfile(f, dtype='>d'), dtype=float)
    #removes junk data + headers. Range can probably be precisely determined
    dat = dat[95:len(dat) - 4]
    return dat
 

step_length  = 2.198 * 10**-9 # +/- 0.001 nm



MYF = read('C:/Users/18072/OneDrive/Documents/Academic/University/Physics/2nd_Year/Laboratory/Interferometry/Mercury/Yellow_filter/(MYF)N17000_S802_E988')
MGF_v1 = read('C:/Users/18072/OneDrive/Documents/Academic/University/Physics/2nd_Year/Laboratory/Interferometry/Mercury/Green_filter/(MGF)N17000_S802_E988')
MGF_v2 = read('C:/Users/18072/OneDrive/Documents/Academic/University/Physics/2nd_Year/Laboratory/Interferometry/Mercury/Green_filter/(MGF)N360000_S670_E1120')

WGF_v1 = read('C:/Users/18072/OneDrive/Documents/Academic/University/Physics/2nd_Year/Laboratory/Interferometry/Tungsten/Green_filter/(WGF)N25500_clock_VFINAL')
WGF_v2 = read('C:/Users/18072/OneDrive/Documents/Academic/University/Physics/2nd_Year/Laboratory/Interferometry/Tungsten/Green_filter/(WGF)N8000_V2')
WGF_v3 = read('C:/Users/18072/OneDrive/Documents/Academic/University/Physics/2nd_Year/Laboratory/Interferometry/Tungsten/Green_filter/(WGF)N8000_V1')
WNF_v1 = read('C:/Users/18072/OneDrive/Documents/Academic/University/Physics/2nd_Year/Laboratory/Interferometry/Tungsten/No_filter/(WNF)N8000_middle')
WNF_v2 = read('C:/Users/18072/OneDrive/Documents/Academic/University/Physics/2nd_Year/Laboratory/Interferometry/Tungsten/No_filter/(WNF)N8000_negative')
WNF_v3 = read('C:/Users/18072/OneDrive/Documents/Academic/University/Physics/2nd_Year/Laboratory/Interferometry/Tungsten/No_filter/(WNF)N24750_S880_E910')
WYF_v1 = read('C:/Users/18072/OneDrive/Documents/Academic/University/Physics/2nd_Year/Laboratory/Interferometry/Tungsten/Yellow_filter/(WYF)N24750')
WYF_v2 = read('C:/Users/18072/OneDrive/Documents/Academic/University/Physics/2nd_Year/Laboratory/Interferometry/Tungsten/Yellow_filter/(WYF)N24750_S880_E910_V2')


###############################################################################
""" 2) FOURIER TRANSFORMS"""

def fourier(func, delta):
    """Does FFT transform of f with proper scaling. x and k are Fourier pair.
    delta is necessary to know how distant are the samples apart.
    Returns tuple (freq,Fourier),
        - freq is wavenumber/angular frequency
        - Fourier = F(freq) is Fourier transform of f(t) = func"""
    N_samples = len(func)
    # we need Nyquist wavenumbers on either side
    freq = np.linspace(-1.0/(2.0*delta), 1.0/(2.0*delta), N_samples)
    Fourier = np.fft.fft(func)
    # flip around ans scale vertically
    Fourier = 2.0/N_samples * np.concatenate((Fourier[(N_samples)//2:],
                                                    Fourier[:(N_samples)//2]))
    return (freq, Fourier)
    
    

def inverse_fourier(func, delta):
    """Does inverse FFT, the parameters are the same as above"""
    N_samples = len(func)
    # Converting to same Nyquist wavenumbers on either side
    time = np.linspace(-1.0/(2.0*delta), 1.0/(2.0*delta), N_samples)
    func = N_samples/2.0 * np.concatenate((func[(N_samples)//2:],
                                                    func[:(N_samples)//2]))
    iFourier = np.fft.ifft(func)
    return (time, iFourier)

    
###############################################################################
""" 3) FILTERING OUT UNNECESSARY DATA"""
    
def sq_filter(x, a, b):
    return square_filter(x, (a+b)/2., np.abs(b-a))
    

def square_filter(x, mean, bandwidth):
    func = np.zeros(len(x))
    for idx in range(len(x)):
        # if the values of the function like within one sd of the mean, include them in the output
        # otherwise set values to zero
        # VERY important that only bandwidth is divided by two otherwise the two fourier limits will not be upper and lower
        if x[idx] > mean-(bandwidth/2) and x[idx] < mean+(bandwidth/2):
            func[idx] = 1
    return func 


def cut(data, start, end):
    newlist = data[start:end]
    return newlist
    
    
###############################################################################
""" 4) PLOTTING THE DATA AND FOURIER TRANSFORMS """    

def centre(data):

    old_time =[]
    for i in range(0,len(data)):
        old_time.append(i)
    
    m = max(data)
 
    # a good alternative to using the argmax function
    middle = [i for i, j in enumerate(data) if j == m]
    
    # factor of 5 is because micrometer moves 5 times further than mirror
    # factor of 20 is because the sampling rate is 20 which means mirror takes 20 steps between each sample
    # factor of 10**3 to convert to mm from m
    step_to_mm = 5. * step_length * 30 * 10**3
    
    new_time = []
    for i in old_time:
        new_time.append((i - middle[0])*step_to_mm)
    return new_time

    
def fouriers(data, a, b, c, d):
    
    # c and d are the start and end of the data respectively
    # this is useful for getting rid of noise at the beginning of the data
    data = cut(data, c, d)
    
    # x and k are fourier conjugate pairs
    # k is the frequency, spectrum is the fourier transform
    # 1 is the delta function input
    k, spectrum = fourier(data, 1)
    
    #Step length in fourier domain
    dk = k[1]-k[0]

    #square filter = 1 between a and b, 0 otherwise
    # this is to remove the low frequency waves which otherwise make up the entire fourier domain signal
    filtered_spectrum = spectrum * sq_filter(k, a, b)
    
    #Perfroms inverse to get original data but with wavelengths filtered
    # this removes the offset in the amplitude and smooths out the data
    x, filtered_data = inverse_fourier(filtered_spectrum, dk)
    

    # factor of 20 is because the sampling rate is 20 which means mirror takes 20 steps between each sample
    step_to_wavenumber = 1 / ((step_length*30))


    x = centre(data)
    #print (filtered_spectrum[1000:2000])
    #print (k[k[a]:k[b]])
    
    plt.figure(1)
    plt.plot(x, data)
    plt.title("Space space data, no pass band")
    plt.xlabel("Distance from null point / mm")
    plt.ylabel("Intensity")
    
    plt.figure(2)
    plt.plot(x, filtered_data)
    plt.title("Space space data, pass band")
    plt.xlabel("Distance from null point / mm")
    plt.ylabel("Intensity")

    plt.figure(3)
    # taking the absolute value is important as it ensures the negative values are ignored
    plt.plot(k*step_to_wavenumber, np.abs(spectrum)*10**10)
    plt.title("Fourier space, no pass band")
    plt.xlabel("Wavenumber / (1/m)")
    plt.ylabel("Intensity")
    
    plt.figure(4)
    plt.plot(k*step_to_wavenumber, np.abs(filtered_spectrum)*10**10)
    plt.title("", fontsize = fs)
    plt.xlabel("Wavenumber / (1/m)", fontsize = fs)    #Uncertainty in step length was 0.001 nm
    plt.ylabel("Intensity", fontsize = fs)
    plt.xlim(a*step_to_wavenumber, b*step_to_wavenumber)
    

    ###############################################################################
    """ 5) FITTING """

    """
    Ind_tuple = argrelextrema(filtered_spectrum, np.greater)
    Ind_list = []
    for i in Ind_tuple:
        Ind_list.append(i)
    #maxInd = np.asarray(Ind_list)
    #maxInd = maxInd.reshape(np.ma.size(maxInd))
    maxInd = Ind_list
    #for i in ydata:
        
    #c = (diff(sign(diff(data))) < 0).nonzero()[0] + 1 # local max
    profile = filtered_spectrum[maxInd]
    print (len(profile))
    nstep = []
    for i in range(0,len(profile)):
        nstep.append(i)
    print (len(nstep))
    plt.figure(5)
    plt.plot(nstep, profile, 'b')
    y = profile
    
    nstep = []
    for i in range(0,len(profile)):
        i = float(i)
        nstep.append(i)
        
    j = nstep
    """
    j = k*step_to_wavenumber
    def gaussian1(j, amp1, wid1):
            "1-d gaussian: gaussian(x, amp, cen, wid)"
            return (amp1/(np.sqrt(2*np.pi)*wid1)) * np.exp(-(j-1733225.411)**2 /(2*wid1**2))
            
    def gaussian2(j, amp2, wid2):
            "1-d gaussian: gaussian(x, amp, cen, wid)"
            return (amp2/(np.sqrt(2*np.pi)*wid2)) * np.exp(-(j-1726921.848)**2 /(2*wid2**2))
            
    def gaussian(j, amp, wid, cen):
            "1-d gaussian: gaussian(x, amp, cen, wid)"
            return (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(j-cen)**2 /(2*wid**2))

    #mod = Model(gaussian1) + Model(gaussian2)
    mod = Model(gaussian)
    
    #pars = mod.make_params(amp1=4000., amp2=20000., wid1=10000., wid2=15000.)
    pars = mod.make_params(amp=4000, wid=10000, cen=1700000)
    
    # 'mid' and 'center' should be completely correlated, and 'mid' is
    # used as an integer index, so a very poor fit variable:
    #pars['cen1'].vary = False
    
    # fit this model to data array y
    result =  mod.fit(np.abs(filtered_spectrum), params=pars, j=j)
    
    print(result.fit_report())
    
    
    plt.figure(5)
    plt.plot(j, result.best_fit, 'rs')
    plt.plot(j, np.abs(filtered_spectrum), 'b')
    #plt.title("Fourier space, pass band")
    plt.xlabel("Wavenumber / (1/m)", fontsize = fs)    #Uncertainty in step length was 0.001 nm
    plt.ylabel("Intensity", fontsize = fs)
    plt.xlim(a*step_to_wavenumber, b*step_to_wavenumber)

###############################################################################
""" 5) INPUTTING PARAMETERS TO RUN """

# The data contains the amplitudes at certain step values, so there is no SI conversion needed.
data = MYF

fs = 20

#change this to 500, 9000 for Mercury yellow
Space_space_start, Space_space_end = 500, 8400
#Lower and upper allowed frequencies in fourier space , for mercury, yellow, 0.1 , 10 is good
Fourier_space_start , Fourier_space_end = 0.10, 0.15

#Data to be used, lower frequency bound for filter, upper frequency bound for filter
fouriers(data, Fourier_space_start, Fourier_space_end, Space_space_start, Space_space_end)
