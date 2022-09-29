import numpy as np
import os
import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.io as sio
import scipy.signal as signal
import sys
from glob import glob
import matplotlib.cm as cm
import shelve
import complexityMeasures as compM
from multiprocessing import Pool
from sklearn.decomposition import PCA
import pywt
import mne.viz as mviz
import mne
import scipy.special as sp
#from analyzeEntropy import analyzeEntropy

mntNIRS = np.load("NIRSmontage.npy")#[:-2,:]
def waveletThresh(data, wavelet='db5', thresh=5, mode=False):
    coeff = pywt.wavedec(data, wavelet)
    lengths = [len(coeff[k+1]) for k in range(len(coeff[1:]))]
    temp = np.hstack(coeff[1:])
    phi = 1/2*(1+sp.erf(abs(temp)/(np.sqrt(2)*np.std(abs(temp)))))
    p = 2*(1-phi)
    temp[p < .1] = 0
    for k in range(len(lengths)):
        if k == 0:
            coeff[k+1] = temp[0:lengths[k]]
        else:
            coeff[k+1] = temp[sum(lengths[:k]):sum(lengths[:k])+lengths[k]]

    return pywt.waverec(coeff, wavelet)[:len(data)]

def detrend(data, order=1):
    Y = data - data.mean()
    X = np.array([[i**x for i in range(len(data))] for x in range(order+1)], dtype=float)
    W = np.linalg.pinv((X).dot(X.T)).dot(X).dot(Y.T)
    return Y - W.dot(X)# + data.mean()

    
def doMBL(c, eps=[[.974, .693],[.35,2.1]], DPF=[5.98, 7.15], sd=2.5, filt=False,
        ica=False, embed=None, tau=None, mrk=None, oxy="",subj="",trial=""):
    if not mrk is None:
        # Get in timeunits of NIRS
        mrk.time = mrk.time//100
    if not os.path.isfile("beerlamb_{0}_{1}.npy".format(subj,trial)):
        nirs = c.x
        event = c.title
        timestamps = []
        event = []
        if not mrk is None:
            # Get in timeunits of NIRS
            mrk.time = mrk.time//100
        eps = np.array(eps) # extinction coefficient per chromophore per wavelength
        DPF = np.array(DPF) # Differential pathlength factor per wavelength
        sd = np.array(sd) # source detector separation
        # Separate low and high wavelengths
        lowW = nirs[:,:nirs.shape[1]//2]
        highW = nirs[:,nirs.shape[1]//2:]
        freqs = np.fft.fftfreq(len(highW), d=1/c.fs)
        beerlamb = []
        if filt:
            nyq = c.fs/2
            b,a = signal.butter(6,.6/nyq, "low")
            b2,a2 = signal.butter(3,[.8/nyq, 2/nyq], "band")
            kk = []
            ignArr = np.mean(lowW,0) > .03
            for k in range(len(lowW[0])):
                if ignArr[k]:
                    logLow = -np.log10(np.mean(lowW[:c.fs*60,k])/lowW[c.fs*60:,k])/(DPF[1] * sd)
                    logHigh = -np.log10(np.mean(highW[:c.fs*60,k])/highW[c.fs*60:,k])/(DPF[0] * sd)
                    A = np.vstack((logHigh, logLow))
                    beerlamb.append(np.linalg.solve(eps,A))
                    beerlamb[-1][0] = signal.filtfilt(b2,a2,beerlamb[-1][0]) + signal.filtfilt(b,a, beerlamb[-1][0])
                    beerlamb[-1][1] = signal.filtfilt(b2,a2,beerlamb[-1][1]) + signal.filtfilt(b,a, beerlamb[-1][1])
                else:
                    beerlamb.append(np.zeros((2, len(lowW)-c.fs*60)))
            beerlamb = np.array(beerlamb)
    
        tempret = []
        output = []
        outsEn = []
        kk = np.array(np.argpartition(kk,2)[:2])
        np.save("NIRSignore.npy", kk)
        temp = []
        for k in range(lowW.shape[1]):
            if k in kk:
                temp.append(True)
            else:
                temp.append(False)
        kk = temp
        mrk.time = mrk.time - 60*c.fs
        for ppp, (Z) in enumerate(beerlamb[ignArr]):
            tempd = waveletThresh(Z[0,:], thresh=3.3)
            tempo = waveletThresh(Z[1,:], thresh=3.3)
            beerlamb[ignArr][ppp,0,:] = tempd
            beerlamb[ignArr][ppp,1,:] = tempo
            beerlamb[ignArr][ppp,0,:] = detrend(beerlamb[ignArr][ppp,0,:],7)
            beerlamb[ignArr][ppp,1,:] = detrend(beerlamb[ignArr][ppp,1,:],7)
    
    
        # Interpolating bad Chans
        chansBad = np.where(np.mean(lowW,0) < .03)[0] 
        beerlamb = interpolateBadChans(beerlamb, chansBad, ignArr, mntNIRS)
    
        # Zero mean each channel
        beerlamb[:,0,:] = (beerlamb[:,0,:].T - np.mean(beerlamb[:,0,:],1)).T
        beerlamb[:,1,:] = (beerlamb[:,1,:].T - np.mean(beerlamb[:,1,:],1)).T
    else:
        beerlamb = np.load("beerlamb_{0}_{1}.npy".format(subj,trial))
        nyq = c.fs/2
        b,a = signal.butter(6,.6/nyq, "low")
        b2,a2 = signal.butter(3,[.8/nyq,3/nyq], "band")
#        beerlamb = signal.filtfilt(b2,a2,beerlamb)#+np.random.randn(*beerlamb.shape)*np.std(beerlamb)
        kk = np.load("NIRSignore.npy")
        temp = []
        tempret = []
        output = []
        outsEn = []
        for k in range(beerlamb.shape[0]):
            if k in kk:
                temp.append(True)
            else:
                temp.append(False)
        kk = temp

    for j, t in enumerate(mrk.time):
        print(mrk.className[mrk.event.desc[j]-1])
        ret = []
        tauarr = []
        # Go through each channel. Can be improved by tensor operation
        ret = beerlamb[:,:,t-10*c.fs:t+40*c.fs]
        ret[:,0,:] = (ret[:,0,:].T - beerlamb[:,0,t-2*c.fs:t].mean(1)).T
        ret[:,1,:] = (ret[:,1,:].T - beerlamb[:,1,t-2*c.fs:t].mean(1)).T
        for k in range(beerlamb.shape[0]):
#            tau0 = compM.findTau(ret[k,0,:])
#            tau1 = compM.findTau(ret[k,1,:])
            tauarr.append(2)#int(np.mean([tau0,tau1])))
        embed = []
#        plt.plot(ret[:,0,:].T)
#        plt.figure()
#        plt.plot(ret[:,1,:].T)
#        plt.show()
        if oxy == "Oxy":
            ret = ret[:,0,:][:,np.newaxis,:]
        elif oxy == "Deoxy":
            ret = ret[:,1,:][:,np.newaxis,:]
        elif oxy == "Total":
            ret = (ret[:,1,:] + ret[:,0,:])[:,np.newaxis,:]
        else:
            pass
        for k in range(len(ret)):
            if not kk[k]:
#                resfnn = compM.fnn(ret[k], .2, tauarr[k])
                embed.append(10)#len(resfnn))
            else:
                embed.append(1)
#        print("Embedded")
        m = np.mean(embed)
        m = int(m)
        sEn = []
        for k in range(len(ret)):
            if not kk[k]:
                sampEn = compM.SampEn(ret[k], embed[k], tauarr[k], 1)# * np.std([ret[k,0,:ret.shape[2]//2]]))
                sEn.append(sampEn)
            else: 
                sEn.append(0)
        #print(sEn)
        output.append(ret)
        outsEn.append(sEn)
        print(j)
    # Pre Activity
    ret = beerlamb[:,:,100:300]
    for k in range(beerlamb.shape[0]):
        # Wavelet thresholding
#        tau0 = compM.findTau(ret[k,0,:])
#        tau1 = compM.findTau(ret[k,1,:])
        tauarr.append(2)#int(np.mean([tau0,tau1])))
    embed = []
    ret = np.array(ret)
    if oxy == "Oxy":
        ret = ret[:,0,:][:,np.newaxis,:]
    elif oxy == "Deoxy":
        ret = ret[:,1,:][:,np.newaxis,:]
    elif oxy == "Total":
        ret = (ret[:,0,:] + ret[:,1,:])[:,np.newaxis,:]
    else:
        pass

    for k in range(len(ret)):
        if not kk[k]:
#            resfnn = compM.fnn(ret[k], .2, tauarr[k])
            embed.append(10)#len(resfnn))
        else:
            embed.append(1)
    m = np.mean(embed)
    m = int(m)
    sEn = []
    for k in range(len(ret)):
        if not kk[k]:
            sampEn = compM.SampEn(ret[k], embed[k], tauarr[k], .2)# * np.std([ret[k,0,:ret.shape[2]//2]]))
            sEn.append(sampEn)
        else:
            sEn.append(0)
    #print(sEn)
    output.append(ret)
    outsEn.append(sEn)
    print("Pre")

    # Post Activity
    tauarr = []
    ret = beerlamb[:,:,-400:-200]
    for k in range(beerlamb.shape[0]):
#        tau0 = compM.findTau(ret[k,0,:])
#        tau1 = compM.findTau(ret[k,1,:])
        tauarr.append(2)#int(np.mean([tau0,tau1])))
    embed = []
    ret = np.array(ret)
    if oxy == "Oxy":
        ret = ret[:,0,:][:,np.newaxis,:]
    elif oxy == "Deoxy":
        ret = ret[:,1,:][:,np.newaxis,:]
    elif oxy == "Total":
        ret = (ret[:,0,:] + ret[:,1,:])[:,np.newaxis,:]
    else:
        pass
    for k in range(len(ret)):
        if not kk[k]:
#            resfnn = compM.fnn(ret[k], .2, tauarr[k])
            embed.append(10)#len(resfnn))
        else:
            embed.append(1)
    m = np.mean(embed)
    m = int(m)
    sEn = []
    for k in range(len(ret)):
        if not kk[k]:
            sampEn = compM.SampEn(ret[k], embed[k], tauarr[k], .2)# * np.std([ret[k,0,:ret.shape[2]//2]]))
            sEn.append(sampEn)
        else:
            sEn.append(0)
    #print(sEn)
    output.append(ret)
    outsEn.append(sEn)
    print("Post")
       
    return output, np.array(outsEn)

def poolFunc(inp):
    r, sen = doMBL(inp[0], filt=True,ica=False, mrk=inp[-1], oxy=inp[-2],
            subj=str(inp[1]),trial=str(inp[2]))
    print("Subject " + str(inp[1]) + "_" + str(inp[2]) + " Done!")
    if not os.path.isdir("EMBCresultsNIRSSampEn"+str(inp[-2])+"/"):
        os.mkdir("EMBCresultsNIRSSampEn"+str(inp[-2])+"/")
    np.savez("EMBCresultsNIRSSampEn"+str(inp[-2])+"/results" + str(inp[1]) + "_" + str(inp[2]) + ".npy", *r)
    np.save("EMBCresultsNIRSSampEn"+str(inp[-2])+"/sen" + str(inp[1]) + "_" + str(inp[2]) + ".npy", sen)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        oxy = ""#sys.argv[1]
    else:
        oxy = ""
    dats = []
    marks = []
    args = []
    tasks = []
    for s, path in enumerate(sorted(glob("./NIRS*/subj*/"))):
        cnt = sio.loadmat(path + "cnt.mat", struct_as_record=False, squeeze_me=True)["cnt"]
        mrk =sio.loadmat(path + "mrk.mat", struct_as_record=False, squeeze_me=True)["mrk"]
#        if s > 10:
#            break
        for n, (c,mk) in enumerate(zip(cnt,mrk)):
            args.append([c, s,n, "Oxy", mk])
        for n, (c,mk) in enumerate(zip(cnt,mrk)):
            args.append([c, s,n, "Deoxy", mk])
        for n, (c,mk) in enumerate(zip(cnt,mrk)):
            args.append([c, s,n, "Total", mk])
        for n, (c,mk) in enumerate(zip(cnt,mrk)):
            args.append([c, s,n, "", mk])

    print(len(args))
    from joblib import Parallel, delayed
    Parallel(n_jobs=250)(delayed(poolFunc)(a) for a in args)
#    analyzeEntropy()
#    for a in args:
#        poolFunc(a)
