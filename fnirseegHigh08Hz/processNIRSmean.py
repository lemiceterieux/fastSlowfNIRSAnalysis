import numpy as np
import os
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
from sklearn.decomposition import FastICA
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

def ICAjointProbCompRej(data, ignArr, plot=False):
    inputICAmatrix = np.vstack(data[ignArr])
    # Get amount of components correspond to 99.99% of variance with PCA
    clfP = PCA(n_components=.9999)
    pcares = clfP.fit_transform(inputICAmatrix.T)
    
    # Use that for prior amount of components to extract from ICA
    clfI = FastICA(n_components=pcares.shape[1], tol=0.01)
    icares = clfI.fit_transform(inputICAmatrix.T)

    if plot:
     # Plotting before
        info = mne.create_info(pcares.shape[1], c.fs)
        raw = mne.io.RawArray(icares.T, info)
       
        raw.plot(n_channels=icares.shape[1], scalings='auto', show=False, block=False)
       
        info2 = mne.create_info(len(np.vstack(data[ignArr])), c.fs)
        raw2 = mne.io.RawArray(np.vstack(data[ignArr]), info2)
       
        raw2.plot(n_channels=len(np.vstack(data[ignArr])), scalings='auto', show=False, block=False)
    pft = np.apply_along_axis(lambda a: abs(np.fft.fft(a)**2), 0, icares) 
    pft = pft/np.sum(pft,0)
    pent = -np.apply_along_axis(lambda p: np.sum(p[p>0]*np.log2(p[p>0])),0,pft)/np.log2(len(pft))
    print(pent)
    kurt = stats.kurtosis(icares)
    ps, edges = np.histogramdd((pent,stats.kurtosis(icares)), bins=int(len(pent)**(1/3)))
    print(np.max(ps))
    ps = ps/np.max(ps)

    # Joint Prob detection
    for  k in range(len(kurt)):
        indexi = np.where(edges[0] <= pent[k])[0][-1]
        indexj = np.where(edges[1] <= kurt[k])[0][-1]
        if indexi == ps.shape[0]:
            indexi -= 1
        if indexj == ps.shape[1]:
            indexj -= 1
        if ps[indexi,indexj] > .8:
#            plt.plot(icares[:,k])
#            plt.show()
            icares[:,k] = 0
    data[ignArr] = clfI.inverse_transform(icares).T.reshape(sum(ignArr), 2, len(icares))
    if plot:
       info3 = mne.create_info(ret[ignArr].shape[1], c.fs)
       raw3 = mne.io.RawArray(ret[ignArr][0], info3)
       
       raw3.plot(n_channels=len(temp), scalings='auto', show=False, block=False)
       plt.show()

    return data

def detrend(data, order=1):
    Y = data - data.mean()
    X = np.array([[i**x for i in range(len(data))] for x in range(order+1)], dtype=float)
    W = np.linalg.pinv((X).dot(X.T)).dot(X).dot(Y.T)
    return Y - W.dot(X)# + data.mean()

def ICAkurtElim(data, plot=False, ignoreArr=None):
        clf = FastICA(n_components=2)
        Z = clf.fit_transform(data.T)
        Kurt = np.mean(((Z - np.mean(Z,0))/np.std(Z,0))**4,0)
        if plot:
            fig, ax = plt.subplots(2,1)
            ax = ax.ravel()
            for zz in range(Z.shape[1]):
                if ignoreArr is None:
                    acc = np.array([True for k in range(len(mntNIRS))])
                else:
                    acc = ignoreArr
                mviz.plot_topomap(clf.components_[zz,:], mntNIRS[acc, :], show=False, axes=ax[zz])
                ax[zz].set_title("IC " + str(zz) + " {0:.2f}".format(Kurt[zz]))
            fig2, ax2 = plt.subplots(2,1)
            ax2 = ax2.ravel()
            for zz in range(Z.shape[1]):
                ax2[zz].plot(Z[:,zz])
                ax2[zz].set_title("IC " + str(zz) + " {0:.2f}".format(Kurt[zz]))
            plt.show()
        Z[:,Kurt > 4] = 0
        return clf.inverse_transform(Z).T

def interpolateBadChans(data, chansBad, ignArr, mntArr):
    for chans in chansBad:
        distanceVec = 1/np.sqrt(np.sum((mntArr[chans] - mntArr[ignArr])**2,1))
        distanceVec = distanceVec/np.sum(distanceVec)
        data[chans,0,:] = np.sum(np.array([distanceVec[j]*data[j,0,:] for j, r in enumerate(data[ignArr])]),0)
        data[chans,1,:] = np.sum(np.array([distanceVec[j]*data[j,1,:] for j, r in enumerate(data[ignArr])]),0)
    return data

    
def doMBL(c, eps=[[.974, .693],[.35,2.1]], DPF=[5.98, 7.15], sd=2.5, filt=False, ica=False, embed=None, tau=None, mrk=None, oxy="", subj="", trial=""):
    if not mrk is None:
        # Get in timeunits of NIRS
        mrk.time = mrk.time//100

    if True:#not os.path.isfile("beerlamb_{0}_{1}.npy".format(subj,trial)):
        nirs = c.x
        event = c.title
        timestamps = []
        event = []
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
            b2,a2 = signal.butter(3,[.8/nyq, 3/nyq], "band")
            b3,a3 = signal.butter(3,.8/nyq, "high")
            kk = []
            ignArr = np.mean(lowW,0) > .03
            for k in range(len(lowW[0])):
                #templ = signal.filtfilt(b2,a2,lowW[:,k])
                #temph = signal.filtfilt(b2,a2,highW[:,k])
                #lowW[:,k] = signal.filtfilt(b,a,lowW[:,k]) + templ
                #highW[:,k] = signal.filtfilt(b,a,highW[:,k]) + temph
                if ignArr[k]:
                    logLow = -np.log10(np.mean(lowW[:c.fs*60,k])/lowW[c.fs*60:,k])/(DPF[1] * sd)
                    logHigh = -np.log10(np.mean(highW[:c.fs*60,k])/highW[c.fs*60:,k])/(DPF[0] * sd)
                    A = np.vstack((logHigh, logLow))
                    beerlamb.append(np.linalg.solve(eps,A))
                    beerlamb[-1][0] = signal.filtfilt(b2,a2,beerlamb[-1][0])# + signal.filtfilt(b,a, beerlamb[-1][0])
                    beerlamb[-1][1] = signal.filtfilt(b2,a2,beerlamb[-1][1])# + signal.filtfilt(b,a, beerlamb[-1][1])
                else:
                    beerlamb.append(np.zeros((2, len(lowW)-c.fs*60)))
            beerlamb = np.array(beerlamb)
            if ica:
                ICAjointProbCompRej(beerlamb,ignArr)
    
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
            beerlamb[ignArr][ppp,0,:] = detrend(beerlamb[ignArr][ppp,0,:],2)
            beerlamb[ignArr][ppp,1,:] = detrend(beerlamb[ignArr][ppp,1,:],2)
    
    
        # Interpolating bad Chans
        chansBad = np.where(np.mean(lowW,0) < .03)[0] 
        beerlamb = interpolateBadChans(beerlamb, chansBad, ignArr, mntNIRS)
    
        # Zero mean each channel
        beerlamb[:,0,:] = (beerlamb[:,0,:].T - np.mean(beerlamb[:,0,:],1)).T
        beerlamb[:,1,:] = (beerlamb[:,1,:].T - np.mean(beerlamb[:,1,:],1)).T
    
        temp = beerlamb.reshape(-1,beerlamb.shape[-1]).T
        import torch
        W = torch.rand(72,72).float().cpu().requires_grad_()
        temp = torch.from_numpy(temp).cpu().float()
        for i in range(2000):
            L = temp.matmul(W)
            E = 1e-3*((temp[100:-101] - L[101:-100])**2).mean()
            Ew = 1*abs(W).sum()
            (E + Ew).backward()
            with torch.no_grad():
                W -= 1e-2*W.grad
            _ = W.grad.zero_()
        beerlamb = L.T.reshape(36,2,L.shape[0]).detach().numpy()
        np.save("beerlamb_{0}_{1}.npy".format(subj,trial), beerlamb)
    else:
        beerlamb = np.load("beerlamb_{0}_{1}.npy".format(subj,trial))
        nyq = c.fs/2
        #b2,a2 = signal.butter(3,.2/nyq "high")
        #beerlamb = signal.filtfilt(b2,a2,beerlamb)
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
        #print(mrk.className[mrk.event.desc[j]-1])
        ret = []
        tauarr = []
        # Go through each channel. Can be improved by tensor operation
        ret = beerlamb[:,:,t:t+40*c.fs]
        ret[:,0,:] = (ret[:,0,:].T - beerlamb[:,0,t-5*c.fs:t].mean(1)).T
        ret[:,1,:] = (ret[:,1,:].T - beerlamb[:,1,t-5*c.fs:t].mean(1)).T
        if oxy == "Oxy":
            ret = ret[:,0,:][:,np.newaxis,:]
        elif oxy == "Deoxy":
            ret = ret[:,1,:][:,np.newaxis,:]
        elif oxy == "Total":
            ret = (ret[:,1,:] + ret[:,0,:])[:,np.newaxis,:]
        else:
            pass
        sEn = []
        for k in range(len(ret)):
            if not kk[k]:
                sampEn = ret[k].mean()
                sEn.append(sampEn)
            else: 
                sEn.append(0)
        #print(sEn)
        output.append(ret)
        outsEn.append(sEn)
        #print(j)
    # Pre Activity
    ret = beerlamb[:,:,100:300]

    ret = np.array(ret)
    if oxy == "Oxy":
        ret = ret[:,0,:][:,np.newaxis,:]
    elif oxy == "Deoxy":
        ret = ret[:,1,:][:,np.newaxis,:]
    elif oxy == "Total":
        ret = (ret[:,0,:] + ret[:,1,:])[:,np.newaxis,:]
    else:
        pass

    sEn = []
    for k in range(len(ret)):
        if not kk[k]:
            sampEn = ret[k].mean()
            sEn.append(sampEn)
        else:
            sEn.append(0)
    #print(sEn)
    output.append(ret)
    outsEn.append(sEn)
    #print("Pre")

    # Post Activity
    tauarr = []
    ret = beerlamb[:,:,-400:-200]
    ret = np.array(ret)
    if oxy == "Oxy":
        ret = ret[:,0,:][:,np.newaxis,:]
    elif oxy == "Deoxy":
        ret = ret[:,1,:][:,np.newaxis,:]
    elif oxy == "Total":
        ret = (ret[:,0,:] + ret[:,1,:])[:,np.newaxis,:]
    else:
        pass
    sEn = []
    for k in range(len(ret)):
        if not kk[k]:
            sampEn = ret[k].mean()
            sEn.append(sampEn)
        else:
            sEn.append(0)
    #print(sEn)
    output.append(ret)
    outsEn.append(sEn)
    #print("Post")
       
    return output, np.array(outsEn)

def poolFunc(inp):
    r, sen = doMBL(inp[0], filt=True,ica=False, mrk=inp[-1], oxy=inp[-2], subj=str(inp[1]), trial=str(inp[2]))
    print("Subject " + str(inp[1]) + "_" + str(inp[2]) + " Done!")
    if not os.path.isdir("EMBCresultsNIRSMean"+str(inp[-2])+"/"):
        os.mkdir("EMBCresultsNIRSMean"+str(inp[-2])+"/")
    np.savez("EMBCresultsNIRSMean"+str(inp[-2])+"/results" + str(inp[1]) + "_" + str(inp[2]) + ".npy", *r)
    np.save("EMBCresultsNIRSMean"+str(inp[-2])+"/sen" + str(inp[1]) + "_" + str(inp[2]) + ".npy", sen)


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


    from joblib import Parallel, delayed
    Parallel(n_jobs=200)(delayed(poolFunc)(a) for a in args)
#    p = Pool(200)
#    p.map(poolFunc, args)
#    analyzeEntropy()
#    for a in args:
#        poolFunc(a)
