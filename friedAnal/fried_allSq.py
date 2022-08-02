import numpy as np
import matplotlib
import pandas as pd
import dcor
from matplotlib import ticker
from joblib import Parallel, delayed
import sys
import matplotlib.cm as cm
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC
from glob import glob
import scipy.io as sio
import scipy.signal as signal
import mne.viz as mviz
from mne.stats import permutation_cluster_1samp_test
import ptitprince as pt
import umap

mntNIRS = np.load("NIRSmontage.npy")
from sklearn.decomposition import PCA
def analyzeEntropy():
    
    meas = sys.argv[1]
    def Run(task):#for task in ["Hand", "Mind", "Both"]:

        matplotlib.rcParams.update({'font.size': 14})
        #Slow = np.load("Slow/Data_" + meas + "_" + task + "_.npy",allow_pickle=True).tolist()['train'].reshape(29,-1)
        #Fast = np.load("Fast/Data_" + meas + "_" + task + "_.npy",allow_pickle=True).tolist()['train'].reshape(29,-1)
#        Slow = np.load("Slow/FriedAll_" + meas + "_" + task + "_.npy",allow_pickle=True).reshape(29,-1)
#        Fast = np.load("Fast/FriedAll_" + meas + "_" + task + "_.npy",allow_pickle=True).reshape(29,-1)
        Slow = []
        Fast = []
        SlowTime = []
        FastTime = []
        for t in task:
            Slow += [np.load("../fnirseeg02HzLower/FriedAll_" + meas + "_" + t + "_.npy",allow_pickle=True)]
            Fast += [np.load("../fnirseegHigh{0}Hz/FriedAll_".format(sys.argv[2]) + meas + "_" + t + "_.npy",allow_pickle=True)]
        pTask = ""
        Labsss = []
        if task[0] == "Mind":
            pTask="MA vs BL"
            Labsss = ["BL", "MA"]
            SlowTime += [np.load("../oldfnirseeg02HzLower/BaselineTime_" + meas + ".npy",allow_pickle=True)]
            SlowTime += [np.load("../oldfnirseeg02HzLower/mentalATime_" + meas + ".npy",allow_pickle=True)]
            FastTime += [np.load("../oldfnirseegHigh02Hz/BaselineTime_" + meas + ".npy",allow_pickle=True)]
            FastTime += [np.load("../oldfnirseegHigh02Hz/mentalATime_" + meas + ".npy",allow_pickle=True)]
        elif task[0] == "Hand":
            pTask="LH vs RH MI"
            Labsss = ["LH", "RH"]
            SlowTime += [np.load("../oldfnirseeg02HzLower/LeftHandTime_" + meas + ".npy",allow_pickle=True)]
            SlowTime += [np.load("../oldfnirseeg02HzLower/RightHandTime_" + meas + ".npy",allow_pickle=True)]
            FastTime += [np.load("../oldfnirseegHigh02Hz/LeftHandTime_" + meas + ".npy",allow_pickle=True)]
            FastTime += [np.load("../oldfnirseegHigh02Hz/RightHandTime_" + meas + ".npy",allow_pickle=True)]
        else:
            pTask= "All"
        Slow = np.concatenate(Slow,0)
        Fast = np.concatenate(Fast,0)
        print(Slow.shape,Fast.shape)
        Dat = np.concatenate((Slow,Fast),0)
        SlowTime = np.array(SlowTime)
        FastTime = np.array(FastTime)
        Dat = np.array(Dat).transpose(0,2,3,1)
        print(Dat.shape)
        Datnew = np.zeros_like(Dat)
        for i in range(len(mntNIRS)):
            dist = np.sqrt((mntNIRS[:,0]-mntNIRS[i,0])**2 + (mntNIRS[:,1] - mntNIRS[i,1])**2)
            dist = np.exp(-dist/.01)
            Datnew += dist*Dat
        Dat = Datnew
        accslow = Dat[:29]
        accfast = Dat[29:]
        tscoresS = []
        tscoresF = []
        def normalityOrT(a):
#            reducer = PCA(n_components=1)#n_components=5)#umap.UMAP()
            ts = []
#            aa = a.T.reshape(-1,a.T.shape[-1])
#            aa = reducer.fit_transform(aa)
#            aa = aa.reshape(1,29,2).transpose(0,2,1)
#            print(a.shape)
#            b = a[:,1]#/a[:,1].std()
#            a = a[:,0]#/a[:,0].std()
            b = a[:,1]#/a[:,1].std()
            a = a[:,0]#/a[:,0].std()

#            if meas == "Mean":
#                b = ((b.T)/b.std(1)).T
#                a = ((a.T)/a.std(1)).T
#            else:
#                b = ((b.T)/b.std(1)).T
#                a = ((a.T)/a.std(1)).T
            wilcs = [stats.wilcoxon((a[i].T - b[i].T)) for i in range(a.shape[0])]
            wilcs = np.array(wilcs)
            ts.append([np.mean(wilcs[:,0],0), np.min(wilcs[:,1],0)*2])
#            print(wilcs.shape)
#            ts.append(stats.friedmanchisquare(*(a-b)))
#            ts.append([dcor.distance_correlation(a[:,0],a[:,1]),dcor.independence.distance_covariance_test(a[:,0],a[:,1],num_resamples=1000)[0]])
            return ts

        for a in accslow.T:
            tscoresS += [normalityOrT(a)]#stats.wilcoxon(np.array(a)-.5,alternative="greater")[1]]
        for a in accfast.T:
            tscoresF += [normalityOrT(a)]#stats.wilcoxon(np.array(a)-.5,alternative="greater")[1]]
        tscoresS = np.array(tscoresS).squeeze()
        tscoresF = np.array(tscoresF).squeeze()
        multcomp = 1
        #tscoresS[tscoresS[:,1]>0.05/multcomp,0] = 0
        #tscoresF[tscoresF[:,1]>0.05/multcomp,0] = 0
        meds = abs(-(tscoresS[:,0] - tscoresF[:,0]))
        tscoresS = (tscoresS)[:,1]
        tscoresF = (tscoresF)[:,1]
        SlowTime = SlowTime[:,:,tscoresS > 0.05/multcomp].transpose(0,1,2,4,3)#.mean(2)
        FastTime = FastTime[:,:,tscoresF > 0.05/multcomp].transpose(0,1,2,4,3)#.mean(2)
        Slow =accslow[...,np.logical_or(tscoresS > 0.05/multcomp,tscoresF > 0.05/multcomp)].reshape(29,2,-1)
        Fast =accfast[...,np.logical_or(tscoresS > 0.05/multcomp,tscoresF > 0.05/multcomp)].reshape(29,2,-1) 
#        Slow = #np.median(accslow[...,np.logical_or(tscoresS > 0.05/multcomp,tscoresF > 0.05/multcomp)],-1)#.transpose(0,3,1,2).reshape(-1,2,3)#transpose(0,1,2,4,3)#.mean(2)
#        Fast = #np.median(accfast[...,np.logical_or(tscoresS > 0.05/multcomp,tscoresF > 0.05/multcomp)],-1)#.transpose(0,3,1,2).reshape(-1,2,3)#transpose(0,1,2,4,3)#.mean(2)
        reducer = PCA(n_components=3)
        Slow = Slow.reshape(29*2,-1)
        Slow = (Slow - Slow.mean(0))/Slow.std(0)
        Slow = Slow.reshape(29,2,-1)#reducer.fit_transform(Slow.reshape(29*2,-1)).reshape(29,2,-1)#.sum(-1)
        Slow = (Slow[:,0] - Slow[:,1])#.mean(-1)
#        print("Slow",reducer.explained_variance_ratio_.sum(), Slow.shape[-1],"".join(pTask), stats.wilcoxon(Slow[:]))
        reducer = PCA(n_components=3)
        Fast = Fast.reshape(29*2,-1)
        Fast = (Fast - Fast.mean(0))/Fast.std(0)
        Fast = Fast.reshape(29,2,-1)#reducer.fit_transform(Fast.reshape(29*2,-1)).reshape(29,2,-1)#.sum(-1)
        Fast = (Fast[:,0] - Fast[:,1])#.mean(-1)
#        print("Fast",reducer.explained_variance_ratio_.sum(), Fast.shape[-1],"".join(pTask), stats.wilcoxon(Fast[:]))
        print(np.max(SlowTime),np.max(FastTime), Fast.shape, Slow.shape, "print")
#        SlowTime = (SlowTime.T - SlowTime.T[0]).T
#        FastTime = (FastTime.T - FastTime.T[0]).T
#        SlowTime[SlowTime > 100] = 100
#        FastTime[FastTime > 100] = 100
#        SlowTime = ((SlowTime.T - np.nanmean(SlowTime.T,0))/(1e-8+np.nanstd(SlowTime.T,0))).T
#        FastTime = ((FastTime.T - np.nanmean(FastTime.T,0))/(1e-8+np.nanstd(FastTime.T,0))).T
        SlowTime = SlowTime.mean(2)
        FastTime = FastTime.mean(2)
        SlowTimeM = np.nanmean(SlowTime,1)
        FastTimeM = np.nanmean(FastTime,1)
        SlowTimeS = np.nanstd(SlowTime,1)/np.sqrt(FastTime.shape[1])
        FastTimeS = np.nanstd(FastTime,1)/np.sqrt(FastTime.shape[1])
        print(SlowTimeS.shape,FastTimeS.shape)
        fig,ax = plt.subplots(2,3)
        c = ["red","blue"]
        groups = []
        hue= []
        scores = []
        f2, ax2 = plt.subplots(figsize=(7, 5))
        LabsO = ["Oxy", "Deoxy", "Total"]
        for i in range(1):
            for j in range(3):
                ks = stats.kstest(abs(Fast[:,j]),abs(Slow[:,j]))
                if ks[1] < 0.05/(4*3):
                    addHue = "*"
                else:
                    addHue = ""
                groups += [meas+" {2} {1}".format(Labsss[i],addHue,LabsO[j]) for k in range(len(Slow))]
                hue += ["Slow".format(Labsss[i],LabsO[j]) for k in range(len(Slow))]
                scores += abs(Slow[:,j]).tolist()
                groups += [meas+" {2} {1}".format(Labsss[i],addHue,LabsO[j]) for k in range(len(Fast))]
                hue += ["Fast".format(Labsss[i],LabsO[j]) for k in range(len(Slow))]
                scores += abs(Fast[:,j]).tolist()
        print(len(scores),len(groups), Fast.shape)        
        pSlow = pd.DataFrame(data={"Measure":groups,"Scores":scores,"Wave":hue}) 
        dx = "Measure"; dy = "Scores";ort = "h"; pal = "Set2"; sigma = .2
        pt.RainCloud(data=pSlow,alpha=.4,hue="Wave",x="Measure", y="Scores",palette = pal, bw = sigma, width_viol = .6, ax = ax2, orient = ort)
        f2.tight_layout()
        plt.savefig("SlowRainCloud_{0}_{1}".format(meas,"".join(pTask)))
        plt.close(f2)

        groups = []
        hue= []
        scores = []
        f2, ax2 = plt.subplots(figsize=(7, 5))
        LabsO = ["Oxy", "Deoxy", "Total"]
        for i in range(1):
            for j in range(3):
                groups += [meas+" Fast {1}".format(Labsss[i],LabsO[j]) for k in range(len(Fast))]
                hue += ["{0}".format(Labsss[i],LabsO[j]) for k in range(len(Fast))]
                scores += Fast[:].tolist()
        print(len(scores),len(groups), Fast.shape)        
        pFast = pd.DataFrame(data={"Measure":groups,"Scores":scores,"Wave":hue}) 
        dx = "Measure"; dy = "Scores";ort = "h"; pal = "Set2"; sigma = .2
        pt.RainCloud(data=pSlow,alpha=.4,hue="Wave",x="Measure", y="Scores",palette = pal, bw = sigma, width_viol = .6, ax = ax2, orient = ort)
        f2.tight_layout()
        plt.savefig("FastRainCloud_{0}_{1}".format(meas,"".join(pTask)))
        plt.close(f2)


        for i in range(3):
            for j in range(2):
                ax[0,i].plot(np.arange(SlowTimeM.shape[-1])/SlowTimeM.shape[-1]*80,SlowTimeM[j,i],color=c[j],label=Labsss[j])
                ax[0,i].fill_between(np.arange(SlowTimeM.shape[-1])/SlowTimeM.shape[-1]*80,SlowTimeM[j,i] + SlowTimeS[j,i], SlowTimeM[j,i] - SlowTimeS[j,i], color=c[j],alpha=.3)
                ax[1,i].plot(np.arange(SlowTimeM.shape[-1])/SlowTimeM.shape[-1]*80,FastTimeM[j,i],color=c[j],label=Labsss[j])
                ax[1,i].fill_between(np.arange(SlowTimeM.shape[-1])/SlowTimeM.shape[-1]*80,FastTimeM[j,i] + FastTimeS[j,i], FastTimeM[j,i] - FastTimeS[j,i], color=c[j],alpha=.3)
        ax[0,0].set_ylabel("Slow Wave (A.U.)")
        ax[1,0].set_ylabel("Fast Wave (A.U.)")
        ax[0,0].set_title("Oxy")
        ax[0,1].set_title("Deoxy")
        ax[0,2].set_title("Total")
        ax[0,2].legend()

        fig.tight_layout()
        fig.savefig("TimeSeries_{0}_{1}".format(meas,"".join(pTask)))
        plt.close(fig)
        maskTS = (tscoresS < 0.05/1/multcomp).astype(int).astype(str)
        maskTF = (tscoresF < 0.05/1/multcomp).astype(int).astype(str)
        binMap = np.core.defchararray.add(maskTF, maskTS)
#        binMap[binMap == '11'] = '00'
        colorStrs = ['{0}{1}'.format(i,j) for i in range(2) for j in range(2)]
        Ev = ["Slow", "Fast"]
        Events = ["None", "Slow", "Fast", "Slow and Fast"]
#        Events = ["None", "Slow", "Fast"]
        colorStrs = colorStrs#[:-1]
        str2Num = dict(zip(colorStrs,np.arange(len(colorStrs))))
        bin2Num = np.array([str2Num[i] for i in binMap.ravel()]).reshape(binMap.shape).astype(float)
#        cmap = (matplotlib.cm.get_cmap("GnBu",len(colorStrs)))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",["white","blue","green","red"],N=len(colorStrs))
#        bin2Num[binMap == '0000'] = np.nan
        # NIRS Topo
        fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
        ax2.spines['polar'].set_visible(False)
       # ax.grid(False)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        theta = np.arctan2(mntNIRS[:,1], mntNIRS[:,0])
        R = np.sqrt((mntNIRS**2).sum(1))
        sc = plt.scatter(theta[bin2Num==bin2Num],R[bin2Num==bin2Num],vmin=0,vmax=len(colorStrs), c=bin2Num[bin2Num==bin2Num],cmap=cmap,edgecolors='k')
#        plt.scatter(theta[bin2Num!=bin2Num],R[bin2Num!=bin2Num],s=80,facecolors='none',edgecolors='k')
        cb = plt.colorbar(sc,shrink=0.85, pad = 0.1,ticks=.5+np.arange(0,len(colorStrs)))
        cb.ax.set_yticklabels(Events)
        cb.ax.set_ylabel('', rotation=270)
        plt.title("{0} {1}".format(meas, "".join(pTask)))
        ax2.set_xticklabels(['Right', '', 'Front', '', 'Left', '', 'Posterior', ''])
#        tsMask = np.logical_or(tscoresS > 0.05/1/multcomp,tscoresF >  0.05/1/multcomp)
#        accslow[:,:,:,tsMask] =0 
#        accfast[:,:,:,tsMask] =0 
        Dat = np.concatenate((accslow,accfast),0)
        ent = []
#        def scoreEnt(garbage):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from scipy.stats import rankdata
        sig = []
        for ppp in range(1):
            reducer = PCA(n_components=.999)#n_components=5)#umap.UMAP()
#            labels = np.zeros(len(Dat))
#            labels[29*len(task):] = 1
            trans = reducer.fit_transform((Dat[:,1] - Dat[:,0]).reshape(Dat.shape[0],-1))#rankdata(Dat,axis=1))#,labels)
            print(trans.shape,Dat[:,1].reshape(Dat.shape[0],-1).shape)
            nfeats = np.min(((tscoresS < 0.05/4/multcomp).sum(),(tscoresF < 0.05/4/multcomp).sum()))
            trans = trans[:,:]
            entropy = dcor.distance_correlation(trans[:29*len(task)],trans[29*len(task):])
            ent += [entropy]
            sig += [dcor.independence.distance_covariance_test(trans[:29*len(task)],trans[29*len(task):],num_resamples=10000)[0]]
            #sig += [dcor.independence.distance_correlation_t_test(trans[:29*len(task)],trans[29*len(task):])[0]]
#            Histo = np.concatenate((trans[:29*len(task)],trans[29*len(task):]),1)
#            bins,_ = np.histogramdd(Histo,bins=4)
#            print(bins.shape, trans.shape)
#            bins = bins/bins.sum()
#            MI = []
#            HX = []
#            HY = []
#            for i in range(bins.shape[0]):
#                for j in range(bins.shape[1]):
#                    if bins[i,j].sum() != 0:
#                        HX += [-bins[i,j].sum()*np.log(bins[i,j].sum())]
#                    if bins[:,:,i,j].sum() != 0:
#                        HY += [-bins[:,:,i,j].sum()*np.log(bins[:,:,i,j].sum())]
#            for i in range(bins.shape[0]):
#                for j in range(bins.shape[1]):
#                    for k in range(bins.shape[2]):
#                        for l in range(bins.shape[3]):
#    #                        for m in range(bins.shape[4]):
#    #                            for n in range(bins.shape[5]):
#    #                                if bins[i,j,k,l,m,n] != 0 and bins[i,j,k].sum() != 0 and bins[:,:,l,m,n].sum() != 0:
#    #                                    MI += [bins[i,j,k,l,m,n]]
#    #                                    MI[-1] *= np.log(bins[i,j,k,l,m,n]/(bins[i,j,k].sum()*bins[:,:,l,m,n].sum()))
#    
#                            if bins[i,j,k,l] != 0 and bins[i,j].sum() != 0 and bins[:,:,k,l].sum() != 0:
#                                MI += [bins[i,j,k,l]]
#                                MI[-1] *= np.log(bins[i,j,k,l]/(bins[i,j].sum()*bins[:,:,k,l].sum()))
#    #        return np.sum(MI),trans
#            ent += [np.sum(MI)]#/(np.sum(HX)+np.sum(HY))]
#        accslow[:,:,:,tscoresS < 0.05/4/multcomp] =0 
#        accfast[:,:,:,tscoresF < 0.05/4/multcomp] =0

        entropy = np.median(ent)
        sig = np.median(sig)

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.spines['polar'].set_visible(False)
       # ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        theta = np.arctan2(mntNIRS[:,1], mntNIRS[:,0])
        R = np.sqrt((mntNIRS**2).sum(1))
#        meds = np.median(accslow - accfast,0)
        print(np.max(meds))
        sc = plt.scatter(theta[bin2Num==0],R[bin2Num==0],s=2*20,vmin=0.0,vmax=60.0, c=meds[bin2Num==0],cmap='hot',clip_on=False,linewidth=0)
        sc = plt.scatter(theta[bin2Num==1],R[bin2Num==1],s=2*50,vmin=0.0,vmax=60.0, c=meds[bin2Num==1],cmap='hot',clip_on=False,linewidth=3,edgecolors='k')
        sc = plt.scatter(theta[bin2Num==2],R[bin2Num==2],s=2*50,vmin=0.0,vmax=60.0, c=meds[bin2Num==2],cmap='hot',clip_on=False,linewidth=3,edgecolors='k')
        sc = plt.scatter(theta[bin2Num==3],R[bin2Num==3],s=2*100,vmin=0.0,vmax=60.0, c=meds[bin2Num==3],cmap='hot',clip_on=False,linewidth=3,edgecolors='k')
#        plt.scatter(theta[bin2Num!=bin2Num],R[bin2Num!=bin2Num],s=80,facecolors='none',edgecolors='k')
        cb = plt.colorbar(sc,shrink=0.85, pad = 0.1,ticks=[0.0,  60.0])
#        cb.ax.set_yticklabels(["Fast > Slow","Zero","Fast < Slow"])
        cb.ax.set_ylabel(r'$|Fast - Slow|$', rotation=270)
        plt.title("{0} {1}, Distance Correlation {2:.4f} ({3:.4f})".format(meas, "".join(pTask),entropy, sig))
        ax.set_xticklabels(['Right', '', 'Front', '', 'Left', '', 'Posterior', ''])
        fig.savefig("accTvals/{0}All_FriedMedsNewAll_".format(sys.argv[2]) + meas + "_" + "".join(task)  + "_Topo.png",bbox_inches='tight')
        plt.close(fig)
        ax2.set_title("{0} {1}, Distance Correlation {2:.4f} ({3:.4f})".format(meas, "".join(pTask),entropy, sig))
        fig2.savefig("accTvals/{0}All_FriedNewAll_".format(sys.argv[2]) + meas + "_" + "".join(task)  + "_Topo.png")
        plt.close(fig2)



        fig = plt.figure()
#        E = Parallel(n_jobs=20)(delayed(scoreEnt)(i) for i in range(20))
#        ent = E[0]
#        trans = E[1][0]
        colors = cm.nipy_spectral(np.linspace(0, 1, 2*len(task)))
        for i in range(len(task)):
            plt.scatter(trans[29*(i):29*(i+1),0], trans[29*(i):29*(i+1),1],color=colors[2*i],label="Slow " + task[i])
            plt.scatter(trans[29*(i+1):29*(i+2),0], trans[29*(i+1):29*(i+2),1],color=colors[2*i+1],label="Fast " + task[i])
        plt.legend()
        plt.title("{1} {2} Distance Correlation {0:.4f} ({3:.4f})".format(entropy, meas, "".join(pTask),sig))
        plt.savefig(sys.argv[2] + "All_" + meas + "_" + "".join(task) + "_Scatter.png")
        plt.close(fig)


    Parallel(n_jobs=4)(delayed(Run)(task) for task in [["Hand"], ["Mind"]])#, ["Hand", "Mind"]])

if __name__ == "__main__":
    analyzeEntropy()
