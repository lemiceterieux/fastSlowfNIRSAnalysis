import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt
import scipy.stats as stats
import dcor

for meas in ["Mean","SampEn"]:
    Low = []
    High = []
    Low = np.load("../fnirseeg02HzLower/Data_"+meas+".npy",allow_pickle=True).tolist()['train']
#    print(Low.shape)
    High = np.load("../fnirseegHigh02Hz/Data_"+meas+".npy",allow_pickle=True).tolist()['train']

    label = np.load("Low/Data_distEn.npy",allow_pickle=True).tolist()['label']
    mask = np.load("MaskArrMVPA.npy")
    rMatsLow = []
    rMatsHigh = []
    pMatsLow = []
    pMatsHigh = []
    
    mbase = label == 0
    mment = label == 1 
    mlhand = label == 2 
    mrhand = label == 3 
    for i in range(Low.shape[0]):
        chanLow = []
        chanHigh = []
        pchanLow = []
        pchanHigh = []
        for j in range(len(mask)):
            Lbase = Low[mbase][:,[j]]
            Lment = Low[mment][:,[j]]
            Llhand = Low[mlhand][:,[j]]
            Lrhand = Low[mrhand][:,[j]]

            Lbase = Lbase.reshape(29,30,-1)
            Lment = Lment.reshape(29,30,-1)
            Llhand = Llhand.reshape(29,30,-1)
            Lrhand = Lrhand.reshape(29,30,-1)
            Ls = [Lbase, Lment, Llhand, Lrhand]
            Hbase = High[mbase][:,[j]]
            Hment = High[mment][:,[j]]
            Hlhand = High[mlhand][:,[j]]
            Hrhand = High[mrhand][:,[j]]

            Hbase = Hbase.reshape(29,30,-1)
            Hment = Hment.reshape(29,30,-1)
            Hlhand = Hlhand.reshape(29,30,-1)
            Hrhand = Hrhand.reshape(29,30,-1)
            Hs = [Hbase, Hment, Hlhand, Hrhand]
  
            clow = []
            ddc = dcor.distance_correlation
            didct = dcor.independence.distance_covariance_test

            cL = []
            cH = []
            pcL = []
            pcH = []
            for k in range(0,4):
               for l in range(k+1,4):
                   cL.append(ddc(Ls[k][i], Ls[l][i]))
                   cH.append(ddc(Hs[k][i], Hs[l][i]))
                   pcL.append(0)#didct(Ls[k][i], Ls[l][i],num_resamples=280)[0])
                   pcH.append(0)#didct(Hs[k][i], Hs[l][i],num_resamples=280)[0])

            chanLow += [cL]
            chanHigh += [cH]
            pchanLow +=  [pcL]
            pchanHigh += [pcH]
    
        rMatsLow += [chanLow]
        rMatsHigh += [chanHigh]
        pMatsLow += [pchanLow]
        pMatsHigh += [pchanHigh]
    Labs = ["Base and  Mental Arithmetic", "Base and Left Hand Imqgery", "Base and Right Hand Imagery", "Mental Aritmetic and Left Hand Imagery", "Mental Arithmetic and Right Hand Imagery", "Left Hand Imagery and Right Hand Imagery"]
    LabPlots = ["bm", "bl", "br", "ml", "mr", "lr"]
    rMatsLow = np.array(rMatsLow)
    rMatsHigh = np.array(rMatsHigh)
    rMatsHigh = rMatsHigh.transpose(0,2,1)
    rMatsLow = rMatsLow.transpose(0,2,1)
    rMatsHighnew = np.zeros_like(rMatsHigh)
    mntNIRS = np.load("NIRSmontage.npy")
    for i in range(len(mntNIRS)):
        dist = np.sqrt((mntNIRS[:,0]-mntNIRS[i,0])**2 + (mntNIRS[:,1] - mntNIRS[i,1])**2)
        dist = np.exp(-dist/.1)
        rMatsHighnew += dist*rMatsHigh
    rMatsHigh = rMatsHighnew
    rMatsLownew = np.zeros_like(rMatsLow)
    for i in range(len(mntNIRS)):
        dist = np.sqrt((mntNIRS[:,0]-mntNIRS[i,0])**2 + (mntNIRS[:,1] - mntNIRS[i,1])**2)
        dist = np.exp(-dist/.1)
        rMatsLownew += dist*rMatsLow
    rMatsLow = rMatsLownew
    rMatsHigh = rMatsHigh.transpose(0,2,1)
    rMatsLow = rMatsLow.transpose(0,2,1)

    print(rMatsLow.shape)
    PearsonpCh = []
    FriedmanCh = []
    FriedmanP = []
    DCORRSA = []
    for i in range(36):
        c = []
        for ii in range(6):
            c.append(36*stats.wilcoxon(rMatsLow[:,i,ii], rMatsHigh[:,i,ii])[1])
        PearsonpCh.append(c)
        fried = stats.friedmanchisquare(*(rMatsLow[:,i] - rMatsHigh[:,i]).T)
        FriedmanCh.append(fried[0])
        FriedmanP.append(fried[1])
#        DCORRSA.append(didct(rMatsLow[:,i], rMatsHigh[:,i])[0])#,num_resamples=1000)[0])
    np.save("Pearson"+meas,PearsonpCh)
#    np.save("dcorRSA"+meas,PearsonpCh)
    mntNIRS = np.load("NIRSmontage.npy")
    name = [i for i in range(len(mntNIRS))]
#
    mask = np.array(PearsonpCh) >0.05/6
    maskfr = np.array(FriedmanP) < 0.05/36
    fried = np.array(FriedmanCh)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.spines['polar'].set_visible(False)
#    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticklabels(['Right', '', 'Front', '', 'Left', '', 'Posterior', ''])
    theta = np.arctan2(mntNIRS[:,1], mntNIRS[:,0])
    R = np.sqrt((mntNIRS**2).sum(1))
    plt.scatter(theta[maskfr], R[maskfr], c=fried[maskfr],vmin=0,vmax=40,cmap='hot',clip_on=False,linewidth=3,edgecolors='k',s=2*50)
    sc = plt.scatter(theta[~maskfr],R[~maskfr], c=fried[~maskfr],vmin=0,vmax=40,cmap='hot',s=2*20)
    if meas != "Mean":
        cb = plt.colorbar(sc,shrink=0.85, pad = 0.1,ticks=[0, 40])
        cb.ax.set_yticklabels(["{0:d}".format(0),"{0:d}".format(40)])
    plt.title("Friedman test group RDM {0}".format(meas))
    plt.tight_layout()
    fig.savefig("FriedRSA_" + meas + "_" + LabPlots[k] + "_Topo.png")
    plt.close(fig)
#    maskddic = np.array(DCORRSA) >0.05
    print(mask)
    import mne.viz as mviz

    # NIRS Topo
#    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#    ax.spines['polar'].set_visible(False)
#    ax.set_xticklabels([])
#    ax.set_yticklabels([])
#    ax.set_xticklabels(['Right', '', 'Front', '', 'Left', '', 'Posterior', ''])
#    theta = np.arctan2(mntNIRS[:,1], mntNIRS[:,0])
#    R = np.sqrt((mntNIRS**2).sum(1))
#    plt.scatter(theta[maskddic], R[maskddic], c="red",label="Non-Significant")
#    plt.scatter(theta[~maskddic],R[~maskddic], c="green", label="Significant")
#    plt.legend()
#    plt.title("Distance Correlation between RSA RDMs of Low and High for {0}".format(meas))
#    plt.tight_layout()
#    fig.savefig("RSAdCor_" + meas + "_Topo.png")
#    plt.close(fig)
    for k in range(6):
        # NIRS Topo
    #    im, _ = mviz.plot_topomap(PearsonpCh, mntNIRS,  show=False, show_names=True,
    #            image_interp=None,names=name, contours=0, cmap='Greens_r', vmax=0.05, mask=mask)
    #    cb = plt.colorbar(im)
    #    cb.remove()
        print(mntNIRS.shape)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.spines['polar'].set_visible(False)
       # ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticklabels(['Right', '', 'Front', '', 'Left', '', 'Posterior', ''])
        theta = np.arctan2(mntNIRS[:,1], mntNIRS[:,0])
        R = np.sqrt((mntNIRS**2).sum(1))
        plt.scatter(theta[mask[:,k]], R[mask[:,k]], c="red",label="Non-Significant")
        plt.scatter(theta[~mask[:,k]],R[~mask[:,k]], c="green", label="Significant")
        plt.legend()
        plt.title("Paired Wilcoxon test group RDM {0}, {1}".format(meas, Labs[k]))
        plt.tight_layout()
        fig.savefig("Pearson_" + meas + "_" + LabPlots[k] + "_Topo.png")
        plt.close(fig)
