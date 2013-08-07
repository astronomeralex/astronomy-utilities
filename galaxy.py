#This module implements classes to facilitate SED fitting with GalMC

class Flux(object):
    """
    this flux class will contain a single photometric measurement
    """
    def __init__(self,flux,err,bandpass,silent=False):
        self.flux = flux
        self.err = err
        self.filter = Filter(bandpass,silent)
    
class Filter(object):
    """
    the class contains information on the filter, including the name,
    location of the file describing the transmission, and the central wavelength
    """
    def __init__(self,name,silent = False):
        self.name = name
        self.transfile = 'filtercurves/'+self.name + '.res' #the location of the transmission file
        #now check and see if the transmission file exists
        if os.path.exists(self.transfile):
            transfile = open(self.transfile)
            transmission = transfile.readlines()
            transfile.close()
            waves = []
            trans = []
            for i,line in enumerate(transmission):
                transmission[i]=line.split()
                transmission[i][0] = float(transmission[i][0])
                transmission[i][1] = float(transmission[i][1])
                waves.append(transmission[i][0])
                trans.append(transmission[i][1])
            self.central = waves[trans.index(max(trans))]
            trans = np.array(trans)
            waves = np.array(waves)
            #normalize transmission
            trans = trans / trans.max()
            self.transmission = trans
            self.waves = waves
            #now find longest wavelength at 10% transmission
            #this is useful for determining if NIR filters could be contaminated by
            #3.3um PAH feature
            self.long10 = waves[np.where(trans > 0.1)[0][-1]]
            
        else:
            if not silent:
                print "Can't find transmission file at " + self.transfile
