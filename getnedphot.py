# Example: Retrieve photometry as an SED tabulation for NGC 1275
#http://nedwww.ipac.caltech.edu/cgi-bin/nph-datasearch?search_type=Photometry&of=xml_main&objname=NGC+1275
#
#This module will facillitate fetching photometry data from NED
#
#

import urllib
import urllib2
import string
from astropy.io.votable import parse_single_table

def single_object_nedphot(obj,savepath='/astro/grads/arh5361/ned_photometry/'):
    """
    This will fetch the photometry for the given object from NED
    and will save the xml file in the given directory
    obj is a string that is the name of the object that NED will recognize
    The xml file will be saved in savepath
    """
    values = {'search_type' : 'Photometry', 'of' : 'xml_main' , 'objname' :obj}
    url = 'http://nedwww.ipac.caltech.edu/cgi-bin/nph-datasearch?'
    data = urllib.urlencode(values)
    response = urllib2.urlopen(url + data)
    page = response.read()
    outfile = open(savepath + obj_format(obj)+'.xml','w')
    outfile.write(page)
    outfile.close()
    
def obj_format(obj):
    """
    obj is a string that is the name of a ned object
    this function will format the string into something that is
    suitable for saving
    """
    #first replace spaces with underscores
    obj = obj.replace(" ","_")
    #this might expand in the future?
    
    return obj
    
def nedphot(objs,savepath='/astro/grads/arh5361/ned_photometry/',overwrite=False):
    """
    this takes in a list of strings that are objects neds will recognize
    it will see if the photometry exists in the local photometry database
    and if not, it will fetch the photometry from ned
    it will also read in the photometry xml files and return a list where
    each entry is an astropy table object containing the photometry
    """
    output = []
    for obj in objs:
        if overwrite:
            single_obj_nedphot(obj,savepath)
            table = parse_single_table(savepath + obj_format(obj)+'.xml')
        else:
            try:
                table = parse_single_table(savepath + obj_format(obj)+'.xml')
            except IOError:
                single_object_nedphot(obj,savepath)
                table = parse_single_table(savepath + obj_format(obj)+'.xml')
        output.append(table)
    return output
