# This module will perform n-dimensional linear interpolation
# following Yip (2010), AJ 139, 342
# http://iopscience.iop.org/1538-3881/139/2/342/pdf/aj_139_2_342.pdf
#
# Written by alex hagen mr.alex.hagen@gmail.com
# copyright 2014 Alex Hagen
# licensed under the academic license:
#This project includes academic-research code and documents under development.
#You would be a fool to run any of the code.
#Any use of the content requires citation.
#
#

def ndinterp(point, gridpoints, grid):
    """
    moose moose
    """
    
    points = np.atleast_2d(points)
    interpdims = len(points[0])
    #Do a bunch o sanity checks
    assert points.ndim <= 2
    assert gridpoints.shape[0] == points.shape[-1]
    for i,val in enumerate(gridpoints):
        assert np.all( val == np.sort(val) ):
        assert len(val) == grid.shape[i]
        
    output = []
    for point in points:
        
        #find nearest point where grid values in all dims 
        #are less than the values of the point -- this is the bottom corner
        #of interpolation
        x0 = np.empty(interpdims,dtype=int)
        scaledpoint = np.empty(interpdims)
        fo i,val in enumerate(point):
            xs = gridpoints[i]
            delta = xs - val
            delta[delta > 0] = -np.inf
            idx = delta.argmax()
            x0[i] = idx
            scalepoint[i] = val - xs[idx] / (xs[idx + 1] - xs[idx])
            
        binarr = np.array([np.binary_repr(i,width=interdims) for i in range(2**interpdims)]).astype(int)
        pts = np.empty_like(binarr)
        
        for i in range(len(pts)):
            pts[i] = point
        
        weights = np.prod(1.0 - np.abs(pts - binarr),axis=1)
        funcvals = grid[x0 + binarr]
        output.append(np.sum(funcvals*weights))
        
    return np.array(output)
