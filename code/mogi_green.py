# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:53:45 2024

@author: cdrg
"""

import numpy as np

from coordinate import cart_2_pol, pol_2_cart
                     
def mogi_func(x, y, z, \
              source_x, source_y, source_z, source_radius,\
              poissons_ratio, shear_modulus):
    
    '''

    Parameters
    ----------
    x : float, TWD97 x
    y : float, TWD97 y
    z : float, TWD97 z

    Returns
    -------
    None.
    
    '''
        
    # Shift center
    xxn = x - source_x 
    yyn = y - source_y

    # Convert to polar coordinates
    [rho, phi] = cart_2_pol(xxn,yyn)

    # Distance from source
    R = np.sqrt((source_z)**2+(rho)**2)

    # Displacements
    Ur = source_radius**3 * (1.0-poissons_ratio) * rho / (shear_modulus * R**3)
    Uz = source_radius**3 * (1.0-poissons_ratio) * source_z / (shear_modulus * R**3)
    
    [Ux, Uy] = pol_2_cart(Ur,phi)
    return [Ux, Uy, Uz]

