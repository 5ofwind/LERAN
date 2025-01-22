import logging
import math

from models.modules.subnet_constructor import subnet

from models.modules.inv_arch import *

logger = logging.getLogger('base')

####################
# define network
####################

def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    subnet_type = which_model['subnet_type']
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'

    down_num = int(math.log(opt_net['scale'], 2))

    netG = Net1(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['block_num'], down_num)

    return netG

def define_G2(opt):
    opt_net = opt['network_G2']
    which_model = opt_net['which_model_G']
    subnet_type = which_model['subnet_type']
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'

    down_num = int(math.log(opt_net['scale'], 2))

    netG2 = Net2(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['block_num'], down_num)

    return netG2