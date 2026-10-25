"""Name/sign conversion between the Fusion CAD joints and reviewed V15 policy."""
import numpy as np

LEG_MAP={'FL':'FL','FR':'FR','RL':'BL','RR':'BR'}
KINDS={'HAA':'hip_roll','HFE':'hip_pitch','KFE':'knee'}
SIGNS={'FL':(-1.,-1.,1.),'FR':(-1.,1.,-1.),
       'RL':(1.,-1.,1.),'RR':(1.,1.,-1.)}

def joint_mapping(policy_names):
    names=tuple(policy_names)
    if len(set(names))!=len(names):raise ValueError('Duplicate policy joint')
    physical=[];sign=[]
    for name in names:
        try:
            leg,kind=name.split('_')
            physical.append(LEG_MAP[leg]+'_'+KINDS[kind])
            sign.append(SIGNS[leg][('HAA','HFE','KFE').index(kind)])
        except (ValueError,KeyError) as error:
            raise ValueError('Unknown policy joint '+name) from error
    return tuple(physical),np.array(sign)

def physical_values(values,sign):
    values=np.asarray(values);sign=np.asarray(sign)
    if values.shape[-1:]!=sign.shape:raise ValueError('Joint vector shape mismatch')
    return values*sign

def policy_values(values,sign):
    return physical_values(values,sign)

def physical_command_fields(position,effort,sign):
    """Keep native float lists at the ROS/PhysX and JSON transport boundary."""
    return tuple(physical_values(values,sign).tolist() for values in (position,effort))
