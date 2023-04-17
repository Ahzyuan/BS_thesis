import numpy as np

A4=(297,210)
A3=(420,297)
A2=(594,420)
A1=(840,597)

def main(paper_size, margin, type='square', dot_interval=0.25):
    '''
    paper_size: (weight, height)
    margin: the margin 
    type: dot or square
    dot_interval: the interval between two dots, present as the radio of the dot's d
    '''
    c=np.arange(5,100,1)
    r=np.arange(5,100,1)
    d=np.arange(10,100,5)
    limit_weight=paper_size[0]-margin*2
    limit_height=paper_size[1]-margin*2

    if type=='square':
        for item_size in d:
            left_weight=limit_weight-item_size*c
            left_height=limit_height-item_size*r
            pick_c=c[left_weight>0]
            pick_r=r[left_height>0]
            if pick_r.any() and pick_c.any():
                print('square_L:{}mm column: {}~{} row: {}~{}'.format(item_size, pick_c[0], pick_c[-1], pick_r[0], pick_r[-1]))
    else:
        for item_size in d:
            left_weight=limit_weight-dot_interval*item_size*((1/dot_interval+1)*c-1)
            left_height=limit_height-dot_interval*item_size*((1/dot_interval+1)*r-1)
            pick_c=c[left_weight>0]
            pick_r=r[left_height>0]
            if pick_r.any() and pick_c.any():
                print('dot_d:{}mm column: {}~{} row: {}~{}'.format(item_size, pick_c[0], pick_c[-1], pick_r[0], pick_r[-1]))

main(A3,15)

            
