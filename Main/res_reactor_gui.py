import numpy as np

class res_reactor():
    def __init__(self) -> None:
        self.dis_mat=np.array([])   # (3,n), record the dis of TZP for n frames

    def get_v0(self,fps):
        filted_idx=list(map(lambda x:np.where(x<2000)[0], self.dis_mat))
        filted_dis=list(map(lambda x:x[x<2000], self.dis_mat))
        cls_mul_v0=list(map(lambda valid_dis,valis_idx: np.diff(valid_dis[::-1])*(fps*0.5)/ np.diff(valis_idx)[::-1],\
                        filted_dis, filted_idx))
        def determin_v0(v0_list):
            valid_val=v0_list[v0_list>0]
            if valid_val.any():
                return valid_val.mean()
            else:
                return 0
        
        cls_v0 = np.array(list(map(determin_v0,cls_mul_v0)))
        v0=determin_v0(cls_v0)

        return v0

    def get_brake_a(self, v0, vt, light_color, cls_names):
        if v0!=0:
            now_dis=self.dis_mat[:,-1]
            min_idx=np.argmin(now_dis)
            now_min_dis=now_dis[min_idx]
            min_obj=cls_names[min_idx]

            if now_min_dis<2000:
                if min_obj=='Z':
                    if v0>vt:
                        vt=vt
                    else:
                        return 0
                elif min_obj=='T':
                    if light_color=='G':
                        return 0
                    else:
                        vt=0
                else:
                    vt=0
            
            brake_a=(vt+v0)*(vt-v0)/(v0-2*now_min_dis)
        else:
            brake_a=0

        return abs(brake_a)
