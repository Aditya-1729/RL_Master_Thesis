import numpy as np
import matplotlib.pyplot as plt

# class reward_multiplier:
#     def __init__ (self,c_weight, f_w_weight,window):
#         self.f_w_weight= f_w_weight         #force_in_window_weight
#         self.c_weight = c_weight            #contact_fractional_weight
#         self.f_t =1.5                       #force_target
#         self.f_l=1                          #force_lower_limit    
#         self.f_h=5                          #force_higher_limit
#         self.f_cap=30                       #foce_upper_cap
#         self.f_contact=0.3                  #min_contact_force
#         self.window=window                  #whether to use force window
#         self.f_r=True                       #force_reward
    
#     def calc_multiplier(self):
#         if self.window==True:
#             f_limit_weight = 1 - self.f_w_weight - self.c_weight
#             # print(c_weight, f_w_weight, f_limit_weight)
#             min_reward = -1
            
#             #calculations done for f_cap=30
#             x = min_reward/(self.c_weight - self.f_w_weight- f_limit_weight)
#             self.r_c = self.c_weight*x
#             r_w = self.f_w_weight*x
#             r_l = f_limit_weight*x
#             print(f"r_c:{self.r_c}, r_w:{r_w}, r_l:{r_l}")

#             self.alpha_w = r_w/np.square(self.f_cap-self.f_t)
#             self.alpha_h = r_l/(self.f_cap - self.f_h)**2
#             self.alpha_l = self.alpha_h
#             print(f"alpha_w:{self.alpha_w}, alpha_l:{self.alpha_l}, alpha_h: {self.alpha_h}")
#             return self.r_c, self.alpha_l, self.alpha_w, self.alpha_h
#         else:
#             c_r=15                                                      #contact_reward
#             tf_m = 1/self.c_weight                                      #target_force_mult, heuristically selected
#             horizon=2000
#             if self.f_r==True:
#                 self.norm_factor = horizon/(horizon*(c_r + tf_m))
#             else:
#                 self.norm_factor = 1/c_r
#             print(f"norm_factor:{self.norm_factor}")
#             min_reward = -1
#             f_m = ((-1/self.norm_factor)*(min_reward) +c_r)/((self.f_cap-self.f_t)**2)  #minimum reward = max penalty with no force rewards
#             print(f"fm:{f_m}")
#             return f_m, c_r, tf_m, self.norm_factor


# class quadratic(reward_multiplier):
#     def __init__ (self,c_weight, reward_scale, f_w_weight, window, force_reward):
#         self.force_reward=True
#         self.reward=0
#         self.window_penalty=0
#         self.l_penalty = 0
#         self.h_penalty=0
#         self.reward_scale=reward_scale
#         super().__init__(c_weight, f_w_weight,window)
        
            
#     def calc_reward(self,x):
#         if self.window==True:
#             self.r_c, self.alpha_l, self.alpha_w, self.alpha_h = self.calc_multiplier()
#             window_penalty = self.alpha_w*np.square(x-1.5)
#             self.reward -= self.alpha_w*np.square(x-1.5)
#             print(f"reward_window:{self.reward}")
#             if x>self.f_h:
#                 h_penalty = self.alpha_h*(x-self.f_h)**2
#                 self.reward-=h_penalty
#                 print(f"reward_exc:{-self.alpha_h*np.square(x-self.f_h)}")
#             if x<1. and x>0.3:
#                 l_penalty=self.alpha_l*np.square(x-1)
#                 self.reward-=l_penalty
#             if x>=0.3:
#                 self.reward+=self.r_c
#                 print(f"r_c:{self.r_c}")
#             if x>self.f_cap:
#                 self.reward = -1
#             print(f"total_reward:{self.reward}")
#             # return window_penalty, l_penalty, h_penalty, self.r_c, reward*self.reward_scale
#             return self.reward
#         else:
#             self.alpha_h, self.r_c, self.alpha_w, self.norm_factor =self.calc_multiplier()
#             if x>0.3:
#                 self.reward+=self.r_c
#             self.alpha_l=self.alpha_h
#             print(f"alpha_h:{self.alpha_h}")
#             if x>0.3 and x!=self.f_t:
#                 force_penalty = self.alpha_h*(x-self.f_t)**2
#                 self.reward-=force_penalty
#                 # print(f"force_penalty:{force_penalty},reward:{reward}")
#             if self.force_reward==True:
#                 print(f"alpha_w:{self.alpha_w}")
#                 if x>=1 and x<5:
#                     # print(f"alpha_w:{alpha_w}")
#                     self.reward+=self.alpha_w
#             # print(f"norm_factor:{norm_factor}")
#             self.reward = self.norm_factor*self.reward*self.reward_scale
#             if x >self.f_cap:
#                 self.reward=-1
#             return self.reward

# for x in np.arange(0.31,40,0.1):
#     print(quadratic(c_weight=0.5, reward_scale=1, f_w_weight=0, window=False, force_reward=True).calc_reward(x))


# class plot_reward:
#     def __init__ (self):

#         self.x_values = np.arange(0.31,40,0.1)
#         self.reward_scale=1
#         self.c_weights= [0.333]
#         self.f_w_weights = [0.2]
    
#     def plot(self):
#         # Plot the initial quadratic polynomial
#         fig, ax = plt.subplots()
#         for c in self.c_weights:
#             for f in self.f_w_weights:
#                 w_p=[]
#                 h_p=[]
#                 r=[]
#                 R=[]
#                 for x in self.x_values:
#                     window_penalty, l_penalty, h_penalty, r_c, reward = quadratic(c,1,f,True).calc_reward(x)
#                     w_p.append(window_penalty*-1 + r_c)
#                     h_p.append(h_penalty*-1)
#                     r.append(reward)
#                     # R.append(reward_3(x, c, reward_scale=reward_scale, window=False))
#                     # R.append(reward_2(x, 0.3, reward_scale=reward_scale, window=False))
#                 ax.plot(self.x_values, w_p, label=f"wp+r_c: c_w:{c},f_w:{f}")
#                 ax.plot(self.x_values,r, label=f"reward: c_w:{c},f_w:{f}")
#                 # ax.plot(x_values,R, label=f"reward: c_w:{c}")
#                 ax.legend()
#             ax.set_title('Quadratic Plot')
#         plt.show()



def calc_multiplier(c_weight, f_w_weight, f_t =11.5, f_l=11, f_h=12, f_cap=30, f_contact=0.3, window=True, f_r=True):
    if window==True:
        f_limit_weight = 1 - f_w_weight - c_weight
        # print(c_weight, f_w_weight, f_limit_weight)
        min_reward = -1
        #calculations done for f_cap=30
        x = min_reward/(c_weight - f_w_weight- f_limit_weight)
        r_c = c_weight*x
        r_w = f_w_weight*x
        r_l = f_limit_weight*x
        print(f"r_c:{r_c}, r_w:{r_w}, r_l:{r_l}")

        initial_alpha_w = r_w/np.square(f_cap-f_t)
        initial_alpha_h = r_l/(f_cap - f_h)**2
        initial_alpha_l = initial_alpha_h
        print(f"alpha_w:{initial_alpha_w}, alpha_l:{initial_alpha_l}, alpha_h: {initial_alpha_h}")
        return r_c, initial_alpha_l, initial_alpha_w, initial_alpha_h
    else:
        c_r=15
        tf_m = 1/c_weight
        horizon=2000
        if f_r==True:
            norm_factor = horizon/(horizon*(c_r + tf_m))
        else:
            norm_factor = 1/c_r
        print(f"norm_factor:{norm_factor}")
        min_reward = -1
        f_m = ((-1/norm_factor)*(min_reward) +c_r)/((f_cap-f_t)**2)
        print(f"fm:{f_m}")
        return f_m, c_r, tf_m, norm_factor

def quadratic(x, r_c, alpha_l, alpha_h, alpha_w, norm_factor=1, reward_scale=1, window=True, f_t=1.5, f_h=5, f_cap=30, force_reward=True):
    reward=0
    window_penalty=0
    l_penalty = 0
    h_penalty=0
    if window==True:
        window_penalty = alpha_w*np.square(x-1.5)
        reward -= alpha_w*np.square(x-1.5)
        print(f"reward_window:{reward}")
        if x>f_h:
            h_penalty = alpha_h*(x-f_h)**2
            reward-=h_penalty
            print(f"reward_exc:{-alpha_h*np.square(x-f_h)}")
        if x<1. and x>0.3:
            l_penalty=alpha_l*np.square(x-1)
            reward-=l_penalty
        if x>=0.3:
            reward+=r_c
            print(f"r_c:{r_c}")
        if x>f_cap:
            reward = -1
        print(f"total_reward:{reward}")
        return window_penalty, l_penalty, h_penalty, r_c, reward*reward_scale
    else:
        if x>0.3:
            reward+=r_c
        alpha_l=alpha_h
        print(f"alpha_h:{alpha_h}")
        if x>0.3 and x!=f_t:
            force_penalty = alpha_h*(x-f_t)**2
            reward-=force_penalty
            # print(f"force_penalty:{force_penalty},reward:{reward}")
        if force_reward==True:
            print(f"alpha_w:{alpha_w}")
            if x>=1 and x<5:
                # print(f"alpha_w:{alpha_w}")
                reward+=alpha_w
        # print(f"norm_factor:{norm_factor}")
        reward = norm_factor*reward*reward_scale
        if x >f_cap:
            reward=-1
        return reward

def reward_4(x, c_weight=0.5, f_w_weight=0.4, reward_scale=1):
    r_c, initial_alpha_l, initial_alpha_w, initial_alpha_h = calc_multiplier(c_weight, f_w_weight, f_cap=30)
        
    #reward_calculation    
    window_penalty, l_penalty, h_penalty, r_c, reward = quadratic(x, r_c, \
                                            initial_alpha_l, initial_alpha_h, initial_alpha_w, reward_scale=1, f_cap=30)
    
    return window_penalty, l_penalty, h_penalty, r_c, reward

x_values = np.arange(0.31,4,0.1)

def reward_3(x, c_weight=0.5, reward_scale=1, window=False):
    f_m, c_r, tf_m, n = calc_multiplier(c_weight, f_w_weight=0, \
                                                  f_t =1.5, f_l=1, f_h=5, f_cap=30, f_contact=0.3, window=window, f_r=True)
    reward = quadratic(x, c_r, f_m, f_m, tf_m, norm_factor=n, \
              reward_scale=reward_scale, window=False, f_t=1.5, f_h=5, f_cap=30, force_reward=True)
    return reward

def reward_2(x, c_weight=0.5, reward_scale=1, window=False):
    f_m, c_r, tf_m, n = calc_multiplier(c_weight, f_w_weight=0, \
                                                  f_t =1.5, f_l=1, f_h=5, f_cap=30, f_contact=0.3, window=window, f_r=False)
    reward = quadratic(x, c_r, f_m, f_m, tf_m, norm_factor=1/c_r, \
              reward_scale=reward_scale, window=False, f_t=1.5, f_h=5, f_cap=30, force_reward=False)
    return reward

def plot_reward4():
    x_values = np.arange(0.31,40,0.1)
    reward_scale=1
    c_weights= [0.333,0.4]
    f_w_weights = [0.2]

    # Plot the initial quadratic polynomial
    fig, ax = plt.subplots()
    for c in c_weights:
        for f in f_w_weights:
            w_p=[]
            h_p=[]
            r=[]
            R=[]
            for x in x_values:
                w_p.append(reward_4(x,c_weight=c,f_w_weight=f)[0]*-1 + reward_4(x,c_weight=c,f_w_weight=f)[3])
                h_p.append(reward_4(x,c_weight=c,f_w_weight=f)[2]*-1)
                r.append(reward_4(x,c_weight=c,f_w_weight=f)[4])
                # R.append(reward_3(x, c, reward_scale=reward_scale, window=False))
                # R.append(reward_2(x, 0.3, reward_scale=reward_scale, window=False))
            ax.plot(x_values, w_p, label=f"wp+r_c: c_w:{c},f_w:{f}")
            ax.plot(x_values,r, label=f"reward: c_w:{c},f_w:{f}")
            # ax.plot(x_values,R, label=f"reward: c_w:{c}")
            ax.legend()
        ax.set_title('Quadratic Plot')
    plt.show()

def plot_reward3():
    x_values = np.arange(0.31,40,0.1)
    c_weight=0.2
    reward_scale=1

    c_weights= [1.5]
 

    # Plot the initial quadratic polynomial
    fig, ax = plt.subplots()
    for c in c_weights:
        r=[]
        R = []
        for x in x_values:
            r.append(reward_3(x, c, reward_scale=reward_scale, window=False))
            R.append(reward_2(x, c, reward_scale=reward_scale, window=False))
        ax.plot(x_values,r, label=f"reward: c_w:{c}")
        ax.plot(x_values,R, label=f"reward: c_w:{c}")
        ax.legend()
        ax.set_title('Quadratic Plot')
    plt.show()



if __name__=="__main__":
    plot_reward4()


