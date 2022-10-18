import numpy as np
import gc
from scipy.interpolate import UnivariateSpline
from scipy import optimize

from sklearn.metrics import accuracy_score

def step_fun(x):
    return 1.0*(x>0)

class Model_dist(object):
    
    def __init__(self, x, n_samples=100):
        xa = np.min(x)
        xb = np.max(x)
        
        self.data = x
        self.x_samples = np.linspace(xa, xb, n_samples, endpoint=True)
        
        self.F_samples = np.zeros(n_samples)
        for i in range(self.x_samples.shape[0]):
            self.F_samples[i] = np.sum((self.data<=self.x_samples[i])*1.0)/self.data.shape[0]
        self.cdf = UnivariateSpline(self.x_samples, self.F_samples, k=3, s=0)
        
        self.pdf = self.cdf.derivative()
        self.f_samples = np.zeros(n_samples)
        for i in range(self.f_samples.shape[0]):
            self.f_samples[i] = self.pdf(self.x_samples[i])

class Model_acc(object):
    
    def __init__(self, dist):
        self.dist = dist
        self.tra = dist.x_samples[0]
        self.trb = dist.x_samples[-1]
        
    def set_param(self, alpha, t0):
        self.alpha = alpha
        self.t0 = t0
    
    def query_acc(self, t):
        eta = 1e-8 # numeric protector
        
        n_u = 1 - self.alpha
        n_s1 = self.alpha*(1. - self.dist.cdf(self.t0))
        n_s0 = self.alpha - n_s1
        
        n_u_0 = (1.-self.alpha)*self.dist.cdf(t)
        n_u_1 = (1.-self.alpha)-n_u_0
        
        n_s1_0 = self.alpha*(self.dist.cdf(t)-self.dist.cdf(self.t0))*step_fun(t-self.t0)
        n_s1_1 = n_s1 - n_s1_0
        
        n_s0_0 = self.alpha*self.dist.cdf(t)-n_s1_0
        n_s0_1 = n_s0 - n_s0_0
        
        acc = max(n_s0_0,n_s0_1) + max(n_s1_0, n_s1_1) + max(n_u_0, n_u_1)
        
        n_total_0 = n_s0_0 + n_s1_0 + n_u_0
        n_total_1 = n_s0_1 + n_s1_1 + n_u_1
        maj = max(n_total_0, n_total_1)
        
        ### derivative
        d_u0_dt0 = 0
        d_u0_da = -self.dist.cdf(t)
        
        d_u1_dt0 = 0 
        d_u1_da = -1.0-d_u0_da
        
        d_10_dt0 = -self.alpha*self.dist.pdf(self.t0)*step_fun(t-self.t0)
        d_10_da = (self.dist.cdf(t)-self.dist.cdf(self.t0))*step_fun(t-self.t0)
        
        d_11_dt0 = -self.alpha*self.dist.pdf(self.t0) - d_10_dt0
        d_11_da = 1-self.dist.cdf(self.t0) - d_10_da
        
        d_00_dt0 = -d_10_dt0
        d_00_da = self.dist.cdf(t) - d_10_da
        
        d_01_dt0 = self.alpha*self.dist.pdf(self.t0)- d_00_dt0
        d_01_da = self.dist.cdf(self.t0) - d_00_da
        
        max_select = np.array(\
                    [\
                      step_fun(n_u_0-n_u_1), step_fun(n_u_1-n_u_0),\
                      step_fun(n_s0_0-n_s0_1), step_fun(n_s0_1-n_s0_0),\
                      step_fun(n_s1_0-n_s1_1), step_fun(n_s1_1-n_s1_0),\
                     ]\
                    )
        
        dacc_dt0 = np.inner(max_select,\
                            np.array([d_u0_dt0, d_u1_dt0, d_00_dt0, d_01_dt0, d_10_dt0, d_11_dt0])
                           )
        
        dacc_da = np.inner(max_select,\
                            np.array([d_u0_da, d_u1_da, d_00_da, d_01_da, d_10_da, d_11_da])\
                           )        
        
        return acc, maj, [dacc_da, dacc_dt0]

    def query_acc_list(self, t_list):
        acc = []
        maj = []
        dacc_da = []
        dacc_dt0 = []
        for xi in t_list:
            acc_i, maj_i, der = self.query_acc(xi)
            acc.append(acc_i)
            maj.append(maj_i)
            dacc_da.append(der[0])
            dacc_dt0.append(der[1])
            
        acc = np.array(acc)
        maj = np.array(maj)
        accdev = acc-maj
        dacc_da = np.array(dacc_da)
        dacc_dt0 = np.array(dacc_dt0)
        
        return {
            "lbd":t_list,
            "acc":acc,
            "maj":maj,
            "accdev":accdev,
            "dacc_da":dacc_da,
            "dacc_dt0":dacc_dt0,
        }
    
    def loss_fun(self, res):
        n_acc = len(res["lbd"])
        res_fit = self.query_acc_list(res["lbd"])
        loss = 1./n_acc*0.5*np.sum((res_fit["acc"]-res["acc"])**2) #l2
        #loss = 1./n_acc*np.sum(np.abs(res_fit["acc"]-res["acc"])) #l1
        
        #df only for l2
        d_loss_da = 1./n_acc*np.inner((res_fit["acc"]-res["acc"]), res_fit["dacc_da"])
        d_loss_dt0 = 1./n_acc*np.inner((res_fit["acc"]-res["acc"]), res_fit["dacc_dt0"])
        
        return loss , np.array([d_loss_da, d_loss_dt0])
    
    ## this is a revised version of fit function
    ## 2-level brute force minimization is used
    ## modified by Yuzi He, 2020-06-04
    def fit(self, res):
        bounds = [(0.0, 1.0),(self.tra, self.trb)] 
        
        def fun(x):
            self.set_param(x[0], x[1])              
            return self.loss_fun(res)[0]
        def d_fun(x):
            self.set_param(x[0], x[1])
            return self.loss_fun(res)[1]
        
        #x0 = [0.4, 0.4*(self.tra+self.trb)]
        t_len = self.trb - self.tra
        x0 = [np.random.uniform(0.2,0.8), np.random.uniform(self.tra + 0.2*t_len, self.tra + 0.8*t_len)]
        bounds = [[0.0,1.0], [self.tra, self.trb]]
        
        if(False):
            #opt_res = optimize.minimize(fun, x0, method='L-BFGS-B', bounds=bounds, jac=None, options={"disp":True})
            opt_res = optimize.minimize(fun, x0, method='SLSQP', bounds=bounds, jac=None, options={"disp":False})
        
            rms = opt_res["fun"]**0.5
            a_opt = opt_res["x"][0]
            t0_opt = opt_res["x"][1]
            converge = opt_res["success"]
        
            self.set_param(a_opt, t0_opt) 
        
            if(not converge): print("Fitting fail to converge!")
        
        if(True):
            Ns = 40
            ## first level
            opt_res = optimize.brute(fun, bounds, Ns=Ns)
            
            ## second level
            dx0 = (bounds[0][1]-bounds[0][0])/Ns
            dx1 = (bounds[1][1]-bounds[1][0])/Ns
            bounds1 = [(max(opt_res[0]-dx0, 0), min(opt_res[0]+dx0, 1)),\
                       (max(opt_res[1]-dx1, self.tra), min(opt_res[1]+dx1, self.trb))]
            opt_res = optimize.brute(fun, bounds1, Ns=Ns)
            
            rms = fun(opt_res)**0.5
            a_opt = max(min(opt_res[0],1),0)
            t0_opt = max(min(opt_res[1], self.trb),self.tra)
            converge = True
            self.set_param(a_opt, t0_opt)
        
        if(False):
            from skopt import gp_minimize
            opt_res = gp_minimize(fun, bounds, n_calls=100, random_state=0)
            
            rms = opt_res.fun
            a_opt = opt_res.x[0]
            t0_opt = opt_res.x[1]
            converge = True
            self.set_param(a_opt, t0_opt)
        
        return {
            "rms":rms,
            "alpha":a_opt,
            "t0":t0_opt,
            "converge":converge,
        }

def meta_change_detect_np(X, t, clf, n_trial=6, n_seg=20, verbos=False):
    
    n = X.shape[0]
    n_train = int(n*0.5)
    n_valid = int(n*0.3)
    n_test = int(n*0.2)
    
    res = []
    for i_trial in range(n_trial):
        gc.collect()
        # if(True):
        #     print("trial %3d"%(i_trial))

        idx = np.arange(n)
        np.random.shuffle(idx)
        X_i = X[idx, :]; t_i = t[idx]
        X_train = X_i[:n_train]; t_train = t_i[:n_train]
        X_valid = X_i[n_train:n_train+n_valid]; t_valid = t_i[n_train:n_train+n_valid]
        X_test = X_i[n_train+n_valid:n]; t_test = t_i[n_train+n_valid:n]
        
        t_left = np.min(t_train)
        t_right = np.max(t_train)
        t_list = np.linspace(t_left, t_right, n_seg+2, endpoint=True)[1:-1] ## to avoid too few label, when using MLP
        
        res_trial = {"lbd":[], "acc":[], "maj":[], "accdev":[], "model_t0":None, "model_alpha":None, "model_curve":[]}
        
        ## for every t_i
        for t_i in t_list:
            y_train = 1*(t_train>t_i)
            y_valid = 1*(t_valid>t_i)
            y_test = 1*(t_test>t_i)

            clf.fit(X_train, y_train)

            y_hat_valid = clf.predict(X_valid)
            y_hat_test = clf.predict(X_test)
            
            acc_valid = accuracy_score(y_valid, y_hat_valid)
            acc_test = accuracy_score(y_test, y_hat_test)
            
            maj = np.max([np.mean(y_test), 1.-np.mean(y_test)])
            accdev = acc_test - maj
            # if(verbos): print("t_i = %.4f, maj = %.4f, acc = %.4f, accdev = % -.4f"%(t_i, maj, acc_test, accdev))
            
            res_trial["lbd"].append(t_i)
            res_trial["maj"].append(maj)

            res_trial["acc"].append(max(acc_test, maj))
            res_trial["accdev"].append(max(0, accdev))
        
        ## infer
        model_dist = Model_dist(t, 100)
        
        model_acc = Model_acc(model_dist)
        fit_res = model_acc.fit(res_trial)
        # print("t0 = %.4f, alpha = %.4f"%(fit_res["t0"], fit_res["alpha"]))
        
        ## add infer results
        n_seg_model = 100
        model_t_list = np.linspace(t_left, t_right, n_seg_model, endpoint=True)
        model_curve = model_acc.query_acc_list(model_t_list)["accdev"]
        
        res_trial["model_t0"] = fit_res["t0"]
        res_trial["model_alpha"] = fit_res["alpha"]
        res_trial["model_curve"] = [model_t_list, model_curve]
        
        ## add to res
        res.append(res_trial)
    
    model_t0_all = np.array([res[i]["model_t0"] for i in range(n_trial)])
    model_alpha_all = np.array([res[i]["model_alpha"] for i in range(n_trial)])
    
    # print("t0 = %.4f +- %.4f"%(np.mean(model_t0_all), np.std(model_t0_all)))
    # print("alpha = %.4f +- %.4f"%(np.mean(model_alpha_all), np.std(model_alpha_all)))
    
    return res


def get_avg_res(res):
    alpha_list = []
    t0_list = []
    for i in range(len(res)):
        t0_list.append(res[i]["model_t0"])
        alpha_list.append(res[i]["model_alpha"])
    t0_list = np.array(t0_list)
    alpha_list = np.array(alpha_list)
    return {
        "t0_mean":np.mean(t0_list),
        "t0_std":np.std(t0_list),
        "alpha_mean":np.mean(alpha_list),
        "alpha_std":np.std(alpha_list)
    }


def change_point_tree(X, t, model, min_range=0.1, max_d = 3, alpha_cut=1e-1, n_seg=10, n_trial=3):
    ## initialize
    que = []
    root = {"depth":0, "left":None, "right":None}
    node0 = {"t_left":np.min(t), "t_right":np.max(t)}
    root["data"] = node0
    que.append(root)
    change_point_result = []
    
    while(len(que)>0):
        tree_i = que.pop(0)
        node_i = tree_i["data"]
        
        # print(node_i)
        d_i = tree_i["depth"]
        t_range = node_i["t_right"] - node_i["t_left"]
        if(d_i>max_d or t_range<min_range): continue
        
        ## find new splitting point
        idx = (t>node_i["t_left"])&(t<=node_i["t_right"])
        res = meta_change_detect_np(X[idx], t[idx], model, n_trial=n_trial, n_seg=n_seg, verbos=False)
        
        res_avg = get_avg_res(res)
        t0 = res_avg["t0_mean"]

        if(t0-node_i["t_left"]<min_range or node_i["t_right"]-t0<min_range): continue
        if(res_avg["alpha_mean"]<alpha_cut): continue

        change_point_result.append(res_avg)

        ## record t0 in que node
        ratio =  np.sum(t[idx]<=t0)/t[idx].shape[0]
        node_i["t0"] = t0
        node_i["ratio"] = ratio
        node_i["alpha"] = res_avg["alpha_mean"]
        node_i["res"] = res
        
        # print("ratio = ", ratio)
        
        ## create two tree nodes
        tree_j0 = {"depth":d_i+1, "left":None, "right":None}
        tree_j1 = {"depth":d_i+1, "left":None, "right":None}
        
        ## creates two que nodes, link
        node_j0 = {"t_left":node_i["t_left"], "t_right":t0}
        node_j1 = {"t_left":t0, "t_right":node_i["t_right"]}
        
        ## link que node to tree nodes
        tree_j0["data"] = node_j0;
        tree_j1["data"] = node_j1;
        
        tree_i["left"] = tree_j0
        tree_i["right"] = tree_j1
        
        que.append(tree_j0)
        que.append(tree_j1)
        
    return root, change_point_result
