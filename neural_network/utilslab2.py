import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import pipeline
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D

def drawDecisionBoundary(model, df, **kwargs):
    n = 300
    x1 = np.linspace(df.x1.min(),df.x1.max(),n)
    x2 = np.linspace(df.x2.min(),df.x2.max(),n)
    mesh = np.meshgrid(x1, x2)
#    print(mesh[0].shape,mesh[1].shape)
#    print(mesh[0].reshape(-1,1).shape)
    mx = np.concatenate((mesh[0].reshape(-1,1),mesh[1].reshape(-1,1)),1)
#    print(mx.shape)
    prob=model.predict_proba(mx)[:,0].reshape(n,n)
#    print(prob.min(),prob.max())
    plt.contour(x1,x2,prob,levels=np.array([0.5]), **kwargs)



def predict_last_layer(model,df):
    if type(model) == sklearn.pipeline.Pipeline:
        assert(type(model._final_estimator)==sklearn.neural_network.MLPClassifier)
        m = sklearn.pipeline.Pipeline(model.steps[:-1])
        X = m.transform(df[["x1","x2"]])
        net = model._final_estimator
    elif type(model) == sklearn.neural_network.MLPClassifier:
        net = model
        X = df[["x1","x2"]]
    else:
        raise Exception("predict_last_layer: model type not implemented", type(model))
#    print(net.hidden_layer_sizes)
    layer_units = [X.shape[1]] + list(net.hidden_layer_sizes) + [net.n_outputs_]
    activations = [X]
    for i in range(net.n_layers_ - 1):
        activations.append(np.empty((X.shape[0],layer_units[i + 1])))
    xact = net._forward_pass(activations)
#    print([x.shape for x in xact])
    return xact[-2], net.coefs_, net.intercepts_


def draw3dhyperplane(ax, x,y, normal, d):
    n = 2
    def nprange(s):
        a,b = np.min(s), np.max(s)
        d = max((b-a)*10/100,0.04)
        return a-d,b+d
    if normal[2]!=0:
        minx, maxx = nprange(x)
        miny, maxy = nprange(y)
        x = np.linspace(minx, maxx, n)
        y = np.linspace(miny,maxy,n)
        xx,yy = np.meshgrid(x, y)
        z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
        ax.plot_surface(xx, yy, z, alpha=0.5)
    else:
        raise Exception("draw3dhyperplane: normal[2]==0")


def drawLine(a, b, c, x_range,y_range,**kwargs):# a x + b y + c = 0
    n = 1000
    x = np.linspace(x_range[0], x_range[1], n)
    if b == 0:
        if a == 0:
            raise Exception("no straight line to plot")
        else:
            x = np.repeat(-c/a, n)
            y = y_range
    else:
        y = (- a*x - c) / b
        x = [x for i,x in enumerate(x) if y_range[0]<=y[i]<=y_range[1]]
        y = [xi for xi in y if y_range[0]<=xi<=y_range[1]]
    plt.plot(x, y,**kwargs)

def drawNewFeaturesLine(model, df):
    p, coefs, intercepts = predict_last_layer(model,df)
    m = coefs[-2].shape[1]
    x_range = np.min(df.x1.values), np.max(df.x1.values)
    y_range = np.min(df.x2.values), np.max(df.x2.values)
    scale = [1, 1]
    mean = [0, 0]
    if type(model) == sklearn.pipeline.Pipeline:
        scale = model.steps[-2][1].scale_
        mean = model.steps[-2][1].mean_
    for i in range(m):
        a, b = coefs[-2][:,i]
        c = intercepts[-2][i]
        a = a / scale[0]
        b = b / scale[1]
        c = c - a * mean[0] - b * mean[1]
        drawLine(a,b,c-2,x_range,y_range,color="yellow")
        drawLine(a,b,c,x_range,y_range,color="black")
        drawLine(a,b,c+2,x_range,y_range,color="red")

def drawPhiSpace(model, df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p, coefs, intercepts = predict_last_layer(model,df)
    ax.scatter(p[:,0],p[:,1],p[:,2],c=df.c)
#    print(coefs[-1],coefs[-1].reshape(3))
    draw3dhyperplane(ax,p[:,0],p[:,1],coefs[-1].reshape(3),intercepts[-1])
    ax.set_xlabel("$\Phi_1(x_1,x_2)$")
    ax.set_ylabel("$\Phi_2(x_1,x_2)$")
    ax.set_zlabel("$\Phi_3(x_1,x_2)$")
    plt.show()


def fitneuralnet(s):
    neural_net = MLPClassifier(solver='lbfgs', alpha=1e-15)
    model = pipeline.make_pipeline(preprocessing.StandardScaler(), neural_net)
    model.fit(s[["x1","x2"]], s.c)
    return model

def fitlogistic(s):
#    model = pipeline.make_pipeline(preprocessing.StandardScaler(), LogisticRegression())
    model = LogisticRegression(C=1e6)
    model.fit(s[["x1","x2"]], s.c)
    return model


if __name__ == "__main__":
    df = pd.read_csv("lab2.csv")
    m=fitlogistic(df)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df.x1,df.x2,df.x2,c=df.c)
    plt.show()


