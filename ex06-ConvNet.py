from Network import *
import data

import math
import numpy
import vis


mnist = data.load("mnist")
test_data = mnist["test"].values()
train_data = mnist["train"]

net = Network(
    layers = [InputLayer([28,28]),
              ConvolutionLayer([6,6], 8, activation = ReLU),
              PoolLayer([2,2]),
              FullyConnectedLayer(100, activation = ReLU),
              SoftMaxLayer(10)],
    cost=log_likelihood)

for m in range(100):
    net.train(*train_data.get_batch(30), learning_rate = 0.01 )

xs, ys = train_data.get_batch(5)
info = net.introspect(xs, ys)
info.keys()


for i in range(20):
    for m in range(1000):
        net.train(*train_data.get_batch(30), learning_rate = 0.01 )

    print "Training Step", i
    print "Test Accuracy:", net.accuracy(*test_data)
    print "Test Cost:", net.test(*test_data)
    print ""

    print "Network Visualization and Introspection:"
    #print ""
    #print "Sample Inputs:"
    sample_xs, sample_ys = train_data.get_batch(30)
    #vis.vis_matrix(sample_xs[:4], s2x2 = True)

    full_info = net.introspect(sample_xs, sample_ys)

    for layer_name in sorted(full_info.keys()):
        print ""
        info = full_info[layer_name]
        if "ConvolutionLayer" in layer_name:
            print layer_name
            #print "Sample Activations (should be non-constant, non-blank):"
            #F = info["y"].shape[1]
            ## Actually, let's just cycle the one we show.
            ## They're pretty messy.
            #for fn in [i%F]:
            #    print "Filter", fn, "Activations:"
            #    vis.vis_matrix(info["y"][:4,fn,:22,:22], s2x2 = True)
            #    print 80*"-"

            # A good diagnostic for conv nets is often whether they are learning
            # nice filters in their first layer. This is because it tells us
            # if any signal is succesfully getting back to the first layer,
            # which in deep models can be hard:
            print ""
            print "Filters (if signal is reaching them they should look 'nice'):"
            C = info["W"].shape[1]
            if C == 1:
                vis.vis_matrix_signed(info["W"][:,0])
            else:
                for c in range(C):
                    print 80*"-"
                    vis.vis_matrix_signed(info["W"][:,c])
            print "Filters Gradient (bad if this is blank):"
            if C == 1:
                vis.vis_matrix_signed(info["W_grad"][:,0])
            else:
                for c in range(C):
                    print 80*"-"
                    vis.vis_matrix_signed(info["W_grad"][:,c])
        elif "FullyConnectedLayer" in layer_name:
            print layer_name
            print ""
            print "Activations (vertical streaks are bad):"
            # A big of an ugly cludge to visualize each matrix entry with one char
            # Normaly visualize as 2 char or a half char, to avoid rectangular
            vis.vis_matrix(np.repeat(info["y"], 2, axis = 0)[:15, :200], s2x2 = True)
    

    print 80*"#"
    print ""

    

