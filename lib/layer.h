//
//  layer.h
//  mln
//
//  Created by lewardo on 17/11/2019.
//  Copyright (c) 2019 lewardo. All rights reserved.
//

#pragma once

class Layer {
    int nn;
    std::vector<Neuron> neurons;
    
    Layer *next; // previous layer
    
public:
    Layer(int nn) {
        nn = nn;
        
        for(int n = 0; n < nn; n++)
            neurons.push_back( Neuron() ); // initialise the neurons
        
    }
    
    Layer& operator=(const Layer&) = delete;
    
    void refLayer(Layer* _next) {
        next = _next;
    
        for(int n = 0; n < nn; n++)
            neurons[n].init(next == nullptr ? 0 : (int) next -> nn); // if layer behind it, initialise weights/neurons, otherwise just neurons (0 weights)
    }
    
    static inline float sigmoid(float x) {
        return 1.0f / (1.0f + expf(-x)); // activation function
    }
    
    static inline float d_sigmoid(float x) {
        return x * (1.0f - x); // derivative of sigmoid in respect to the output of the neuron
    }
    
    void assign(const std::vector<float> &list) {
        for (int n = 0; n < list.size(); n++)
            neurons[n].val = list[n];
    }
    
    void propagate() {
        
        for(int n = 0; n < next -> nn; n++) {
            float sum = 0.0f; // sum of weighted values
            
            for(int t = 0; t < nn; t++)
                sum += neurons[t].val * neurons[t].w[n]; // previous neuron value * connecting weight
            
            next -> neurons[n].val = sigmoid(sum + next -> neurons[n].b); // set next value with bias and sigmoid function
            // TODO: actfunc namespace for multiple functions.
        }
    }
    
    void backtrack(float lr) {
        for(int t = 0; t < nn; t++) { // every neuron in prev layer
            float error = 0.0f,
                  dn,               //dE/dz[Lk]
                  dw;               //dE/dw[Ljk]
            
            for(int n = 0; n < next -> nn; n++) { // loop throught all neurons connected to previous neuron
                dn = next -> neurons[n].err; // dE/dz[Lk]

                dw = dn * neurons[t].val; // dw[Ljk]/dE
                
                error += dn * neurons[t].w[n]; // prev neuron error
                
                neurons[t].w[n] -= dw * lr; // subrtact error prop. to learning rate
                next -> neurons[n].b -= dn * lr; // same but w/o the prev neuron factor
            }
            
            neurons[t].err = error * d_sigmoid(neurons[t].val); // set prev neuron error
        }
    }
    
    friend class mlp;
};

