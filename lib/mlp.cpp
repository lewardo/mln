//
//  mlp.cpp
//  mln
//
//  Created by lewardo on 17/04/2020.
//  Copyright (c) 2020 lewardo. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

#include "mlp.h"

mlp::mlp() {
    initialised = false;
}

mlp::mlp(const std::vector<int>& arg) {
    nl = (int) arg.size();
    npl = arg;
    
    for(int n : npl)
        layers.push_back(Layer(n));
    
    for(int n = 0; n < nl; n++)
        layers[n].refLayer(n == nl - 1 ? nullptr : &layers[n + 1]);
    
    initialised = true;
};

void mlp::init(const std::vector<int>& arg) {
    if(initialised) return;
    
    nl = (int) arg.size();
    npl = arg;
    
    for(int n : npl)
        layers.push_back(Layer(n));
    
    for(int n = 0; n < nl; n++)
        layers[n].refLayer(n == nl - 1 ? nullptr : &layers[n + 1]);
    
    initialised = true;
};

int mlp::getval(int n) {
    if(initialised == false) return 0;
    if(n < 0) return nl;
    return npl[n];
}

mlp_err_t mlp::predict(const std::vector<float>& input, std::vector<float>& output) {
    output.clear();
    layers[0].assign(input);
    
    if(!initialised) return MLP_FAIL_UNINITIALISED;
    
    for(int n = 0; n < nl - 1; n++)
        layers[n].propagate();
    
    for(int n = 0; n < layers[nl - 1].nn; n++)
        output.push_back((float) layers[nl - 1].neurons[n].val);
    
    return MLP_SUCCESS;
};

mlp_err_t mlp::predict(const std::vector<float>& input) {
    layers[0].assign(input);
    
    if(!initialised) return MLP_FAIL_UNINITIALISED;
    
    for(int n = 0; n < nl - 1; n++)
        layers[n].propagate();
    
    return MLP_SUCCESS;
};

float mlp::train(const std::vector<float>& input, const std::vector<float>& target, float lr) {
    predict(input);
    
    if(!initialised) return -1;
    
    for(int n = 0; n < npl[nl - 1]; n++)
        layers[nl - 1].neurons[n].err = (layers[nl - 1].neurons[n].val - target[n]) * Layer::d_sigmoid(layers[nl - 1].neurons[n].val);
    
    for(int n = nl - 2; n >= 0; n--)
        layers[n].backtrack(lr);
    
    return tot_err(target, layers[nl - 1].neurons);
};

mlp_err_t mlp::save(std::string path) {
    FILE * srcFile;
    
    if(!initialised) return MLP_FAIL_UNINITIALISED;
    
    if((srcFile = fopen(path.c_str(), "w")) == NULL) return MLP_FAIL_LOAD_FILE;
    
    fprintf(srcFile, "%d\n", nl);
    
    for(int n : npl)
        fprintf(srcFile, "%d\t", n);
    
    fprintf(srcFile, "\n");
    
    for(int layer = 0; layer < nl - 1; layer++) {
        for(int nb = 0; nb < npl[layer + 1]; nb++) {
            for(int na = 0; na < npl[layer]; na++)
                fprintf(srcFile, "%.16f\t", layers[layer].neurons[na].w[nb]);
            fprintf(srcFile, "%.16f\n", layers[layer + 1].neurons[nb].b);
        };
    };
    
    fclose(srcFile);
    
    return MLP_SUCCESS;
};

mlp_err_t mlp::load(std::string path) {
    FILE * srcFile;
    int in_nl, val;
    std::vector<int> in_npl;
    
    if((srcFile = fopen(path.c_str(), "r")) == NULL) return MLP_FAIL_LOAD_FILE;
    
    fscanf(srcFile, "%d", &in_nl);
    
    for(int n = 0; n < in_nl; n++) {
        fscanf(srcFile, "%d", &val);
        in_npl.push_back(val);
    }
    
    if(initialised) return MLP_FAIL_UNINITIALISED;
    else init(in_npl);
    
    for(int layer = 0; layer < nl - 1; layer++) {
        for(int nb = 0; nb < npl[layer + 1]; nb++) {
            for(int na = 0; na < npl[layer]; na++)
                fscanf(srcFile, "%f", &layers[layer].neurons[na].w[nb]);
            fscanf(srcFile, "%f", &layers[layer + 1].neurons[nb].b);
        };
    };
    
    fclose(srcFile);
    
    return MLP_SUCCESS;
};


