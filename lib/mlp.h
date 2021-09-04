//
//  mlp.h
//  mln
//
//  Created by lewardo on 05/09/2019.
//  Copyright (c) 2019 lewardo. All rights reserved.
//

#pragma once

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

#include "neuron.h"
#include "layer.h"

enum mlp_err_t {
    MLP_SUCCESS,
    MLP_FAIL_UNINITIALISED,
    MLP_FAIL_LOAD_FILE,
};

using data_t = std::vector<std::pair<std::vector<float>, std::vector<float>>>;

class mlp {
public:
    bool initialised;

    mlp();                                                                          // declare nn w/o initialisation
    mlp(const std::vector<int>& npl);                                                      // declare and initialise nn
    
    void init(const std::vector<int>& npl);                                                // initialise nn after declaration
    
    int getval(int n);                                                              // return number of layers if n < 0 else return num neurons in layer n
    
    float train(const std::vector<float>& input, const std::vector<float>& target, float lr);     // train the nn once
    float train(const std::pair<const std::vector<float>&, const std::vector<float>&>& data, float lr) { // train the nn once
        return train(data.first, data.second, lr);
    };
    
    mlp_err_t predict(const std::vector<float>& input, std::vector<float>& output);         // predict output from given input
    mlp_err_t predict(const std::vector<float>& input);                             // predict output from given input

    mlp_err_t save(std::string path);                                               // save to a .txt or .bin file
    mlp_err_t load(std::string path);                                               // load from .txt or .bin file
    
private:
    int nl = -1;
    std::vector<int> npl;
    std::vector<Layer> layers;
    
    static inline float err(float tg, float val) {
        return 0.5f * (tg - val) * (tg - val);
    }
    
    static float tot_err(std::vector<float> tg, std::vector<Neuron> output) {
        float err = 0.0;
        
        for(int n = 0; n < tg.size(); n++)
            err += mlp::err(tg[n], output[n].val);
        
        return err;
    }
    
    
};
