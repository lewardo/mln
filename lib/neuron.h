//
//  neuron.h
//  mln
//
//  Created by lewardo on 17/11/2019.
//  Copyright (c) 2019 lewardo. All rights reserved.
//

#ifndef neuron_h
#define neuron_h

struct Neuron {
    float val,
          err;        //dE/dz[Lk]
    
    float b;
    
    int nw;
    std::vector<float> w;
    
    void init(int n) {
        if(n) b = Neuron::random();
        
        for(int cw = 0; cw < n; cw++)
            w.push_back((float) Neuron::random());
        
        val = err = 0.0f;
    }
    
    static inline float random() {
        return ((float) rand() / (float) RAND_MAX) - 0.5f;  // rand func between ±0.5
    }
};

#endif /* neuron_h */
