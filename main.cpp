//
//  main.cpp
//  mln
//
//  Created by lewardo on 05/09/2019.
//  Copyright (c) 2019 lewardo. All rights reserved.
//

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "lib/mlp.h"

void flagFail(char c) {
    printf("did not recognise flag %c\n", c);
    
    printf("print help? [y/n]");
    char a = getc(stdin);
    
    if(a == 'y' || a == 'Y') {
        printf("In training mode (flag -t) use flags:\n");
        printf("\t-t      -> specify training mode\n");
        printf("\t-i path -> input file 'path' to train mlp on, see formats.txt for formats\n");
        printf("\t-o path -> output file 'path' where the weightd of the trained nn are stored\n");
        printf("\t-e num  -> 'num' amounts of epochs training the mlp\n\n");
        printf("In predicting mode (flag -p) use flags:\n");
        printf("\t-p      -> specify predicting mode\n");
        printf("\t-i path -> input file with data to predict values from\n");
        printf("\t-o path -> output file where predicted values are printed\n");
        printf("\t-n path -> mlp topology file to specify weights (training mode output file)\n");
    }
}


int main(int argc, const char * argv[]) {
    data_t data;
    mlp nn = mlp();
    
    std::vector<int> npl;
    int nl = 0, ni, num, epochs = -1;
    float val;
    char command, mode = '-';

    std::string srcFileStr, outFileStr, nnFileStr;
    FILE * srcFile = nullptr, * outFile = nullptr;
    
    for(int8_t n = 1; n < argc; n += 2) {                                       // skip file name (start at index 1) and skip flag arguments (+=2)
        const char * arg = argv[n + 1];
        
        sscanf(argv[n], "-%c", &command);
        
        switch(command) {
            case 't': {                                                        // training mode
                if(mode == 'p') return EXIT_FAILURE;
                mode = 't';
                n--;
                
                break;
            }
                
            case 'p': {                                                        // predicting mode
                if(mode == 't') return EXIT_FAILURE;
                mode = 'p';
                n--;
                
                break;
            }
                
            case 'n': {                                                         // nn topology input file
                nnFileStr = arg;
                break;
            }
                
            case 'i': {                                                         // input data file, see formats.txt for format
                srcFileStr = arg;
                break;
            }
                
            case 'o': {                                                         // output file
                outFileStr = arg;
                break;
            }
                
            case 'e': {                                                         // num epochs
                epochs = atoi(arg);
                break;
            }
                
            default: {
                flagFail(command);
                return EXIT_FAILURE;
                
                break;
            }
        }
    }
    
    switch(mode) {
        case 't': {
            float err = 0;
            
            srcFile = fopen(srcFileStr.c_str(), "r");
            
            if(!outFileStr.empty()) outFile = fopen(outFileStr.c_str(), "w");
            else printf("warning: no output file given\n");
            
            if(!nnFileStr.empty()) printf("warning: no nn structure file needed (flag -n)\n");
            
            fscanf(srcFile, "%d%d", &ni, &nl);
            
//            printf("%d\t%d\n", ni, nl);
            
            for(int n = 0; n < nl; n++) {
                fscanf(srcFile, "%d", &num);
                npl.push_back(num);
            }
            
            for(int n = 0; n < ni; n++) {
                std::vector<float> in, tg;
                
                for(int m = 0; m < npl.front(); m++) {
                    fscanf(srcFile, "%f", &val);
                    in.push_back(val);
                }
                
                for(int m = 0; m < npl.back(); m++) {
                    fscanf(srcFile, "%f", &val);
                    tg.push_back(val);
                }
                
                data.push_back(std::make_pair(in, tg));
            }
            
            printf("input file %s, mode %c\n", srcFileStr.c_str(), mode);
            
            nn.init(npl);
            
            if(epochs < 0) {
                printf("add -e tag to specify number of epochs\n");
                
                return EXIT_FAILURE;
            }
            
            for (int n = 0; n < epochs; n++) {
                err = nn.train(data[n % data.size()], 1);
            }
            
            printf("nn trained %d times, training error: %f\n", epochs, err);
            
            if(!outFileStr.empty()) nn.save(outFileStr.c_str());
            
            break;
        }
            
        case 'p': {
            srcFile = fopen(srcFileStr.c_str(), "r");

            if(nnFileStr.empty()) {
                printf("warning: no nn structure file given (training output file)\n");
                return EXIT_FAILURE;
            }

            if(!outFileStr.empty()) outFile = fopen(outFileStr.c_str(), "w");
            else printf("warning: no output file given\n");

            fscanf(srcFile, "%d", &ni);
            
            nn.load(nnFileStr);
            
            printf("data input file %s, mode %c\n", srcFileStr.c_str(), mode);
            
            for(int n = 0; n < ni; n++) {
                std::vector<float> trdata;

                for(int v = 0; v < nn.getval(0); v++) {
                    fscanf(srcFile, "%f", &val);
                    trdata.push_back(val);
                }
                
                if(!outFileStr.empty()) {
                    std::vector<float> outdata;
                    nn.predict(trdata, outdata);

                    for(int m = 0; m < outdata.size(); m++) {
                        fprintf(outFile, "%f\t", outdata[m]);
                    }
                    
                    fprintf(outFile, "\n");
                }
                
            }
        
            printf("output file %s\n", outFileStr.c_str());

            break;
        }
            
        default: {
            printf("please add -t or -p flag to specify training or predicting mode\n");
            
            return EXIT_FAILURE;
            break;
        }
    }
    
    if(!srcFileStr.empty()) fclose(srcFile);
    if(!outFileStr.empty()) fclose(outFile);

    return 0;
}
