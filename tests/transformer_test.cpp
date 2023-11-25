//
// Created by lht on 23-11-16.
//

#include <iostream>

#include "transformer.hpp"


int main()
{
    const int src_size = 10;
    const int tgt_size = 10;
    const int seq_length = 32;

    ml_transformer::Transformer model(src_size, tgt_size, 10, 8, 5, 3);

    const arma::mat src (src_size, seq_length, arma::fill::randn);
    const arma::mat tgt (tgt_size, seq_length, arma::fill::randn);

    arma::mat output;
    model.Forward(src, tgt, output);

    output.brief_print("Test output:");
}
