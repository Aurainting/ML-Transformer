#pragma once

#include <ensmallen.hpp>
#include <mlpack/core.hpp>

#include "decoder.hpp"
#include "encoder.hpp"
#include "position_encoding.hpp"

namespace ml_transformer {
/*!
 * @brief Transformer in mlpack.
 * @tparam OutputLayerType
 * @tparam InitializationRuleType
 * @tparam MatType
 */
template <typename OutputLayerType = mlpack::NegativeLogLikelihood,
          typename InitializationRuleType = mlpack::RandomInitialization,
          typename MatType = arma::mat>
class Transformer {
public:
  /*!
   * @brief Default constructor.
   */
  Transformer() = default;

  Transformer(const size_t src_size, const size_t tgt_size,
              const size_t out_size, const size_t d_model, const size_t d_hid,
              const size_t num_layers = 3, const size_t num_heads = 3,
              const size_t max_seq_length = 1024, const double drop_ratio = 0.4)
      : srcSize(src_size), srcEmbed(d_model, src_size), tgtSize(tgt_size),
        tgtEmbed(d_model, tgt_size), outSize(out_size), modelDim(d_model),
        numLayers(num_layers), dropRatio(drop_ratio),
        posEncoder(d_model, max_seq_length), encoders(), decoders(),
        fc(out_size, d_model) {
    // Construct encoder_layers and decoder_layers.
    for (size_t i = 0; i != numLayers; ++i) {
      encoders.emplace_back(d_model, num_heads, d_hid, drop_ratio);
      decoders.emplace_back(d_model, num_heads, d_hid, drop_ratio);
    }

    // Init.
    mlpack::XavierInitialization glorotInit;
    glorotInit.Initialize(srcEmbed, srcEmbed.n_rows, srcEmbed.n_cols);
    glorotInit.Initialize(tgtEmbed, tgtEmbed.n_rows, tgtEmbed.n_cols);
    glorotInit.Initialize(fc, fc.n_rows, fc.n_cols);
  }

  /*!
   * @brief Copy the other Transformer layer (but not weights).
   * @param other
   */
  Transformer(const Transformer &other) = default;

  /*!
   * @brief Take ownership of the members of the other Transformer layer (but
   * not weights).
   * @param other
   */
  Transformer(Transformer &&other) = default;

   /*!
   * @brief
   * @param source
   * @param target
   * @param output
   */
  void Forward(const MatType &source, const MatType &target, MatType &output) {
    mlpack::Dropout dropLayer(dropRatio);

    MatType srcEmbedded = srcEmbed * source;
    posEncoder.Forward(srcEmbedded, srcEmbedded);
    dropLayer.Forward(srcEmbedded, srcEmbedded);

    MatType tgtEmbedded = tgtEmbed * target;
    posEncoder.Forward(tgtEmbedded, tgtEmbedded);
    dropLayer.Forward(tgtEmbedded, tgtEmbedded);

    MatType srcOutput{};
    encoders[0].Forward(srcEmbedded, srcOutput);
    for (size_t i = 1; i != numLayers; ++i) {
      encoders[i].Forward(srcOutput, srcOutput);
    }

    MatType tgtOutput{};
    decoders[0].Forward(tgtEmbedded, srcOutput, arma::ones(modelDim, modelDim),
                        arma::ones(modelDim, modelDim), tgtOutput);
    for (size_t i = 1; i != numLayers; ++i) {
      decoders[i].Forward(tgtOutput, srcOutput, arma::ones(modelDim, modelDim),
                          arma::ones(modelDim, modelDim), tgtOutput);
    }

    output = fc * tgtOutput;
  }

private:
  //! \brief
  size_t srcSize{};

  //! \brief
  MatType srcEmbed;

  //! \brief
  size_t tgtSize{};

  //! \brief
  MatType tgtEmbed;

  //! \brief
  size_t outSize{};

  //! \brief
  size_t modelDim{};

  //! \brief
  size_t numLayers{};

  //! \brief
  double dropRatio{};

  //! \brief
  PositionalEncoding<MatType> posEncoder;

  //! \brief
  std::vector<Encoder<MatType>> encoders;

  //! \brief
  std::vector<Decoder<MatType>> decoders;

  //! \brief
  MatType fc;
};
} // namespace ml_transformer
