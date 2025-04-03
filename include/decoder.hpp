#pragma once

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/linear.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/regularizer/no_regularizer.hpp>
#include <mlpack/methods/ann/layer/batch_norm.hpp>

#include "multihead_attention.hpp"
#include "normalize.hpp"

namespace ml_transformer {
/*!
 * @brief Decoder of Transformer.
 * @tparam MatType
 */
template <typename MatType = arma::mat, typename RegularizerType = mlpack::NoRegularizer>
class Decoder {
 public:
  /*!
   * @brief Default constructor.
   */
  Decoder() = default;

  /*!
   * @brief Specific constructor.
   * @param d_model
   * @param num_heads
   * @param d_hid
   * @param drop_ratio
   */
  Decoder(const size_t d_model, const size_t num_heads, const size_t d_hid,
          const double drop_ratio = 0.4)
      : modelDim(d_model), numHeads(num_heads), hiddenDim(d_hid),
        dropRatio(drop_ratio), attnLayer(d_model, num_heads),
        crossAttnLayer(d_model, num_heads), fc1(), reluLayer(), fc2() {
    // Nothing to do here.
  }

  /*!
   * @brief Main process of Decoder.
   * @param input
   * @param enOutput
   * @param mask1
   * @param mask2
   * @param output
   */
  void Forward(const MatType &input, const MatType &enOutput,
               const MatType &mask1, const MatType &mask2, MatType &output) {
    mlpack::Dropout dropLayer(dropRatio);
    Normalize normLayer(1e-5);

    MatType attnOutput{};
    attnLayer.Forward(input, input, input, mask1, attnOutput);

    dropLayer.Forward(attnOutput, attnOutput);

    MatType x{};
    normLayer.Forward(input + attnOutput, x);

    crossAttnLayer.Forward(enOutput, enOutput, x, mask2, attnOutput);

    dropLayer.Forward(attnOutput, attnOutput);

    normLayer.Forward(x + attnOutput, x);

    MatType forwardOutput{};
    fc1.Forward(x, forwardOutput);
    reluLayer.Forward(forwardOutput, forwardOutput);
    fc2.Forward(forwardOutput, forwardOutput);

    dropLayer.Forward(forwardOutput, forwardOutput);

    normLayer.Forward(x + forwardOutput, output);
  }

 private:
  //! \brief
  size_t modelDim{};

  //! \brief
  size_t numHeads{};

  //! \brief
  size_t hiddenDim{};

  //! \brief
  double dropRatio{};

  //! \brief
  MultiHeadAttention<MatType> attnLayer;

  //! \brief
  MultiHeadAttention<MatType> crossAttnLayer;

  //! \brief
  mlpack::LinearType<MatType, RegularizerType> fc1;
  mlpack::ReLU reluLayer;
  mlpack::LinearType<MatType, RegularizerType> fc2;
};

} // namespace ml_transformer
