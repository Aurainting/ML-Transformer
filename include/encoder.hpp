#pragma once

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/linear.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/regularizer/no_regularizer.hpp>

#include "multihead_attention.hpp"
#include "normalize.hpp"

namespace ml_transformer {
/*!
 * @brief Encoder of Transformer.
 * @tparam MatType
 */
template <typename MatType = arma::mat, typename RegularizerType = mlpack::NoRegularizer>
class Encoder : public mlpack::Layer<MatType> {
 public:
  /*!
   * @brief Default constructor.
   */
  Encoder() = default;

  /*!
   * @brief Specific constructor.
   * @param d_model
   * @param num_heads
   * @param d_hid
   * @param drop_ratio
   */
  Encoder(const size_t d_model, const size_t num_heads, const size_t d_hid,
          const double drop_ratio = 0.4)
      : modelDim(d_model), numHeads(num_heads), hiddenDim(d_hid),
        dropLayer(drop_ratio), normLayer(), attnLayer(d_model, num_heads),
        linearLayer1(), reluLayer(), linearLayer2() {
    // Nothing to do here.
  }

  ~Encoder() = default;

  //! Clone the LinearType object. This handles polymorphism correctly.
  Encoder* Clone() const { return new Encoder(*this); }

  /*!
   * @brief Main process of Encoder.
   * @param input
   * @param output
   */
  void Forward(const MatType &input, MatType &output) {
    MatType attnOutput{};
    attnLayer.Forward(input, input, input, arma::ones(modelDim, modelDim),
                      attnOutput);

    dropLayer.Forward(attnOutput, attnOutput);

    MatType x{};
    normLayer.Forward(input + attnOutput, x);

    MatType forwardOutput{};
    linearLayer1.Forward(x, forwardOutput);
    reluLayer.Forward(forwardOutput, forwardOutput);
    linearLayer2.Forward(forwardOutput, forwardOutput);

    dropLayer.Forward(forwardOutput, forwardOutput);

    normLayer.Forward(x + forwardOutput, output);
  }

  void Backward(const MatType& input,
                const MatType& output,
                const MatType& gy,
                MatType& g)
  { /* Nothing to do here */ }

  void Gradient(const MatType& input,
                const MatType& error,
                MatType& gradient)
  { /* Nothing to do here */ }

  void SetWeights(typename MatType::elem_type* weightsPtr) {

  }

  [[nodiscard]] size_t WeightSize() const { return 0; }

 private:
  //! \brief
  size_t modelDim{};

  //! \brief
  size_t numHeads{};

  //! \brief
  size_t hiddenDim{};

  //! \brief
  mlpack::DropoutType<MatType> dropLayer;

  //! \brief
  ml_transformer::Normalize<MatType> normLayer;

  //! \brief
  ml_transformer::MultiHeadAttention<MatType> attnLayer;

  //! \brief
  mlpack::LinearType<MatType, RegularizerType> linearLayer1;

  //! \brief
  mlpack::ReLU reluLayer;

  //! \brief
  mlpack::LinearType<MatType, RegularizerType> linearLayer2;
};

} // namespace ml_transformer
