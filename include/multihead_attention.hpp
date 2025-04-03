#pragma once

#include <armadillo>
#include <cmath>
#include <string>

namespace ml_transformer {
/*!
 * Implementation of the Softmax class.
 * @ref
 * https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/ann/layer/softmax_impl.hpp
 */
class Softmax {
public:
  template <typename MatType>
  static void Forward(const MatType& input, MatType& output) {
    MatType softmaxInput = exp(input.each_row() - max(input, 0));
    output = softmaxInput.each_row() / sum(softmaxInput, 0);
  }

  template <typename MatType>
  static void Backward(const MatType& /* input */, const MatType& output,
                       const MatType& gy, MatType& g) {
    g = output % (gy - repmat(sum(gy % output), output.n_rows, 1));
  }
};

/*!
 * @brief Multi-head Attention Layer.
 * @tparam MatType Matrix type (arma::mat by default)
 */
template <typename MatType = arma::mat> class MultiHeadAttention {
public:
  /*!
   * @brief Specific constructor.
   * @param d_model Model's dimension.
   * @param num_heads Number of attention heads.
   */
  explicit MultiHeadAttention(const size_t d_model, const size_t num_heads = 3)
      : modelDim(d_model), numHeads(num_heads),
        queryDim(static_cast<size_t>(d_model / num_heads)),
        queryWeight(d_model, d_model),
        keyDim(static_cast<size_t>(d_model / num_heads)),
        keyWeight(d_model, d_model),
        valueDim(static_cast<size_t>(d_model / num_heads)),
        valueWeight(d_model, d_model), outWeight(d_model, d_model), attnProbs(),
        auxBackwardV(), attnOutput(), inSoftmaxGrad() {
    // Initialize
  }

  /*!
   * @brief Forward process of multi-head attention.
   * @param Q Input of query vector.
   * @param K Input of key vector.
   * @param V Input of value vector.
   * @param Mask Input of mask matrix.
   * @param output Output of multi-head attention.
   */
  void Forward(const MatType& Q, const MatType& K, const MatType& V,
               const MatType& Mask, MatType& output) const {
    // Reset auxiliary parameters
    ResetAuxiliary();

    // Begin forward
    const MatType weightQ = queryWeight * Q;

    const MatType weightK = keyWeight * K;

    // Calculate attention scores.
    const MatType attnScores = weightQ * weightK.t() / std::sqrt(queryDim);

    // Apply mask.
    const MatType attnScoresM = attnScores % Mask;

    // Apply softmax to obtain attention probabilities.
    Softmax::Forward<MatType>(attnScoresM, attnProbs);

    // Multiply by values to obtain the final output.
    auxBackwardV = attnProbs * valueWeight;
    attnOutput = auxBackwardV * V;

    // Apply output weight
    output = outWeight * attnOutput;
  }

  /*!
   * @brief Backward process of multi-head attention.
   * @param Q Input of query vector.
   * @param K Input of key vector.
   * @param V Input of value vector.
   * @param Mask Input of mask matrix.
   * @param output Output of multi-head attention (result from `Forward`).
   * @param gy The back-propagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& Q, const MatType& K, const MatType& V,
                const MatType& Mask, const MatType& output, const MatType& gy,
                MatType& g) {
    PrepareAuxiliary(Q, K, V, Mask, gy);

    // Backward `Q`
    const MatType qGrad = queryWeight.t() * (inSoftmaxGrad % Mask) *
                          (keyWeight * K / std::sqrt(queryDim));

    std::cout << "Q Grad" << qGrad << std::endl;

    // Backward `K`
    const MatType kGrad =
        ((queryWeight * Q).t() * (inSoftmaxGrad % Mask) * keyWeight).t();

    std::cout << "K grad" << kGrad << std::endl;

    // Backward `V`
    MatType g_v = (outWeight * auxBackwardV).t() * gy;

    std::cout << "V grad" << g_v << std::endl;
  }

  /*!
   * @brief Calculate the gradient of multi-head attention.
   * @param Q Input of query vector.
   * @param K Input of key vector.
   * @param V Input of value vector.
   * @param Mask Input of mask matrix.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& Q, const MatType& K, const MatType& V,
                const MatType& Mask, const MatType& error, MatType& gradient) {
    PrepareAuxiliary(Q, K, V, Mask, error);

    // Gradient `queryWeight`
    const MatType qw_grad = inSoftmaxGrad % Mask * keyWeight * K * Q.t();
    std::cout << "queryWeight grad" << qw_grad << std::endl;

    // Gradient `keyWeight`
    const MatType kw_grad =
        (K * (queryWeight * Q).t() * (inSoftmaxGrad % Mask)).t();
    std::cout << "keyWeight grad" << kw_grad << std::endl;

    // Gradient `valueWeight`
    MatType vw_grad = (outWeight * attnProbs).t() * error * V.t();
    std::cout << "valueWeight grad" << vw_grad << std::endl;

    // Gradient `outWeight`
    MatType ow_grad = error * attnOutput.t();
    std::cout << "outWeight grad" << ow_grad << std::endl;
  }

private:
  //! \brief Model's dimension.
  size_t modelDim;

  //! \brief Number of attention heads.
  size_t numHeads;

  //! \brief Query dimension.
  size_t queryDim;

  //! \brief Query weight.
  MatType queryWeight;

  //! \brief Key dimension.
  size_t keyDim;

  //! \brief Key weight.
  MatType keyWeight;

  //! \brief Value Dimension.
  size_t valueDim;

  //! \brief Value weight.
  MatType valueWeight;

  //! \brief Output weight.
  MatType outWeight;

  //! \brief Output of softmax
  MatType attnProbs;

  //! \brief Auxiliary parameters for backward of value
  MatType auxBackwardV;

  //! \brief Output of scaled dot-product attention
  MatType attnOutput;

  //! \brief Gradient in softmax
  MatType inSoftmaxGrad;

  /*!
   * @brief Reset all auxiliary parameters.
   */
  void ResetAuxiliary() {
    attnProbs.reset();
    auxBackwardV.reset();
    attnOutput.reset();
    inSoftmaxGrad.reset();
  }

  /*!
   * @brief Prepare all auxiliary parameters.
   * @param Q Input of query vector.
   * @param K Input of key vector.
   * @param V Input of value vector.
   * @param Mask Input of mask matrix.
   * @param error The calculated error.
   */
  void PrepareAuxiliary(const MatType& Q, const MatType& K, const MatType& V,
                        const MatType& Mask, const MatType& error) {
    // Auxiliary parameters constructed from `Forward`
    if (attnProbs.n_elem == 0 || auxBackwardV.n_elem == 0 ||
        attnOutput.n_elem == 0) {
      MatType tmpOut;
      Forward(Q, K, V, Mask, tmpOut);
    }

    // Auxiliary parameters in `Softmax`
    if (inSoftmaxGrad.n_elem == 0) {
      const MatType softmaxGrad = outWeight.t() * error * (valueWeight * V).t();
      Softmax::Backward({}, attnProbs, softmaxGrad, inSoftmaxGrad);
    }
  }
};
} // namespace ml_transformer
