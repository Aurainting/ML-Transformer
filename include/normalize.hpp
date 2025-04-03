#pragma once

#include <armadillo>

namespace ml_transformer {
/*!
 * @brief Layer Normalization
 * @tparam MatType
 */
template <typename MatType = arma::mat> class Normalize {
public:
  /*!
   * @brief Specific constructor.
   * @param eps A value added to the denominator for numerical stability (1e-5
   * by default)
   */
  explicit Normalize(const double eps = 1e-5) : eps(eps) {}

  /*!
   * @brief Main process of Normalize.
   * @param input (C, N) where C is the number of features or channels, and N is
   * the batch size
   * @param output (C, N)
   */
  void Forward(const MatType& input, MatType& output) const {
    const auto mu = arma::mean(input);
    const auto var = arma::var(input, 1);

    output = (input.each_row() - mu).each_row() / arma::sqrt(var + eps);
  }

  /*!
   * @brief Backward of layer normalization
   * @param gy
   * @param g
   */
  void Backward(const MatType& /* input */, const MatType& /* output */,
                const MatType& gy, MatType& g) const {
    g = gy;
  }

private:
  //! \brief A value added to the denominator for numerical stability
  double eps;
};
} // namespace ml_transformer
