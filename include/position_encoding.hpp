#pragma once

#include <armadillo>

namespace ml_transformer {
/*!
 * @brief Positional Encoding Network.
 * @tparam MatType
 */
template <typename MatType = arma::mat> class PositionalEncoding {
public:
  /*!
   * @brief Specific constructor.
   * @param d_model
   * @param max_seq_length
   */
  explicit PositionalEncoding(const size_t d_model,
                              const size_t max_seq_length = 1024)
      : modelDim(d_model % 2 == 0 ? d_model : d_model + 1),
        maxSeqLength(max_seq_length), pe(arma::zeros(max_seq_length, d_model)) {
    const arma::mat position =
        arma::linspace(0, maxSeqLength - 1, maxSeqLength);
    const arma::mat omega = arma::exp(arma::regspace(0, 2, modelDim - 1) *
                                      (-std::log(10'000.0) / modelDim));

    const auto evenCols = arma::regspace<arma::uvec>(0, 2, pe.n_cols - 1);
    const auto oddCols = arma::regspace<arma::uvec>(1, 2, pe.n_cols - 1);
    pe.cols(evenCols) = arma::sin(position * omega.t());
    pe.cols(oddCols) = arma::cos(position * omega.t());

    pe = pe.t();
  }

  /*!
   * @brief Main process of positional encoding.
   * @param input
   * @param output
   */
  void Forward(const MatType& input, MatType& output) const {
    output = input + pe.head_cols(input.n_cols);
  }

  void Backward(const MatType& input, const MatType& output,
                       const MatType& gy, MatType& g) const {
    g = gy;
  }

private:
  //! \brief
  size_t modelDim;

  //! \brief
  size_t maxSeqLength;

  //! \brief
  MatType pe;
};

} // namespace ml_transformer
