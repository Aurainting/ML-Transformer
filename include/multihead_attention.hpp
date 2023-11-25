/*!
 * @author Aurumting
 * @date 23-11-25
 */
#ifndef ML_TRANSFORMER_MULTIHEAD_ATTENTION_HPP
#define ML_TRANSFORMER_MULTIHEAD_ATTENTION_HPP

#include <armadillo>

#include <mlpack/methods/ann/layer/softmax.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>


namespace ml_transformer {
    /*!
     * @brief Multi-head Attention Layer.
     * @tparam MatType Matrix type (arma::mat by default)
     */
    template<typename MatType = arma::mat>
    class MultiHeadAttention {
    public:
        /*!
         * @brief Default constructor.
         */
        MultiHeadAttention() = default;

        /*!
         * @brief Specific constructor.
         * @param d_model Model's dimension.
         * @param num_heads Number of attention heads.
         */
        explicit MultiHeadAttention(const size_t d_model,
                                    const size_t num_heads = 3)
                : modelDim(d_model)
                , numHeads(num_heads)
                , queryDim(d_model / num_heads)
                , queryWeight(d_model, d_model)
                , keyDim(d_model / num_heads)
                , keyWeight(d_model, d_model)
                , valueDim(d_model / num_heads)
                , valueWeight(d_model, d_model)
                , outWeight(d_model, d_model)
        {
            // Init.
            mlpack::XavierInitialization glorotInit;

            glorotInit.Initialize(queryWeight, queryWeight.n_rows, queryWeight.n_cols);
            glorotInit.Initialize(keyWeight, keyWeight.n_rows, keyWeight.n_cols);
            glorotInit.Initialize(valueWeight, valueWeight.n_rows, valueWeight.n_cols);
            glorotInit.Initialize(outWeight, outWeight.n_rows, outWeight.n_cols);
        }

        /*!
         * @brief Main process of Multi-head Attention.
         * @param Q
         * @param K
         * @param V
         * @param Mask
         * @param output
         */
        void Forward(const MatType& Q,
                     const MatType& K,
                     const MatType& V,
                     const MatType& Mask,
                     MatType& output) {
            const MatType weightQ = queryWeight * Q;

            const MatType weightK = keyWeight * K;

            const MatType weightV = valueWeight * V;

            MatType attnOutput {};
            ScaledDotProductAttention(weightQ, weightK, weightV, Mask, attnOutput);

            output = outWeight * attnOutput;
        }

    private:
        //! \brief Model's dimension.
        size_t modelDim {};

        //! \brief Number of attention heads.
        size_t numHeads {};

        //! \brief Query dimension.
        size_t queryDim {};

        //! \brief Query weight.
        MatType queryWeight;

        //! \brief Key dimension.
        size_t keyDim {};

        //! \brief Key weight.
        MatType keyWeight;

        //! \brief Value Dimension.
        size_t valueDim {};

        //! \brief Value weight.
        MatType valueWeight;

        //! \brief Output weight.
        MatType outWeight;

        /*!
         * @brief Perform scaled dot-product attention.
         * @param Q
         * @param K
         * @param V
         * @param Mask
         * @param output
         */
        void ScaledDotProductAttention(const MatType& Q,
                                       const MatType& K,
                                       const MatType& V,
                                       const MatType& Mask,
                                       MatType& output) {
            // Calculate attention scores.
            const MatType attnScores = Q * K.t() / std::sqrt(queryDim);

            // Apply mask.
            const MatType attnScoresM = attnScores % Mask;

            // Apply softmax to obtain attention probabilities.
            mlpack::Softmax softmaxLayer;
            MatType attnProbs {};
            softmaxLayer.Forward(attnScoresM, attnProbs);

            // Multiply by values to obtain the final output.
            output = attnProbs * V;
        }
    };
}

#endif //ML_TRANSFORMER_MULTIHEAD_ATTENTION_HPP
