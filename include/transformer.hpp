/*!
 * @author Aurumting
 * @date 23-11-25
 */
#ifndef ML_TRANSFORMER_TRANSFORMER_HPP
#define ML_TRANSFORMER_TRANSFORMER_HPP

#include <mlpack/core.hpp>
#include "multihead_attention.hpp"

#include <vector>


namespace ml_transformer {
    /*!
     * @brief Transformer in mlpack.
     * @tparam MatType Matrix type (arma::mat by default).
     */
    template<typename MatType = arma::mat>
    class Transformer final : public mlpack::Layer<MatType> {
    public:
        /*!
         * @brief Default constructor.
         */
        Transformer() = default;

        Transformer(const size_t src_size,
                    const size_t tgt_size,
                    const size_t out_size,
                    const size_t d_model,
                    const size_t d_hid,
                    const size_t num_layers = 3,
                    const size_t num_heads = 3,
                    const size_t max_seq_length = 1024,
                    const double drop_ratio = 0.4)
                : srcSize(src_size)
                , srcEmbed(d_model, src_size)
                , tgtSize(tgt_size)
                , tgtEmbed(d_model, tgt_size)
                , outSize(out_size)
                , modelDim(d_model)
                , numLayers(num_layers)
                , dropRatio(drop_ratio)
                , posEncoder(d_model, max_seq_length)
                , encoders()
                , decoders()
                , fc(out_size, d_model)
        {
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
        Transformer(const Transformer& other) = default;

        /*!
         * @brief Take ownership of the members of the other Transformer layer (but not weights).
         * @param other
         */
        Transformer(Transformer&& other) = default;

        /*!
         * @brief Clone the Transformer object. This handles polymorphism correctly.
         * @return Cloned object.
         */
        Transformer* Clone() const override
        { return new Transformer(*this); }

        /*!
         * @brief
         * @param source
         * @param target
         * @param output
         */
        void Forward(const MatType& source,
                     const MatType& target,
                     MatType& output) {
            mlpack::Dropout dropLayer(dropRatio);

            MatType srcEmbedded = srcEmbed * source;
            posEncoder.Forward(srcEmbedded, srcEmbedded);
            dropLayer.Forward(srcEmbedded, srcEmbedded);

            MatType tgtEmbedded = tgtEmbed * target;
            posEncoder.Forward(tgtEmbedded, tgtEmbedded);
            dropLayer.Forward(tgtEmbedded, tgtEmbedded);

            MatType srcOutput {};
            encoders[0].Forward(srcEmbedded, srcOutput);
            for (size_t i = 1; i != numLayers; ++i) {
                encoders[i].Forward(srcOutput, srcOutput);
            }

            MatType tgtOutput {};
            decoders[0].Forward(tgtEmbedded, srcOutput,
                                arma::ones(modelDim, modelDim),
                                arma::ones(modelDim, modelDim),
                                tgtOutput);
            for (size_t i = 1; i != numLayers; ++i) {
                decoders[i].Forward(tgtOutput, srcOutput,
                                    arma::ones(modelDim, modelDim),
                                    arma::ones(modelDim, modelDim),
                                    tgtOutput);
            }

            output = fc * tgtOutput;
        }

        //! \brief Inner class for Transformer.
    private:
        /*!
         * @brief My Normalization Network.
         */
        class Normalize {
        public:
            /*!
             * @brief Default constructor.
             */
            Normalize() = default;

            /*!
             * @brief Specific constructor.
             * @param eps
             */
            explicit Normalize(const double eps)
                    : eps(eps)
            {
                // Nothing to do here.
            }

            /*!
             * @brief Main process of Normalize.
             * @param input
             * @param output
             */
            void Forward(const MatType& input,
                         MatType& output) {
                const auto mu = arma::mean(input);
                const auto var = arma::var(input);

                output = (input.each_row() - mu).each_row() / arma::sqrt(var + eps);
            }

        private:
            //! \brief
            double eps {};
        };

        /*!
         * @brief Positional Encoding Network.
         */
        class PositionalEncoding {
        public:
            /*!
             * @brief Default constructor.
             */
            PositionalEncoding() = default;

            /*!
             * @brief Specific constructor.
             * @param d_model
             * @param max_seq_length
             */
            explicit PositionalEncoding(const size_t d_model,
                                        const size_t max_seq_length=1024)
                    : modelDim(d_model)
                    , maxSeqLength(max_seq_length)
                    , pe(arma::zeros(max_seq_length, d_model))
            {
                assert((void("Only even-numbered `d_model` is supported now."), d_model % 2 == 0));

                const arma::mat position = arma::linspace(0, maxSeqLength - 1, maxSeqLength);
                const arma::mat omega = arma::exp(arma::regspace(0, 2, modelDim - 1)
                                                  * (-std::log(10'000.0) / modelDim));

                const auto evenCols = arma::regspace<arma::uvec>(0, 2, pe.n_cols-1);
                const auto oddCols = arma::regspace<arma::uvec>(1, 2, pe.n_cols-1);
                pe.cols(evenCols) = arma::sin(position * omega.t());
                pe.cols(oddCols) = arma::cos(position * omega.t());

                pe = pe.t();
            }

            /*!
             * @brief Main process of positional encoding.
             * @param input
             * @param output
             */
            void Forward(const MatType& input, MatType& output) {
                output = input + pe.head_cols(input.n_cols);
            }

        private:
            //! \brief
            size_t modelDim {};

            //! \brief
            size_t maxSeqLength {};

            //! \brief
            MatType pe {};
        };

        /*!
         * @brief Position-wise Feed-forward Network.
         */
        class PositionWiseFeedForward {
        public:
            /*!
             * @brief Default constructor.
             */
            PositionWiseFeedForward() = default;

            /*!
             * @brief Specific constructor.
             * @param d_model
             * @param d_hid
             */
            explicit PositionWiseFeedForward(const size_t d_model,
                                             const size_t d_hid)
                    : modelDim(d_model)
                    , hiddenDim(d_hid)
                    , fc1(d_hid, d_model)
                    , fc2(d_model, d_hid)
            {
                mlpack::XavierInitialization glorotInit;
                glorotInit.Initialize(fc1, fc1.n_rows, fc1.n_cols);
                glorotInit.Initialize(fc2, fc2.n_rows, fc2.n_cols);
            }

            /*!
             * @brief Main process of Position-wise Feed-forward.
             * @param input
             * @param output
             */
            void Forward(const MatType& input, MatType& output) {
                output = fc1 * input;

                mlpack::ReLU reluLayer;
                reluLayer.Forward(output, output);

                output = fc2 * output;
            }

        private:
            //! \brief
            size_t modelDim {};

            //! \brief
            size_t hiddenDim {};

            //! \brief
            MatType fc1;

            //! \brief
            MatType fc2;
        };

        /*!
         * @brief Encoder of Transformer.
         */
        class Encoder {
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
            Encoder(const size_t d_model,
                    const size_t num_heads,
                    const size_t d_hid,
                    const double drop_ratio = 0.4)
                    : modelDim(d_model)
                    , numHeads(num_heads)
                    , hiddenDim(d_hid)
                    , dropRatio(drop_ratio)
                    , attnLayer(d_model, num_heads)
                    , forwardLayer(d_model, d_hid)
            {
                // Nothing to do here.
            }

            /*!
             * @brief Main process of Encoder.
             * @param input
             * @param output
             */
            void Forward(const MatType& input, MatType& output) {
                mlpack::Dropout dropLayer(dropRatio);
                Normalize normLayer(1e-5);

                MatType attnOutput {};
                attnLayer.Forward(input, input, input, arma::ones(modelDim, modelDim), attnOutput);

                dropLayer.Forward(attnOutput, attnOutput);

                MatType x {};
                normLayer.Forward(input + attnOutput, x);

                MatType forwardOutput {};
                forwardLayer.Forward(x, forwardOutput);

                dropLayer.Forward(forwardOutput, forwardOutput);

                normLayer.Forward(x + forwardOutput, output);
            }

        private:
            //! \brief
            size_t modelDim {};

            //! \brief
            size_t numHeads {};

            //! \brief
            size_t hiddenDim {};

            //! \brief
            double dropRatio {};

            //! \brief
            MultiHeadAttention<MatType> attnLayer;

            //! \brief
            PositionWiseFeedForward forwardLayer;
        };

        /*!
         * @brief Decoder of Transformer.
         */
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
            Decoder(const size_t d_model,
                    const size_t num_heads,
                    const size_t d_hid,
                    const double drop_ratio = 0.4)
                    : modelDim(d_model)
                    , numHeads(num_heads)
                    , hiddenDim(d_hid)
                    , dropRatio(drop_ratio)
                    , attnLayer(d_model, num_heads)
                    , crossAttnLayer(d_model, num_heads)
                    , forwardLayer(d_model, d_hid)
            {
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
            void Forward(const MatType& input,
                         const MatType& enOutput,
                         const MatType& mask1,
                         const MatType& mask2,
                         MatType& output) {
                mlpack::Dropout dropLayer(dropRatio);
                Normalize normLayer(1e-5);

                MatType attnOutput {};
                attnLayer.Forward(input, input, input, mask1, attnOutput);

                dropLayer.Forward(attnOutput, attnOutput);

                MatType x {};
                normLayer.Forward(input + attnOutput, x);

                crossAttnLayer.Forward(enOutput, enOutput, x, mask2, attnOutput);

                dropLayer.Forward(attnOutput, attnOutput);

                normLayer.Forward(x + attnOutput, x);

                MatType forwardOutput {};
                forwardLayer.Forward(x, forwardOutput);

                dropLayer.Forward(forwardOutput, forwardOutput);

                normLayer.Forward(x + forwardOutput, output);
            }

        private:
            //! \brief
            size_t modelDim {};

            //! \brief
            size_t numHeads {};

            //! \brief
            size_t hiddenDim {};

            //! \brief
            double dropRatio {};

            //! \brief
            MultiHeadAttention<MatType> attnLayer;

            //! \brief
            MultiHeadAttention<MatType> crossAttnLayer;

            //! \brief
            PositionWiseFeedForward forwardLayer;
        };

    private:
        //! \brief
        size_t srcSize {};

        //! \brief
        MatType srcEmbed;

        //! \brief
        size_t tgtSize {};

        //! \brief
        MatType tgtEmbed;

        //! \brief
        size_t outSize {};

        //! \brief
        size_t modelDim {};

        //! \brief
        size_t numLayers {};

        //! \brief
        double dropRatio {};

        //! \brief
        PositionalEncoding posEncoder;

        //! \brief
        std::vector<Encoder> encoders;

        //! \brief
        std::vector<Decoder> decoders;

        //! \brief
        MatType fc;
    };
}

#endif //ML_TRANSFORMER_TRANSFORMER_HPP
