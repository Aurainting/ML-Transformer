#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "position_encoding.hpp"

TEST_CASE("Test Positional Encoding") {
  const ml_transformer::PositionalEncoding pe_layer(8);
}
