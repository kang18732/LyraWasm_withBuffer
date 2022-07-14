#include <emscripten/bind.h>
#include <emscripten/fetch.h>

#include "lyra_decoder.h"
#include "lyra_encoder.h"
#include "encode_and_decode_lib.h"


// Encoder and Decoder.
std::unique_ptr<chromemedia::codec::LyraEncoder> encoder;
std::unique_ptr<chromemedia::codec::LyraDecoder> decoder;


void EncodeAndDecodeWithLyra(uintptr_t data, uint32_t num_samples,
                                    uint32_t sample_rate_hz,
                                    uintptr_t out_data) {
  fprintf(stderr, "EncodeAndDecode called.\n");
  float* data_buffer = reinterpret_cast<float*>(data);

  // Convert the float input data to int16_t.
  std::vector<int16_t> data_to_encode(num_samples);
  std::transform(data_buffer,
                 data_buffer + num_samples, data_to_encode.begin(),
                 [](float x) { return static_cast<int16_t>(x * 32768.0f); });
  std::copy(data_buffer, data_buffer + num_samples, data_to_encode.begin());

  auto maybe_decoded_output = chromemedia::codec::EncodeAndDecode(
      encoder.get(), decoder.get(), data_to_encode, sample_rate_hz,
      /*bitrate=*/3200, /*packet_loss_rate=*/0.f,
      /*float_average_burst_length=*/1.f);
  if (!maybe_decoded_output.has_value()) {
    fprintf(stderr, "Failed to encode and decode.\n");
    return;
  }

  // Convert decode output to float.
  const int num_decoded_samples = maybe_decoded_output.value().size();
  std::vector<float> decoded_output(num_decoded_samples);
        std::transform(maybe_decoded_output.value().begin(),
                         maybe_decoded_output.value().end(), decoded_output.begin(),
                         [](int16_t x) { return static_cast<float>(x) / 32768.0f; });
  float* out_data_buffer = reinterpret_cast<float*>(out_data);
  for (int i = 0; i < num_decoded_samples; i++) {
    *(out_data_buffer + i) = decoded_output[i];
  }
}

EMSCRIPTEN_BINDINGS(module) {
  emscripten::function("encodeAndDecode", EncodeAndDecodeWithLyra,
                       emscripten::allow_raw_pointers());
}