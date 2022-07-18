#include <emscripten/bind.h>
#include <emscripten/fetch.h>

#include <unordered_map>

#include "encode_and_decode_lib.h"
#include "lyra_decoder.h"
#include "lyra_encoder.h"
#include "wavegru_buffer/wavegru_buffer_interface.h"

// Encoder and Decoder.
std::unique_ptr<chromemedia::codec::LyraEncoder> encoder;
std::unique_ptr<chromemedia::codec::LyraDecoder> decoder;

std::atomic<int8_t> FETCHED_MODEL_COUNT(0);

// Forward declaration of encoder and decoder creator fucntions.
void CreateEncoder();
void CreateDecoder();

void asyncDownload(const std::string& url, const std::string& model_name);

class WebassemblyWavegruBuffer
    : public chromemedia::codec::WavegruBufferInterface {
 public:
  WebassemblyWavegruBuffer(const std::vector<std::string>& model_names,
                           const std::string& base_url)
      : base_url_(base_url) {
    fetches_by_name_.reserve(model_names.size());
    for (const auto& model_name : model_names) {
      fetches_by_name_[model_name] = nullptr;
    }
  }

  ~WebassemblyWavegruBuffer() {
    for (const auto& [model_name, fetch] : fetches_by_name_) {
      if (fetch != nullptr) {
        emscripten_fetch_close(fetch);
      }
    }
  }

  void DownloadModels() {
    for (const auto& [model_name, fetch] : fetches_by_name_) {
      if (fetch == nullptr) {
        asyncDownload(ModelUrl(model_name), model_name);
      }
    }
  }

  void SetFetch(const std::string& model_name,
                emscripten_fetch_t* fetch) const {
    fetches_by_name_[model_name] = fetch;
  }

  uint64_t GetBufferSize(const std::string& model_name) const override {
    emscripten_fetch_t* fetch =
        static_cast<emscripten_fetch_t*>(fetches_by_name_[model_name]);
    return fetch->numBytes;
  }

  const char* GetBuffer(const std::string& model_name) const override {
    emscripten_fetch_t* fetch =
        static_cast<emscripten_fetch_t*>(fetches_by_name_[model_name]);
    return fetch->data;
  }

 private:
  std::string ModelUrl(const std::string& model_name) const {
    return base_url_ + model_name;
  }

  mutable std::unordered_map<std::string, emscripten_fetch_t*> fetches_by_name_;
  const std::string base_url_;
};

// Buffer for wavegru models.
WebassemblyWavegruBuffer wavegru_buffer(
    {"lyra_16khz_ar_to_gates_bias.raw.gz",
     "lyra_16khz_ar_to_gates_mask.raw.gz",
     "lyra_16khz_ar_to_gates_weights.raw.gz",
     "lyra_16khz_conditioning_stack_0_bias.raw.gz",
     "lyra_16khz_conditioning_stack_0_mask.raw.gz",
     "lyra_16khz_conditioning_stack_0_weights.raw.gz",
     "lyra_16khz_conditioning_stack_1_bias.raw.gz",
     "lyra_16khz_conditioning_stack_1_mask.raw.gz",
     "lyra_16khz_conditioning_stack_1_weights.raw.gz",
     "lyra_16khz_conditioning_stack_2_bias.raw.gz",
     "lyra_16khz_conditioning_stack_2_mask.raw.gz",
     "lyra_16khz_conditioning_stack_2_weights.raw.gz",
     "lyra_16khz_conv1d_bias.raw.gz",
     "lyra_16khz_conv1d_mask.raw.gz",
     "lyra_16khz_conv1d_weights.raw.gz",
     "lyra_16khz_conv_cond_bias.raw.gz",
     "lyra_16khz_conv_cond_mask.raw.gz",
     "lyra_16khz_conv_cond_weights.raw.gz",
     "lyra_16khz_conv_to_gates_bias.raw.gz",
     "lyra_16khz_conv_to_gates_mask.raw.gz",
     "lyra_16khz_conv_to_gates_weights.raw.gz",
     "lyra_16khz_gru_layer_bias.raw.gz",
     "lyra_16khz_gru_layer_mask.raw.gz",
     "lyra_16khz_gru_layer_weights.raw.gz",
     "lyra_16khz_means_bias.raw.gz",
     "lyra_16khz_means_mask.raw.gz",
     "lyra_16khz_means_weights.raw.gz",
     "lyra_16khz_mix_bias.raw.gz",
     "lyra_16khz_mix_mask.raw.gz",
     "lyra_16khz_mix_weights.raw.gz",
     "lyra_16khz_proj_bias.raw.gz",
     "lyra_16khz_proj_mask.raw.gz",
     "lyra_16khz_proj_weights.raw.gz",
     "lyra_16khz_quant_codebook_dimensions.gz",
     "lyra_16khz_quant_code_vectors.gz",
     "lyra_16khz_quant_mean_vectors.gz",
     "lyra_16khz_quant_transmat.gz",
     "lyra_16khz_scales_bias.raw.gz",
     "lyra_16khz_scales_mask.raw.gz",
     "lyra_16khz_scales_weights.raw.gz",
     "lyra_16khz_transpose_0_bias.raw.gz",
     "lyra_16khz_transpose_0_mask.raw.gz",
     "lyra_16khz_transpose_0_weights.raw.gz",
     "lyra_16khz_transpose_1_bias.raw.gz",
     "lyra_16khz_transpose_1_mask.raw.gz",
     "lyra_16khz_transpose_1_weights.raw.gz",
     "lyra_16khz_transpose_2_bias.raw.gz",
     "lyra_16khz_transpose_2_mask.raw.gz",
     "lyra_16khz_transpose_2_weights.raw.gz"},
    "wavegru/");

void downloadSucceeded(emscripten_fetch_t* fetch) {
  const char* model_name = static_cast<char*>(fetch->userData);
  printf("Finished downloading %llu bytes from URL %s.\n", fetch->numBytes,
         fetch->url);
  wavegru_buffer.SetFetch(model_name, fetch);
  FETCHED_MODEL_COUNT++;
  if (FETCHED_MODEL_COUNT == 49) {
    CreateEncoder();
    if (encoder == nullptr) {
      printf("Failed to create encoder.\n");
    }
    CreateDecoder();
  }
}

void downloadFailed(emscripten_fetch_t* fetch) {
  auto fetch_ptr = static_cast<emscripten_fetch_t*>(fetch->userData);
  printf("Downloading %s failed, HTTP failure status code: %d.\n",
         fetch_ptr->url, fetch_ptr->status);
}

void asyncDownload(const std::string& url, const std::string& model_name) {
  emscripten_fetch_attr_t attr;
  emscripten_fetch_attr_init(&attr);
  strcpy(attr.requestMethod, "GET");
  attr.userData = (void*)model_name.c_str();
  attr.attributes =
      EMSCRIPTEN_FETCH_PERSIST_FILE | EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
  attr.onsuccess = downloadSucceeded;
  attr.onerror = downloadFailed;
  emscripten_fetch(&attr, url.c_str());
}

void InitializeCodec() { wavegru_buffer.DownloadModels(); }

void CreateEncoder() {
  encoder = chromemedia::codec::LyraEncoder::Create(
      /*sample_rate_hz=*/48000,
      /*num_channels=*/1,
      /*bitrate=*/3000,
      /*enable_dtx=*/true, wavegru_buffer);
  if (encoder == nullptr) {
    fprintf(stderr, "Failed to create encoder.\n");
  } else {
    fprintf(stdout, "Successfully created encoder!\n");
  }
}

void CreateDecoder() {
  decoder = chromemedia::codec::LyraDecoder::Create(
      /*sample_rate_hz=*/48000,
      /*num_channels=*/1,
      /*bitrate=*/3000, wavegru_buffer);
  if (decoder == nullptr) {
    fprintf(stderr, "Failed to create decoder.\n");
  } else {
    fprintf(stdout, "Successfully created decoder!\n");
  }
}

bool EncodeAndDecodeWithLyra(uintptr_t data, uint32_t num_samples,
                             uint32_t sample_rate_hz, uintptr_t out_data) {
  fprintf(stdout, "EncodeAndDecode called with %d samples.\n", num_samples);
  float* data_buffer = reinterpret_cast<float*>(data);

  // Convert the float input data to int16_t.
  std::vector<int16_t> data_to_encode(num_samples);
  std::transform(data_buffer, data_buffer + num_samples, data_to_encode.begin(),
                 [](float x) { return static_cast<int16_t>(x * 32767.0f); });
  fprintf(stdout, "The first sample (converted to int16_t) is %d.\n", data_to_encode[0]);
  std::copy(data_buffer, data_buffer + num_samples, data_to_encode.begin());

  auto maybe_decoded_output = chromemedia::codec::EncodeAndDecode(
      encoder.get(), decoder.get(), data_to_encode, sample_rate_hz,
      /*packet_loss_rate=*/0.f,
      /*float_average_burst_length=*/1.f);
  if (!maybe_decoded_output.has_value()) {
    fprintf(stderr, "Failed to encode and decode.\n");
    return false;
  }

  if (maybe_decoded_output.value().empty()) {
    fprintf(stderr,
            "No decoded output. The number of samples sent for encode and "
            "decode (%d) was probably too small.\n",
            num_samples);
    return false;
  }

  // Convert decode output to float.
  const int num_decoded_samples = maybe_decoded_output.value().size();
  std::vector<float> decoded_output(num_decoded_samples);
  std::transform(maybe_decoded_output.value().begin(),
                 maybe_decoded_output.value().end(), decoded_output.begin(),
                 [](int16_t x) { return x / 32767.0f; });
  float* out_data_buffer = reinterpret_cast<float*>(out_data);
  for (int i = 0; i < num_decoded_samples; i++) {
    *(out_data_buffer + i) = decoded_output[i];
  }
  fprintf(stdout, "Encode and decode succeeded. Returning %d samples.\n",
          decoded_output.size());
  return true;
}

bool IsCodecReady() { return encoder != nullptr && decoder != nullptr; }

int main(int argc, char* argv[]) {
  InitializeCodec();
  return 0;
}

EMSCRIPTEN_BINDINGS(module) {
  emscripten::function("isCodecReady", IsCodecReady);
  emscripten::function("encodeAndDecode", EncodeAndDecodeWithLyra,
                       emscripten::allow_raw_pointers());
}