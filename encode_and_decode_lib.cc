#include "encode_and_decode_lib.h"

#include <iostream>
#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gilbert_model.h"
#include "lyra_config.h"
#include "lyra_decoder.h"
#include "lyra_encoder.h"
#include "packet.h"
#include "packet_loss_handler_interface.h"
#include "runfiles_util.h"
#include "wav_util.h"

namespace chromemedia {
namespace codec {
namespace {

int PacketSize(LyraDecoder* decoder) {
  const float bits_per_packet = static_cast<float>(decoder->bitrate()) /
                                decoder->frame_rate() * kNumFramesPerPacket;
  const float bytes_per_packet = bits_per_packet / CHAR_BIT;
  return static_cast<int>(std::ceil(bytes_per_packet));
}

}  // namespace

std::optional<std::vector<uint8_t>> EncodeWithEncoder(
    LyraEncoder* encoder, const std::vector<int16_t>& wav_data,
    int sample_rate_hz) {
  // Encode the wav data and store the encoded features in a vector.
  const auto benchmark_start = absl::Now();
  std::vector<uint8_t> encoded_features;
  int num_samples_per_packet =
      kNumFramesPerPacket * sample_rate_hz / encoder->frame_rate();
  // Iterate over the wav data until the end of the vector.
  for (int wav_iterator = 0;
       wav_iterator + num_samples_per_packet <= wav_data.size();
       wav_iterator += num_samples_per_packet) {
    // Move audio samples from the large in memory wav file frame by frame to
    // the encoder.
    auto encoded_or = encoder->Encode(absl::MakeConstSpan(
        &wav_data.at(wav_iterator), num_samples_per_packet));
    if (!encoded_or.has_value()) {
      std::cerr << "Unable to encode features starting at samples at byte "
                << wav_iterator << ".";
      return std::nullopt;
    }

    // Append the encoded audio frames to the encoded_features accumulator
    // vector.
    encoded_features.insert(encoded_features.end(), encoded_or.value().begin(),
                            encoded_or.value().end());
  }
  const auto elapsed = absl::Now() - benchmark_start;
  fprintf(stdout, "Encoding lapsed seconds %ld.\n",
          absl::ToInt64Seconds(elapsed));
  fprintf(stdout, "Encoding samples per second %f.\n",
          wav_data.size() / absl::ToDoubleSeconds(elapsed));
  fprintf(stdout, "Encoded features size %d.\n", encoded_features.size());
  if (!encoded_features.empty()) {
    fprintf(stdout, "The first encoded feature has value %d\n",
            encoded_features[0]);
  }
  return encoded_features;
}

std::optional<std::vector<int16_t>> DecodeWithDecoder(
    LyraDecoder* decoder, const std::vector<uint8_t>& encoded_features,
    float packet_loss_rate, float average_burst_length) {
  // Decode the encoded features and return the reconstructed audio.
  auto gilbert_model =
      GilbertModel::Create(packet_loss_rate, average_burst_length);
  if (gilbert_model == nullptr) {
    std::cerr << "Could not create a packet loss model." << std::endl;
    return std::nullopt;
  }

  const int packet_size = PacketSize(decoder);
  const int num_samples_per_packet =
      kNumFramesPerPacket * GetNumSamplesPerHop(decoder->sample_rate_hz());
  std::vector<int16_t> decoded_audio;
  const auto decode_benchmark_start = absl::Now();
  for (int encoded_index = 0; encoded_index < encoded_features.size();
       encoded_index += packet_size) {
    const absl::Span<const uint8_t> encoded_packet = absl::MakeConstSpan(
        encoded_features.data() + encoded_index, packet_size);

    absl::optional<std::vector<int16_t>> decoded_or;
    if (gilbert_model->IsPacketReceived()) {
      if (!decoder->SetEncodedPacket(encoded_packet)) {
        std::cerr << "ERROR: Unable to set encoded packet starting at byte "
                  << encoded_index << std::endl;
        return absl::nullopt;
      }
      decoded_or = decoder->DecodeSamples(num_samples_per_packet);
    } else {
      fprintf(stdout, "INFO: Decoding a packet in PLC mode.\n");
      decoded_or = decoder->DecodePacketLoss(num_samples_per_packet);
    }

    if (!decoded_or.has_value()) {
      std::cerr << "ERROR: Unable to decode features starting at byte "
                << encoded_index << std::endl;
      return std::nullopt;
    } else {
      decoded_audio.insert(decoded_audio.end(), decoded_or.value().begin(),
                           decoded_or.value().end());
    }
  }

  const auto decode_elapsed = absl::Now() - decode_benchmark_start;
  std::cout << "INFO: Decoding elapsed seconds : "
            << absl::ToInt64Seconds(decode_elapsed) << std::endl;
  std::cout << "INFO: Decoding samples per second : "
            << decoded_audio.size() / absl::ToDoubleSeconds(decode_elapsed)
            << std::endl;
  fprintf(stdout, "Output from decoder has number of samples %ld\n",
          decoded_audio.size());
  if (!decoded_audio.empty()) {
    fprintf(stdout, "The first output sample is %d\n", decoded_audio[0]);
  }
  return decoded_audio;
}

std::optional<std::vector<int16_t>> EncodeAndDecode(
    LyraEncoder* encoder, LyraDecoder* decoder,
    const std::vector<int16_t>& wav_data, int sample_rate_hz,
    float packet_loss_rate, float average_burst_length) {
  auto encoded_features_or =
      EncodeWithEncoder(encoder, wav_data, sample_rate_hz);

  if (!encoded_features_or.has_value()) {
    std::cerr << "Unable to encode features." << std::endl;
    return std::nullopt;
  }

  return DecodeWithDecoder(decoder, encoded_features_or.value(),
                           packet_loss_rate, average_burst_length);
}

bool End2End(const std::string& input_filename,
             const std::string& output_filename, const std::string& arg0) {
  const std::string output_path =
      tools::GetTestdataRunfilesPath(arg0) + output_filename + "_decoded";
  const std::string model_path = tools::GetModelRunfilesPath(arg0);
  const std::string input_path =
      tools::GetTestdataRunfilesPath(arg0) + input_filename;

  // Reads the entire wav file into memory.
  absl::StatusOr<ReadWavResult> read_wav_result =
      Read16BitWavFileToVector(input_path);

  if (!read_wav_result.ok()) {
    fprintf(stderr, "Reading wavfile failed.\n");
    std::cerr << read_wav_result.status() << std::endl;
    return false;
  }

  std::unique_ptr<LyraEncoder> encoder =
      chromemedia::codec::LyraEncoder::Create(
          /*sample_rate_hz=*/read_wav_result->sample_rate_hz,
          /*num_channels=*/read_wav_result->num_channels,
          /*bitrate=*/3000,
          /*enable_dtx=*/true, model_path);

  if (encoder == nullptr) {
    fprintf(stderr, "Failed to create encoder.\n");
    return false;
  }

  std::unique_ptr<LyraDecoder> decoder =
      chromemedia::codec::LyraDecoder::Create(
          /*sample_rate_hz=*/48000,
          /*num_channels=*/1, /*bitrate=*/3000, model_path);

  if (decoder == nullptr) {
    fprintf(stderr, "Failed to create decoder.\n");
    return false;
  }

  std::vector<int16_t> data_to_encode(read_wav_result->samples.begin(),
                                      read_wav_result->samples.begin() + 49920);

  auto output_or =
      EncodeAndDecode(encoder.get(), decoder.get(), data_to_encode,
                      /*sample_rate_hz=*/48000, /*packet_loss_rate=*/0.f,
                      /*float_average_burst_length=*/1.f);

  if (!output_or.has_value()) {
    fprintf(stderr, "EncodeAndDecode failed. \n");
    return false;
  }

  absl::Status write_status =
      Write16BitWavFileFromVector(output_path, /*kNumChannels=*/1,
                                  /*sample_rate_hz=*/48000, output_or.value());
  if (!write_status.ok()) {
    fprintf(stderr, "Writing output to file failed.");
    std::cerr << write_status << std::endl;
    return false;
  }

  fprintf(stdout, "EncodeAndDecode successful.\n");
  fprintf(stdout, "Input file: %s\n", input_filename.c_str());
  fprintf(stdout, "Decoded output file path: %s\n", output_path.c_str());
  return true;
}

}  // namespace codec
}  // namespace chromemedia
