#include "encode_and_decode_lib.h"

#include "runfiles_util.h"

int main(int argc, char* argv[]) {
  fprintf(stderr, "Runfiles directory is %s\n", tools::GetModelRunfilesPath(argv[0]).c_str());
  chromemedia::codec::End2End("48khz_playback.wav", "48khz_playback.wav", argv[0]);
  return 0;
}