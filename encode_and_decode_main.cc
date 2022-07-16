#include "encode_and_decode_lib.h"
#include "runfiles_util.h"
#include "include/ghc/filesystem.hpp"

int main(int argc, char* argv[]) {
  chromemedia::codec::End2End("48khz_sample_000003.wav",
                              "48khz_sample_000003.wav", argv[0]);
  fprintf(stderr, "Runfiles directory is %s\n",
          tools::GetModelRunfilesPath(argv[0]).c_str());
  fprintf(stderr, "Current directory is %s\n",
          ghc::filesystem::current_path().c_str());
  return 0;
}