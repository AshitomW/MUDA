class Muda < Formula
  desc "CUDA-to-Metal compatibility layer for Apple Silicon"
  homepage "https://github.com/AshitomW/cuda-metal"
  version "0.1.0"
  license "Apache-2.0"

  depends_on "cmake" => :build
  depends_on "ninja" => :build
  depends_on :macos
  depends_on arch: :arm64

  def install
    system "cmake", "-S", ".", "-B", "build", "-GNinja",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DMUDA_BUILD_TESTS=OFF",
                    *std_cmake_args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
  end

  def caveats
    <<~EOS
      To use MUDA as a drop-in CUDA_HOME for CMake, PyTorch extensions, or Numba:
        export CUDA_HOME="#{opt_share}/muda/cuda-home"
        export PATH="#{opt_bin}:$PATH"
        export DYLD_LIBRARY_PATH="#{opt_lib}:$DYLD_LIBRARY_PATH"

      Verify your installation with:
        muda doctor --smoke-test
    EOS
  end

  test do
    assert_match "MUDA", shell_output("#{bin}/muda version")
    assert_match "mudac", shell_output("#{bin}/nvcc --version")
  end
end
