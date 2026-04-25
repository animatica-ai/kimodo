# Troubleshooting installation

Notes for installing `kimodo` on a fresh machine. The `pyproject.toml` install
also builds the `MotionCorrection` C++ extension via CMake, which is where most
issues happen.

## macOS (Apple Silicon, ARM64)

### `CMake Error: SIMDe headers not found`

The native build relies on [SIMDe](https://github.com/simd-everywhere/simde) to
translate x86 SSE/AVX intrinsics to NEON on ARM64.

```bash
brew install simde
CMAKE_PREFIX_PATH=/opt/homebrew uv sync
```

`CMAKE_PREFIX_PATH` is needed so CMake's `find_path` picks up Homebrew's
`/opt/homebrew/include/simde`.

### `error: "This header is only meant to be used on x86 and x64"`

If you hit this on ARM64, your tree is missing the conditional include guard in
`MotionCorrection/src/cpp/Math/SIMD.h`. The fix: ensure `<immintrin.h>` is only
included in the x86 branch (the file should NOT include `<immintrin.h>` before
the `#if defined(__aarch64__) || defined(__ARM_NEON)` block).

### `error: unknown type name 'FORCE_INLINE'`

`MotionCorrection/src/cpp/Compiler.h` historically only defined `FORCE_INLINE`
for MSVC and GCC. Apple Clang sets `COMPILER_CLANG`, leaving it undefined.
The Clang branch should look like:

```c
#elif defined(COMPILER_GNUC) || defined(COMPILER_CLANG)
    #define FORCE_INLINE inline __attribute__((always_inline))
```

## Linux

### `cmake: command not found` or version too old

The build requires CMake ≥ 3.15.

```bash
# Ubuntu/Debian
sudo apt install cmake build-essential

# RHEL/Fedora
sudo dnf install cmake gcc-c++
```

### `pybind11 not found, fetching from GitHub...`

This is a status message, not an error — the build will fetch pybind11 v2.11.1
automatically. Same for Eigen (3.4.0).

If you're behind a proxy, pre-install both to avoid the fetch:

```bash
sudo apt install pybind11-dev libeigen3-dev
```

## All platforms

### Build seems to hang

CMake's `FetchContent` clones pybind11 and Eigen on the first build. With a slow
connection this can look stuck — give it a few minutes before killing it.

### Stale build cache after a fix

`uv sync` caches build outputs. After editing files in `MotionCorrection/`,
clear the cache to force a rebuild:

```bash
rm -rf ~/.cache/uv/builds-v0
uv sync
```

### `transformers==5.1.0` resolution conflicts

Kimodo pins `transformers==5.1.0`. If you mix it into another environment, use a
dedicated venv (uv does this for you).

## Runtime

### Out of VRAM loading the LLM2Vec text encoder

LLM2Vec needs ~17 GB VRAM. Use the dummy encoder for constraint-only or
text-ignored generation:

```bash
TEXT_ENCODER_MODE=dummy uv run kimodo_gen "a person walks"
```

The dummy encoder returns zero embeddings — text prompts are ignored, so use
constraints to control motion.

### `Could not resolve model '...' from Hugging Face`

Models live on the Hub under `nv-tlabs/`. Check internet access, or pre-download
into a local cache and point the loader at it:

```bash
export CHECKPOINT_DIR=/path/to/local/checkpoints
export LOCAL_CACHE=true
```

### Verifying a successful install

```bash
uv run python -c "import kimodo; from motion_correction import _motion_correction; print('OK')"
```
