#ifndef MUDA_FP64_METAL_
#define MUDA_FP64_METAL_

#include <metal_stdlib>

using namespace metal;

#if defined(MUDA_FP64_IEEE)

// =============================================================================
// MUDA FP64: Bit-Exact IEEE-754 64-Bit Integer Bitfield Software Emulation
// =============================================================================

namespace {

struct muda_ieee_unpacked {
  bool sign;
  int exp;      // unbiased exponent
  ulong sig;    // 53-bit significand with implicit leading 1 at bit 52
  bool is_nan;
  bool is_inf;
  bool is_zero;
};

inline muda_ieee_unpacked muda_ieee_unpack(ulong u) {
  muda_ieee_unpacked res;
  res.sign = (u >> 63) != 0;
  uint exp_bits = static_cast<uint>((u >> 52) & 0x7FF);
  ulong frac = u & 0x000FFFFFFFFFFFFFULL;

  if (exp_bits == 0x7FF) {
    res.is_nan = (frac != 0);
    res.is_inf = (frac == 0);
    res.is_zero = false;
    res.exp = 2047;
    res.sig = frac;
  } else if (exp_bits == 0) {
    res.is_nan = false;
    res.is_inf = false;
    res.is_zero = (frac == 0);
    if (frac == 0) {
      res.exp = -1022;
      res.sig = 0;
    } else {
      int lz = clz(frac) - 11;
      res.sig = frac << lz;
      res.exp = 1 - 1023 - lz;
    }
  } else {
    res.is_nan = false;
    res.is_inf = false;
    res.is_zero = false;
    res.exp = static_cast<int>(exp_bits) - 1023;
    res.sig = (1ULL << 52) | frac;
  }
  return res;
}

inline ulong muda_ieee_pack(bool sign, int exp, ulong sig, ulong round_sticky) {
  ulong sign_bit = sign ? 0x8000000000000000ULL : 0ULL;
  if (sig == 0) return sign_bit;

  bool round_bit = (round_sticky & 2ULL) != 0;
  bool sticky_bit = (round_sticky & 1ULL) != 0;
  bool lsb = (sig & 1ULL) != 0;

  if (round_bit && (sticky_bit || lsb)) {
    sig += 1;
    if (sig >= (1ULL << 53)) {
      sig >>= 1;
      exp += 1;
    }
  }

  int biased_exp = exp + 1023;
  if (biased_exp >= 2047) {
    return sign_bit | 0x7FF0000000000000ULL;
  }
  if (biased_exp <= 0) {
    int shift = 1 - biased_exp;
    if (shift >= 64) return sign_bit;
    sig >>= shift;
    return sign_bit | (sig & 0x000FFFFFFFFFFFFFULL);
  }

  ulong frac = sig & 0x000FFFFFFFFFFFFFULL;
  return sign_bit | (static_cast<ulong>(biased_exp) << 52) | frac;
}

inline void muda_ieee_mul64x64(ulong a, ulong b, thread ulong& hi, thread ulong& lo) {
  ulong a_lo = a & 0xFFFFFFFFULL;
  ulong a_hi = a >> 32;
  ulong b_lo = b & 0xFFFFFFFFULL;
  ulong b_hi = b >> 32;

  ulong p0 = a_lo * b_lo;
  ulong p1 = a_lo * b_hi;
  ulong p2 = a_hi * b_lo;
  ulong p3 = a_hi * b_hi;

  ulong mid = (p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL);
  lo = (p0 & 0xFFFFFFFFULL) | (mid << 32);
  hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
}

inline ulong muda_ieee64_add(ulong a, ulong b) {
  muda_ieee_unpacked ua = muda_ieee_unpack(a);
  muda_ieee_unpacked ub = muda_ieee_unpack(b);

  if (ua.is_nan) return a | 0x0008000000000000ULL;
  if (ub.is_nan) return b | 0x0008000000000000ULL;
  if (ua.is_inf && ub.is_inf) {
    if (ua.sign != ub.sign) return 0x7FF8000000000000ULL;
    return a;
  }
  if (ua.is_inf) return a;
  if (ub.is_inf) return b;
  if (ua.is_zero) return ub.is_zero ? ((ua.sign && ub.sign) ? a : 0ULL) : b;
  if (ub.is_zero) return a;

  if (ua.exp < ub.exp || (ua.exp == ub.exp && ua.sig < ub.sig)) {
    muda_ieee_unpacked tmp_u = ua; ua = ub; ub = tmp_u;
  }

  int exp_diff = ua.exp - ub.exp;
  ulong sig_a = ua.sig << 3;
  ulong sig_b = 0;
  if (exp_diff >= 56) {
    sig_b = (ub.sig != 0) ? 1ULL : 0ULL;
  } else {
    ulong shifted_b = ub.sig << 3;
    ulong discarded = (exp_diff > 0) ? (shifted_b & ((1ULL << exp_diff) - 1)) : 0ULL;
    sig_b = (shifted_b >> exp_diff) | (discarded != 0 ? 1ULL : 0ULL);
  }

  ulong res_sig = 0;
  bool res_sign = ua.sign;
  int res_exp = ua.exp;

  if (ua.sign == ub.sign) {
    res_sig = sig_a + sig_b;
    if (res_sig >= (1ULL << 56)) {
      bool sticky = (res_sig & 1) != 0;
      res_sig >>= 1;
      if (sticky) res_sig |= 1;
      res_exp += 1;
    }
  } else {
    res_sig = sig_a - sig_b;
    if (res_sig == 0) {
      return 0ULL;
    }
    int lz = clz(res_sig) - 8;
    if (lz > 0) {
      res_sig <<= lz;
      res_exp -= lz;
    }
  }

  ulong sig = res_sig >> 3;
  ulong grs = res_sig & 7;
  bool round_bit = (grs & 4) != 0;
  bool sticky_bit = (grs & 3) != 0;
  ulong round_sticky = (round_bit ? 2ULL : 0ULL) | (sticky_bit ? 1ULL : 0ULL);
  return muda_ieee_pack(res_sign, res_exp, sig, round_sticky);
}

inline ulong muda_ieee64_sub(ulong a, ulong b) {
  return muda_ieee64_add(a, b ^ 0x8000000000000000ULL);
}

inline ulong muda_ieee64_mul(ulong a, ulong b) {
  muda_ieee_unpacked ua = muda_ieee_unpack(a);
  muda_ieee_unpacked ub = muda_ieee_unpack(b);

  if (ua.is_nan) return a | 0x0008000000000000ULL;
  if (ub.is_nan) return b | 0x0008000000000000ULL;

  bool res_sign = ua.sign ^ ub.sign;
  ulong sign_bit = res_sign ? 0x8000000000000000ULL : 0ULL;

  if (ua.is_inf || ub.is_inf) {
    if (ua.is_zero || ub.is_zero) return 0x7FF8000000000000ULL;
    return sign_bit | 0x7FF0000000000000ULL;
  }
  if (ua.is_zero || ub.is_zero) return sign_bit;

  ulong hi = 0, lo = 0;
  muda_ieee_mul64x64(ua.sig, ub.sig, hi, lo);

  int res_exp = ua.exp + ub.exp;
  ulong res_sig = 0;
  ulong round_sticky = 0;

  if ((hi & (1ULL << 41)) != 0) {
    res_exp += 1;
    res_sig = (hi << 11) | (lo >> 53);
    bool round = (lo & (1ULL << 52)) != 0;
    bool sticky = (lo & ((1ULL << 52) - 1)) != 0;
    round_sticky = (round ? 2ULL : 0ULL) | (sticky ? 1ULL : 0ULL);
  } else {
    res_sig = (hi << 12) | (lo >> 52);
    bool round = (lo & (1ULL << 51)) != 0;
    bool sticky = (lo & ((1ULL << 51) - 1)) != 0;
    round_sticky = (round ? 2ULL : 0ULL) | (sticky ? 1ULL : 0ULL);
  }

  return muda_ieee_pack(res_sign, res_exp, res_sig, round_sticky);
}

inline ulong muda_ieee64_div(ulong a, ulong b) {
  muda_ieee_unpacked ua = muda_ieee_unpack(a);
  muda_ieee_unpacked ub = muda_ieee_unpack(b);

  if (ua.is_nan) return a | 0x0008000000000000ULL;
  if (ub.is_nan) return b | 0x0008000000000000ULL;

  bool res_sign = ua.sign ^ ub.sign;
  ulong sign_bit = res_sign ? 0x8000000000000000ULL : 0ULL;

  if (ua.is_inf && ub.is_inf) return 0x7FF8000000000000ULL;
  if (ua.is_zero && ub.is_zero) return 0x7FF8000000000000ULL;
  if (ua.is_inf || ub.is_zero) return sign_bit | 0x7FF0000000000000ULL;
  if (ua.is_zero || ub.is_inf) return sign_bit;

  int res_exp = ua.exp - ub.exp;
  ulong q = 0;
  ulong rem = ua.sig;
  ulong div = ub.sig;

  if (rem < div) {
    rem <<= 1;
    res_exp -= 1;
  }

  for (int i = 0; i < 55; ++i) {
    q <<= 1;
    if (rem >= div) {
      rem -= div;
      q |= 1;
    }
    rem <<= 1;
  }

  ulong res_sig = q >> 2;
  bool round = (q & 2) != 0;
  bool sticky = ((q & 1) != 0) || (rem != 0);
  ulong round_sticky = (round ? 2ULL : 0ULL) | (sticky ? 1ULL : 0ULL);

  return muda_ieee_pack(res_sign, res_exp, res_sig, round_sticky);
}

inline bool muda_ieee_ge128(ulong ahi, ulong alo, ulong bhi, ulong blo) {
  return (ahi > bhi) || (ahi == bhi && alo >= blo);
}

inline void muda_ieee_sub128(thread ulong& ahi, thread ulong& alo, ulong bhi, ulong blo) {
  ulong borrow = (alo < blo) ? 1 : 0;
  alo -= blo;
  ahi = ahi - bhi - borrow;
}

inline void muda_ieee_add128(thread ulong& ahi, thread ulong& alo, ulong bhi, ulong blo) {
  alo += blo;
  ahi += bhi + (alo < blo ? 1 : 0);
}

inline void muda_ieee_shr128_1(thread ulong& hi, thread ulong& lo) {
  lo = (lo >> 1) | (hi << 63);
  hi = hi >> 1;
}

inline void muda_ieee_shr128_2(thread ulong& hi, thread ulong& lo) {
  lo = (lo >> 2) | (hi << 62);
  hi = hi >> 2;
}

inline ulong muda_ieee64_sqrt(ulong a) {
  uint exp_bits = static_cast<uint>((a >> 52) & 0x7FF);
  ulong frac = a & 0x000FFFFFFFFFFFFFULL;
  if (exp_bits == 0x7FF) return (frac != 0) ? (a | 0x0008000000000000ULL) : a;
  if ((a >> 63) != 0) {
    if ((a & 0x7FFFFFFFFFFFFFFFULL) == 0) return a;
    return 0x7FF8000000000000ULL;
  }
  if ((a & 0x7FFFFFFFFFFFFFFFULL) == 0 || exp_bits == 0x7FF) return a;

  int exp = 0;
  ulong m = 0;
  if (exp_bits == 0) {
    int lz = clz(frac) - 11;
    m = frac << lz;
    exp = 1 - 1023 - lz;
  } else {
    exp = static_cast<int>(exp_bits) - 1023;
    m = (1ULL << 52) | frac;
  }

  if (exp & 1) {
    m <<= 1;
    exp -= 1;
  }
  int res_exp = exp / 2;

  ulong rem_hi = m >> 10;
  ulong rem_lo = m << 54;

  ulong root_hi = 0, root_lo = 0;
  ulong bit_hi = 1ULL << 42;
  ulong bit_lo = 0;

  while (muda_ieee_ge128(bit_hi, bit_lo, rem_hi, rem_lo) && (bit_hi != 0 || bit_lo != 0)) {
    muda_ieee_shr128_2(bit_hi, bit_lo);
  }

  while (bit_hi != 0 || bit_lo != 0) {
    ulong test_hi = root_hi;
    ulong test_lo = root_lo;
    muda_ieee_add128(test_hi, test_lo, bit_hi, bit_lo);

    if (muda_ieee_ge128(rem_hi, rem_lo, test_hi, test_lo)) {
      muda_ieee_sub128(rem_hi, rem_lo, test_hi, test_lo);
      muda_ieee_shr128_1(root_hi, root_lo);
      muda_ieee_add128(root_hi, root_lo, bit_hi, bit_lo);
    } else {
      muda_ieee_shr128_1(root_hi, root_lo);
    }
    muda_ieee_shr128_2(bit_hi, bit_lo);
  }

  ulong q = root_lo;
  ulong res_sig = q >> 1;
  bool round = (q & 1) != 0;
  bool sticky = (rem_hi != 0 || rem_lo != 0);

  if (round && (sticky || (res_sig & 1) != 0)) {
    res_sig += 1;
    if (res_sig >= (1ULL << 53)) {
      res_sig >>= 1;
      res_exp += 1;
    }
  }

  int biased_exp = res_exp + 1023;
  if (biased_exp >= 2047) return 0x7FF0000000000000ULL;
  if (biased_exp <= 0) {
    int shift = 1 - biased_exp;
    if (shift >= 64) return 0ULL;
    res_sig >>= shift;
    return res_sig & 0x000FFFFFFFFFFFFFULL;
  }
  return (static_cast<ulong>(biased_exp) << 52) | (res_sig & 0x000FFFFFFFFFFFFFULL);
}

inline ulong muda_ieee64_from_float(float f) {
  uint u = as_type<uint>(f);
  ulong sign = static_cast<ulong>(u >> 31) << 63;
  uint exp = (u >> 23) & 0xFF;
  ulong frac = static_cast<ulong>(u & 0x7FFFFF) << 29;

  if (exp == 0xFF) {
    return sign | 0x7FF0000000000000ULL | (frac ? 0x0008000000000000ULL : 0ULL);
  }
  if (exp == 0) {
    if (frac == 0) return sign;
    int lz = clz(u & 0x7FFFFF) - 9;
    ulong norm_frac = static_cast<ulong>(u & 0x7FFFFF) << (29 + lz);
    ulong double_exp = static_cast<ulong>(1 - 127 + 1023 - lz);
    return sign | (double_exp << 52) | (norm_frac & 0x000FFFFFFFFFFFFFULL);
  }
  ulong double_exp = static_cast<ulong>(exp - 127 + 1023);
  return sign | (double_exp << 52) | frac;
}

inline float muda_ieee64_to_float(ulong u) {
  uint sign = static_cast<uint>(u >> 63) << 31;
  uint exp = static_cast<uint>((u >> 52) & 0x7FF);
  ulong frac = u & 0x000FFFFFFFFFFFFFULL;

  if (exp == 0x7FF) {
    return as_type<float>(sign | 0x7F800000u | (frac ? 0x00400000u : 0u));
  }
  if (exp == 0 && frac == 0) {
    return as_type<float>(sign);
  }

  int unbiased = static_cast<int>(exp) - 1023;
  int float_exp = unbiased + 127;

  if (float_exp >= 255) {
    return as_type<float>(sign | 0x7F800000u);
  }
  if (float_exp <= 0) {
    return as_type<float>(sign);
  }

  uint float_frac = static_cast<uint>(frac >> 29);
  bool round = (frac & (1ULL << 28)) != 0;
  bool sticky = (frac & ((1ULL << 28) - 1)) != 0;
  if (round && (sticky || (float_frac & 1) != 0)) {
    float_frac += 1;
    if (float_frac >= (1u << 23)) {
      float_frac = 0;
      float_exp += 1;
    }
  }
  return as_type<float>(sign | (static_cast<uint>(float_exp) << 23) | (float_frac & 0x7FFFFFu));
}

inline ulong muda_ieee64_from_int(int i) {
  if (i == 0) return 0ULL;
  bool sign = i < 0;
  uint u = sign ? static_cast<uint>(-i) : static_cast<uint>(i);
  int lz = clz(u);
  int exp = 31 - lz;
  ulong sig = static_cast<ulong>(u) << (52 - exp);
  ulong double_exp = static_cast<ulong>(exp + 1023);
  return (sign ? 0x8000000000000000ULL : 0ULL) | (double_exp << 52) | (sig & 0x000FFFFFFFFFFFFFULL);
}

inline ulong muda_ieee64_from_uint(uint u) {
  if (u == 0) return 0ULL;
  int lz = clz(u);
  int exp = 31 - lz;
  ulong sig = static_cast<ulong>(u) << (52 - exp);
  ulong double_exp = static_cast<ulong>(exp + 1023);
  return (double_exp << 52) | (sig & 0x000FFFFFFFFFFFFFULL);
}

inline int muda_ieee64_to_int(ulong u) {
  muda_ieee_unpacked up = muda_ieee_unpack(u);
  if (up.is_nan || up.is_zero || up.exp < 0) return 0;
  if (up.exp > 30) return up.sign ? static_cast<int>(0x80000000u) : 0x7FFFFFFF;
  int shift = 52 - up.exp;
  uint val = static_cast<uint>(up.sig >> shift);
  return up.sign ? -static_cast<int>(val) : static_cast<int>(val);
}

inline uint muda_ieee64_to_uint(ulong u) {
  muda_ieee_unpacked up = muda_ieee_unpack(u);
  if (up.is_nan || up.is_zero || up.sign || up.exp < 0) return 0u;
  if (up.exp > 31) return 0xFFFFFFFFu;
  int shift = 52 - up.exp;
  return static_cast<uint>(up.sig >> shift);
}

inline bool muda_ieee64_eq(ulong a, ulong b) {
  if ((((a >> 52) & 0x7FF) == 0x7FF && (a & 0x000FFFFFFFFFFFFFULL) != 0) ||
      (((b >> 52) & 0x7FF) == 0x7FF && (b & 0x000FFFFFFFFFFFFFULL) != 0)) {
    return false;
  }
  if (((a | b) & 0x7FFFFFFFFFFFFFFFULL) == 0) return true;
  return a == b;
}

inline bool muda_ieee64_lt(ulong a, ulong b) {
  if ((((a >> 52) & 0x7FF) == 0x7FF && (a & 0x000FFFFFFFFFFFFFULL) != 0) ||
      (((b >> 52) & 0x7FF) == 0x7FF && (b & 0x000FFFFFFFFFFFFFULL) != 0)) {
    return false;
  }
  if (((a | b) & 0x7FFFFFFFFFFFFFFFULL) == 0) return false;
  bool sa = (a >> 63) != 0;
  bool sb = (b >> 63) != 0;
  if (sa != sb) return sa;
  return sa ? (a > b) : (a < b);
}

}  // namespace

struct alignas(8) muda_fp64 {
  ulong raw_bits;

  muda_fp64() thread = default;
  muda_fp64() device = default;
  muda_fp64() constant = default;
  muda_fp64() threadgroup = default;

  muda_fp64(const thread muda_fp64& o) thread : raw_bits(o.raw_bits) {}
  muda_fp64(const device muda_fp64& o) thread : raw_bits(o.raw_bits) {}
  muda_fp64(const constant muda_fp64& o) thread : raw_bits(o.raw_bits) {}
  muda_fp64(const threadgroup muda_fp64& o) thread : raw_bits(o.raw_bits) {}

  muda_fp64(ulong raw) thread : raw_bits(raw) {}
  muda_fp64(float f) thread : raw_bits(muda_ieee64_from_float(f)) {}
  muda_fp64(int i) thread : raw_bits(muda_ieee64_from_int(i)) {}
  muda_fp64(uint u) thread : raw_bits(muda_ieee64_from_uint(u)) {}

  operator float() const thread { return muda_ieee64_to_float(raw_bits); }
  operator float() const device { return muda_ieee64_to_float(raw_bits); }
  operator float() const constant { return muda_ieee64_to_float(raw_bits); }
  operator float() const threadgroup { return muda_ieee64_to_float(raw_bits); }

  operator int() const thread { return muda_ieee64_to_int(raw_bits); }
  operator int() const device { return muda_ieee64_to_int(raw_bits); }
  operator int() const constant { return muda_ieee64_to_int(raw_bits); }
  operator int() const threadgroup { return muda_ieee64_to_int(raw_bits); }

  operator uint() const thread { return muda_ieee64_to_uint(raw_bits); }
  operator uint() const device { return muda_ieee64_to_uint(raw_bits); }
  operator uint() const constant { return muda_ieee64_to_uint(raw_bits); }
  operator uint() const threadgroup { return muda_ieee64_to_uint(raw_bits); }

  thread muda_fp64& operator=(const thread muda_fp64& o) thread { raw_bits = o.raw_bits; return *this; }
  device muda_fp64& operator=(const thread muda_fp64& o) device { raw_bits = o.raw_bits; return *this; }
  threadgroup muda_fp64& operator=(const thread muda_fp64& o) threadgroup { raw_bits = o.raw_bits; return *this; }
};

inline muda_fp64 operator+(muda_fp64 a, muda_fp64 b) {
  return muda_fp64(muda_ieee64_add(a.raw_bits, b.raw_bits));
}

inline muda_fp64 operator-(muda_fp64 a, muda_fp64 b) {
  return muda_fp64(muda_ieee64_sub(a.raw_bits, b.raw_bits));
}

inline muda_fp64 operator*(muda_fp64 a, muda_fp64 b) {
  return muda_fp64(muda_ieee64_mul(a.raw_bits, b.raw_bits));
}

inline muda_fp64 operator/(muda_fp64 a, muda_fp64 b) {
  return muda_fp64(muda_ieee64_div(a.raw_bits, b.raw_bits));
}

inline muda_fp64 operator-(muda_fp64 a) {
  return muda_fp64(a.raw_bits ^ 0x8000000000000000ULL);
}

inline bool operator==(muda_fp64 a, muda_fp64 b) {
  return muda_ieee64_eq(a.raw_bits, b.raw_bits);
}

inline bool operator!=(muda_fp64 a, muda_fp64 b) {
  return !muda_ieee64_eq(a.raw_bits, b.raw_bits);
}

inline bool operator<(muda_fp64 a, muda_fp64 b) {
  return muda_ieee64_lt(a.raw_bits, b.raw_bits);
}

inline bool operator<=(muda_fp64 a, muda_fp64 b) {
  return muda_ieee64_lt(a.raw_bits, b.raw_bits) || muda_ieee64_eq(a.raw_bits, b.raw_bits);
}

inline bool operator>(muda_fp64 a, muda_fp64 b) {
  return muda_ieee64_lt(b.raw_bits, a.raw_bits);
}

inline bool operator>=(muda_fp64 a, muda_fp64 b) {
  return muda_ieee64_lt(b.raw_bits, a.raw_bits) || muda_ieee64_eq(a.raw_bits, b.raw_bits);
}

inline muda_fp64 muda_fp64_fma(muda_fp64 a, muda_fp64 b, muda_fp64 c) {
  return (a * b) + c;
}

inline muda_fp64 muda_fp64_abs(muda_fp64 a) {
  return muda_fp64(a.raw_bits & 0x7FFFFFFFFFFFFFFFULL);
}

inline muda_fp64 muda_fp64_min(muda_fp64 a, muda_fp64 b) {
  return (a < b) ? a : b;
}

inline muda_fp64 muda_fp64_max(muda_fp64 a, muda_fp64 b) {
  return (a > b) ? a : b;
}

inline muda_fp64 muda_fp64_sqrt(muda_fp64 a) {
  return muda_fp64(muda_ieee64_sqrt(a.raw_bits));
}

inline muda_fp64 muda_fp64_rsqrt(muda_fp64 a) {
  return muda_fp64(1.0f) / muda_fp64_sqrt(a);
}

#else

// =============================================================================
// MUDA FP64: High-Throughput Double-Single (fast48) Software Emulation
// =============================================================================

struct alignas(8) muda_fp64_ds {
  float hi;
  float lo;
};

inline muda_fp64_ds muda_unpack_ieee64(ulong u) {
  if ((u & 0x7FFFFFFFFFFFFFFFULL) == 0) {
    float z = (u >> 63) ? -0.0f : 0.0f;
    return muda_fp64_ds{z, 0.0f};
  }

  float sign = (u >> 63) ? -1.0f : 1.0f;
  uint exp = static_cast<uint>((u >> 52) & 0x7FF);
  ulong frac = u & 0xFFFFFFFFFFFFFULL;

  if (exp == 0x7FF) {
    float inf = 1.0f / 0.0f;
    return muda_fp64_ds{frac == 0 ? (sign < 0.0f ? -inf : inf) : (0.0f / 0.0f), 0.0f};
  }

  ulong sig = (1ULL << 52) | frac;
  ulong sig_hi = sig >> 29;
  ulong sig_lo = (sig >> 5) & 0xFFFFFFULL;

  int unbiased_exp = static_cast<int>(exp) - 1023;
  float scale_hi = ldexp(1.0f, unbiased_exp - 23);
  float scale_lo = ldexp(1.0f, unbiased_exp - 47);

  float f_hi = static_cast<float>(sig_hi) * scale_hi * sign;
  float f_lo = static_cast<float>(sig_lo) * scale_lo * sign;
  return muda_fp64_ds{f_hi, f_lo};
}

inline ulong muda_pack_ieee64(muda_fp64_ds d) {
  if (d.hi == 0.0f && d.lo == 0.0f) {
    return signbit(d.hi) ? 0x8000000000000000ULL : 0ULL;
  }
  ulong sign = signbit(d.hi) ? 1ULL : 0ULL;
  float abs_hi = abs(d.hi);
  float abs_lo = signbit(d.hi) ? -d.lo : d.lo;

  float s = abs_hi + abs_lo;
  float v = s - abs_hi;
  float e = (abs_hi - (s - v)) + (abs_lo - v);
  abs_hi = s;
  abs_lo = e;

  if (abs_hi == 0.0f) {
    return sign << 63;
  }

  int exp = 0;
  frexp(abs_hi, exp);
  int unbiased_exp = exp - 1;
  int double_exp = unbiased_exp + 1023;
  if (double_exp <= 0) return sign << 63;
  if (double_exp >= 2047) return (sign << 63) | (0x7FFULL << 52);

  float s_hi = ldexp(abs_hi, -unbiased_exp);
  float s_lo = ldexp(abs_lo, -unbiased_exp);

  uint raw_hi = as_type<uint>(s_hi);
  ulong mant_hi = static_cast<ulong>(raw_hi & 0x7FFFFFu);
  ulong frac_hi = mant_hi << 29;
  long frac_lo = static_cast<long>(round(ldexp(s_lo, 47))) << 5;
  long frac = static_cast<long>(frac_hi) + frac_lo;

  while (frac >= (1L << 52)) {
    frac -= (1L << 52);
    double_exp += 1;
  }
  while (frac < 0 && double_exp > 0) {
    double_exp -= 1;
    frac += (1L << 52);
  }
  if (double_exp <= 0) return sign << 63;
  if (double_exp >= 2047) return (sign << 63) | (0x7FFULL << 52);

  return (sign << 63) | (static_cast<ulong>(double_exp) << 52) | (static_cast<ulong>(frac) & 0xFFFFFFFFFFFFFULL);
}

// Double-single arithmetic
inline muda_fp64_ds muda_add_ds(muda_fp64_ds a, muda_fp64_ds b) {
  float s = a.hi + b.hi;
  float v = s - a.hi;
  float e = (a.hi - (s - v)) + (b.hi - v) + (a.lo + b.lo);
  float z = s + e;
  return muda_fp64_ds{z, e - (z - s)};
}

inline muda_fp64_ds muda_sub_ds(muda_fp64_ds a, muda_fp64_ds b) {
  return muda_add_ds(a, muda_fp64_ds{-b.hi, -b.lo});
}

inline muda_fp64_ds muda_mul_ds(muda_fp64_ds a, muda_fp64_ds b) {
  float p = a.hi * b.hi;
  float e = fma(a.hi, b.hi, -p) + a.hi * b.lo + a.lo * b.hi;
  float z = p + e;
  return muda_fp64_ds{z, e - (z - p)};
}

inline muda_fp64_ds muda_div_ds(muda_fp64_ds a, muda_fp64_ds b) {
  float q1 = a.hi / b.hi;
  muda_fp64_ds prod = muda_mul_ds(b, muda_fp64_ds{q1, 0.0f});
  muda_fp64_ds rem = muda_sub_ds(a, prod);
  float q2 = rem.hi / b.hi;
  float z = q1 + q2;
  return muda_fp64_ds{z, (q1 - z) + q2};
}

inline muda_fp64_ds muda_fma_ds(muda_fp64_ds a, muda_fp64_ds b, muda_fp64_ds c) {
  return muda_add_ds(muda_mul_ds(a, b), c);
}

struct alignas(8) muda_fp64 {
  ulong raw_bits;

  muda_fp64() thread = default;
  muda_fp64() device = default;
  muda_fp64() constant = default;
  muda_fp64() threadgroup = default;

  muda_fp64(const thread muda_fp64& o) thread : raw_bits(o.raw_bits) {}
  muda_fp64(const device muda_fp64& o) thread : raw_bits(o.raw_bits) {}
  muda_fp64(const constant muda_fp64& o) thread : raw_bits(o.raw_bits) {}
  muda_fp64(const threadgroup muda_fp64& o) thread : raw_bits(o.raw_bits) {}

  muda_fp64(muda_fp64_ds ds) thread : raw_bits(muda_pack_ieee64(ds)) {}
  muda_fp64(float f) thread : raw_bits(muda_pack_ieee64(muda_fp64_ds{f, 0.0f})) {}
  muda_fp64(int i) thread : raw_bits(muda_pack_ieee64(muda_fp64_ds{static_cast<float>(i), 0.0f})) {}
  muda_fp64(uint u) thread : raw_bits(muda_pack_ieee64(muda_fp64_ds{static_cast<float>(u), 0.0f})) {}

  muda_fp64_ds to_ds() const thread { return muda_unpack_ieee64(raw_bits); }
  muda_fp64_ds to_ds() const device { return muda_unpack_ieee64(raw_bits); }
  muda_fp64_ds to_ds() const constant { return muda_unpack_ieee64(raw_bits); }
  muda_fp64_ds to_ds() const threadgroup { return muda_unpack_ieee64(raw_bits); }

  operator float() const thread { return to_ds().hi; }
  operator float() const device { return to_ds().hi; }
  operator int() const thread { return static_cast<int>(to_ds().hi); }
  operator int() const device { return static_cast<int>(to_ds().hi); }
  operator uint() const thread { return static_cast<uint>(to_ds().hi); }
  operator uint() const device { return static_cast<uint>(to_ds().hi); }

  thread muda_fp64& operator=(muda_fp64_ds ds) thread {
    raw_bits = muda_pack_ieee64(ds);
    return *this;
  }
  device muda_fp64& operator=(muda_fp64_ds ds) device {
    raw_bits = muda_pack_ieee64(ds);
    return *this;
  }
  threadgroup muda_fp64& operator=(muda_fp64_ds ds) threadgroup {
    raw_bits = muda_pack_ieee64(ds);
    return *this;
  }
};

inline muda_fp64 operator+(muda_fp64 a, muda_fp64 b) {
  return muda_fp64(muda_add_ds(a.to_ds(), b.to_ds()));
}

inline muda_fp64 operator-(muda_fp64 a, muda_fp64 b) {
  return muda_fp64(muda_sub_ds(a.to_ds(), b.to_ds()));
}

inline muda_fp64 operator*(muda_fp64 a, muda_fp64 b) {
  return muda_fp64(muda_mul_ds(a.to_ds(), b.to_ds()));
}

inline muda_fp64 operator/(muda_fp64 a, muda_fp64 b) {
  return muda_fp64(muda_div_ds(a.to_ds(), b.to_ds()));
}

inline muda_fp64 operator-(muda_fp64 a) {
  muda_fp64_ds ds = a.to_ds();
  return muda_fp64(muda_fp64_ds{-ds.hi, -ds.lo});
}

inline bool operator==(muda_fp64 a, muda_fp64 b) {
  muda_fp64_ds da = a.to_ds();
  muda_fp64_ds db = b.to_ds();
  return da.hi == db.hi && da.lo == db.lo;
}

inline bool operator!=(muda_fp64 a, muda_fp64 b) {
  return !(a == b);
}

inline bool operator<(muda_fp64 a, muda_fp64 b) {
  muda_fp64_ds da = a.to_ds();
  muda_fp64_ds db = b.to_ds();
  return (da.hi < db.hi) || (da.hi == db.hi && da.lo < db.lo);
}

inline bool operator<=(muda_fp64 a, muda_fp64 b) {
  return (a < b) || (a == b);
}

inline bool operator>(muda_fp64 a, muda_fp64 b) {
  return b < a;
}

inline bool operator>=(muda_fp64 a, muda_fp64 b) {
  return b <= a;
}

inline muda_fp64 muda_fp64_fma(muda_fp64 a, muda_fp64 b, muda_fp64 c) {
  return muda_fp64(muda_fma_ds(a.to_ds(), b.to_ds(), c.to_ds()));
}

inline muda_fp64 muda_fp64_abs(muda_fp64 a) {
  muda_fp64_ds da = a.to_ds();
  return (da.hi < 0.0f) ? -a : a;
}

inline muda_fp64 muda_fp64_min(muda_fp64 a, muda_fp64 b) {
  return (a < b) ? a : b;
}

inline muda_fp64 muda_fp64_max(muda_fp64 a, muda_fp64 b) {
  return (a > b) ? a : b;
}

inline muda_fp64 muda_fp64_sqrt(muda_fp64 a) {
  muda_fp64_ds da = a.to_ds();
  if (da.hi <= 0.0f) return muda_fp64(0.0f);
  float s = sqrt(da.hi);
  muda_fp64_ds d_s{s, 0.0f};
  d_s = muda_mul_ds(muda_add_ds(d_s, muda_div_ds(da, d_s)), muda_fp64_ds{0.5f, 0.0f});
  d_s = muda_mul_ds(muda_add_ds(d_s, muda_div_ds(da, d_s)), muda_fp64_ds{0.5f, 0.0f});
  return muda_fp64(d_s);
}

inline muda_fp64 muda_fp64_rsqrt(muda_fp64 a) {
  return muda_fp64(1.0f) / muda_fp64_sqrt(a);
}

#endif  // MUDA_FP64_IEEE

// Vector types shared across both modes
struct alignas(16) muda_fp64_2 {
  muda_fp64 x;
  muda_fp64 y;
};

struct alignas(32) muda_fp64_3 {
  muda_fp64 x;
  muda_fp64 y;
  muda_fp64 z;
};

struct alignas(32) muda_fp64_4 {
  muda_fp64 x;
  muda_fp64 y;
  muda_fp64 z;
  muda_fp64 w;
};

#endif  // MUDA_FP64_METAL_
