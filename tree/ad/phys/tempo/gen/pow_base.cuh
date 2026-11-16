// Hand override (generator rule pow_base): the shared-base binary64 pow of
// the forward (tempo_pow_base / tempo_pow_with), same primal words, partials
// of x^y in both operands.
TEMPO_HD TempoPowBase tempo_pow_base(AR<double> x) {
  TempoPowBase b; b.x = x; b.hi = 0.0; b.lo = 0.0;
  const uint32_t topx = g64_pow::top12(x.v);
  b.fast = !(topx - 0x001 >= 0x7ff - 0x001);
  if (b.fast) { double lo = 0.0; b.hi = AR<double>(g64_pow::log_inline(g64_pow::asuint64(x.v), &lo)); b.lo = AR<double>(lo); }
  return b;
}
TEMPO_HD AR<double> tempo_pow_with(const TempoPowBase& b, AR<double> y) {
  const uint32_t topy = g64_pow::top12(y.v);
  double z;
  if (!b.fast || (topy & 0x7ff) - 0x3be >= 0x43e - 0x3be) z = mpas_pow(b.x.v, y.v);
  else {
    const double ehi = __dmul_rn(y.v, b.hi.v);
    const double elo = __fma_rn(y.v, b.lo.v, __fma_rn(y.v, b.hi.v, -ehi));
    z = g64_pow::exp_inline(ehi, elo, 0);
  }
  return ad_pow_out(z, b.x, y);
}
