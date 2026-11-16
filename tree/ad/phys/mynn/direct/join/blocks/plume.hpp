#pragma once
#include "numeric.hpp"
namespace direct { namespace plume {
static constexpr float r_d = 287.f;
static constexpr float r_v = 461.6f;
static constexpr float cp = 1004.5f;
static constexpr float cpv = ((4.f) * (r_v));
static constexpr float cliq = 4190.f;
static constexpr float cice = 2106.f;
static constexpr float grav = 9.80616f;
static constexpr float karman = 0.4f;
static constexpr float p1000mb = 100000.f;
static constexpr float rcp = ((r_d) / (cp));
static constexpr float xlv = 2.50e6f;
static constexpr float xlf = 3.50e5f;
static constexpr float xls = ((xlv) + (xlf));
static constexpr float svp1 = 0.6112f;
static constexpr float svp2 = 17.67f;
static constexpr float svp3 = 29.65f;
static constexpr float p608 = ((((r_v) / (r_d))) - (1.f));
static constexpr float ep_2 = ((r_d) / (r_v));
static constexpr float rvovrd = ((r_v) / (r_d));
static constexpr float t0c = 273.15f;
static constexpr float tref = 300.0f;
static constexpr float tkmin = 253.0f;
static constexpr float tice = 238.0f;
static constexpr float ep_3 = ((1.f) - (ep_2));
static constexpr float gtr = ((grav) / (tref));
static constexpr float rk = ((cp) / (r_d));
static constexpr float tv0 = ((p608) * (tref));
static constexpr float tv1 = ((((1.f) + (p608))) * (tref));
static constexpr float xlscp = ((((xlv) + (xlf))) / (cp));
static constexpr float xlvcp = ((xlv) / (cp));
static constexpr float g_inv = ((1.f) / (grav));
static constexpr float wfa_max = 800e10f;
static constexpr float wfa_min = 1e6f;
static constexpr float ifa_max = 500e6f;
static constexpr float ifa_min = 0.0f;
static constexpr float wfa_ht = 2000.f;
static constexpr float ifa_ht = 10000.f;
static constexpr float cphm_st = 5.0f;
static constexpr float cphm_unst = 16.0f;
static constexpr float cphh_st = 5.0f;
static constexpr float cphh_unst = 16.0f;
static constexpr float pr = 0.74f;
static constexpr float g1 = 0.235f;
static constexpr float b1 = 24.0f;
static constexpr float b2 = 15.0f;
static constexpr float c2 = 0.729f;
static constexpr float c3 = 0.340f;
static constexpr float c4 = 0.0f;
static constexpr float c5 = 0.2f;
static constexpr float a1 = ((((b1) * (((1.0f) - (((3.0f) * (g1))))))) / (6.0f));
static constexpr float c1 = ((g1) - (((1.0f) / (((((3.0f) * (a1))) * (2.88449914061481660f))))));
static constexpr float a2 = ((((a1) * (((g1) - (c1))))) / (((g1) * (pr))));
static constexpr float g2 = ((((((b2) / (b1))) * (((1.0f) - (c3))))) + (((((((2.0f) * (a1))) / (b1))) * (((3.0f) - (((2.0f) * (c2))))))));
static constexpr float cc2 = ((1.0f) - (c2));
static constexpr float cc3 = ((1.0f) - (c3));
static constexpr float e1c = ((((((3.0f) * (a2))) * (b2))) * (cc3));
static constexpr float e2c = ((((((9.0f) * (a1))) * (a2))) * (cc2));
static constexpr float e3c = ((((((((9.0f) * (a2))) * (a2))) * (cc2))) * (((1.0f) - (c5))));
static constexpr float e4c = ((((((12.0f) * (a1))) * (a2))) * (cc2));
static constexpr float e5c = ((((6.0f) * (a1))) * (a1));
static constexpr float zero = 0.0f;
static constexpr float one = 1.0f;
static constexpr float two = 2.0f;
static constexpr float three = 3.0f;
static constexpr float four = 4.0f;
static constexpr float five = 5.0f;
static constexpr float six = 6.0f;
static constexpr float seven = 7.0f;
static constexpr float eight = 8.0f;
static constexpr float nine = 9.0f;
static constexpr float ten = 10.0f;
static constexpr float twenty = 20.0f;
static constexpr float thirty = 30.0f;
static constexpr float forty = 40.0f;
static constexpr float fifty = 50.0f;
static constexpr float hundred = 100.0f;
static constexpr float p01 = 0.01f;
static constexpr float p1 = 0.1f;
static constexpr float p2 = 0.2f;
static constexpr float p25 = 0.25f;
static constexpr float p3 = 0.3f;
static constexpr float p333 = ((1.0f) / (3.0f));
static constexpr float p4 = 0.4f;
static constexpr float p5 = 0.5f;
static constexpr float p6 = 0.6f;
static constexpr float p666 = ((2.0f) / (3.0f));
static constexpr float p7 = 0.7f;
static constexpr float p75 = 0.75f;
static constexpr float p8 = 0.8f;
static constexpr float p9 = 0.9f;
static constexpr float p95 = 0.95f;
static constexpr float p99 = 0.99f;
static constexpr float qmin = 0.0f;
static constexpr float zmax = 1.0f;
static constexpr float sqfac = 3.0f;
static constexpr float qkemin = 1.e-5f;
static constexpr float tliq = 269.f;
static constexpr float rr2 = 0.7071068f;
static constexpr float rrp = 0.3989423f;
static constexpr float ckmod = 1.f;
static constexpr float scaleaware = 1.f;
static constexpr float taue = 80.f;
static constexpr float upwind = 1.0f;
inline double plume_ramp(double a,double b){double m=a<b?a:b;return m<=0.5?0.0:(m>=1.5?1.0:m-0.5);}
template<class P> inline void blend_reverse(const P*x,const double*seed,double*bar,P*y=nullptr,bool planted=false){
 const P v0 = x[0];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const P v6 = x[6];
 const P v7 = x[7];
 const P v8 = x[8];
 const P v9 = x[9];
 const P v10 = x[10];
 const P v11 = x[11];
 const P v12 = x[12];
 const P v13 = x[13];
 const P v14 = x[14];
 const P v15 = x[15];
 const P v16 = x[16];
 const P v17 = x[17];
 const P v18 = x[18];
 const float v19 = float(zero);
 const bool v20 = v3 > v19;
 const P v21 = (true && v20) ? (P(v0)) : P(0);
 const P v23 = (true && v20) ? (P(v2)) : P(0);
 const P v24 = (true && v20) ? (P(v3)) : P(0);
 const P v25 = (true && v20) ? (P(v5 - v6)) : P(0);
 const P v26 = (true && v20) ? (P(v25 / v7)) : P(0);
 const P v27 = (true && v20) ? (P(v26)) : P(0);
 const float v187 = (true && v20) ? (float(t0c)) : float(0);
 const bool v188 = (true && v20) ? (v8 >= v187) : bool(0);
 const float v189 = ((true && v20) && v188) ? (float(xlv)) : float(0);
 const float v190 = ((true && v20) && v188) ? (float(cpv)) : float(0);
 const float v191 = ((true && v20) && v188) ? (float(cliq)) : float(0);
 const float v192 = ((true && v20) && v188) ? (float(v190 - v191)) : float(0);
 const float v193 = ((true && v20) && v188) ? (float(t0c)) : float(0);
 const P v194 = ((true && v20) && v188) ? (P(v8 - v193)) : P(0);
 const P v195 = ((true && v20) && v188) ? (P(v192 * v194)) : P(0);
 const P v196 = ((true && v20) && v188) ? (P(v189 + v195)) : P(0);
 const P v197 = ((true && v20) && v188) ? (P(v196)) : P(0);
 const float v198 = ((true && v20) && !v188) ? (float(tice)) : float(0);
 const bool v199 = ((true && v20) && !v188) ? (v8 <= v198) : bool(0);
 const float v200 = (((true && v20) && !v188) && v199) ? (float(xls)) : float(0);
 const float v201 = (((true && v20) && !v188) && v199) ? (float(cpv)) : float(0);
 const float v202 = (((true && v20) && !v188) && v199) ? (float(cice)) : float(0);
 const float v203 = (((true && v20) && !v188) && v199) ? (float(v201 - v202)) : float(0);
 const float v204 = (((true && v20) && !v188) && v199) ? (float(t0c)) : float(0);
 const P v205 = (((true && v20) && !v188) && v199) ? (P(v8 - v204)) : P(0);
 const P v206 = (((true && v20) && !v188) && v199) ? (P(v203 * v205)) : P(0);
 const P v207 = (((true && v20) && !v188) && v199) ? (P(v200 + v206)) : P(0);
 const P v208 = (((true && v20) && !v188) && v199) ? (P(v207)) : P(0);
 const float v209 = (((true && v20) && !v188) && !v199) ? (float(xlv)) : float(0);
 const float v210 = (((true && v20) && !v188) && !v199) ? (float(cpv)) : float(0);
 const float v211 = (((true && v20) && !v188) && !v199) ? (float(cliq)) : float(0);
 const float v212 = (((true && v20) && !v188) && !v199) ? (float(v210 - v211)) : float(0);
 const float v213 = (((true && v20) && !v188) && !v199) ? (float(t0c)) : float(0);
 const P v214 = (((true && v20) && !v188) && !v199) ? (P(v8 - v213)) : P(0);
 const P v215 = (((true && v20) && !v188) && !v199) ? (P(v212 * v214)) : P(0);
 const P v216 = (((true && v20) && !v188) && !v199) ? (P(v209 + v215)) : P(0);
 const P v217 = (((true && v20) && !v188) && !v199) ? (P(v216)) : P(0);
 const float v218 = (((true && v20) && !v188) && !v199) ? (float(xls)) : float(0);
 const float v219 = (((true && v20) && !v188) && !v199) ? (float(cpv)) : float(0);
 const float v220 = (((true && v20) && !v188) && !v199) ? (float(cice)) : float(0);
 const float v221 = (((true && v20) && !v188) && !v199) ? (float(v219 - v220)) : float(0);
 const float v222 = (((true && v20) && !v188) && !v199) ? (float(t0c)) : float(0);
 const P v223 = (((true && v20) && !v188) && !v199) ? (P(v8 - v222)) : P(0);
 const P v224 = (((true && v20) && !v188) && !v199) ? (P(v221 * v223)) : P(0);
 const P v225 = (((true && v20) && !v188) && !v199) ? (P(v218 + v224)) : P(0);
 const P v226 = (((true && v20) && !v188) && !v199) ? (P(v225)) : P(0);
 const float v227 = (((true && v20) && !v188) && !v199) ? (float(t0c)) : float(0);
 const P v228 = (((true && v20) && !v188) && !v199) ? (P(v227 - v8)) : P(0);
 const float v229 = (((true && v20) && !v188) && !v199) ? (float(t0c)) : float(0);
 const float v230 = (((true && v20) && !v188) && !v199) ? (float(tice)) : float(0);
 const float v231 = (((true && v20) && !v188) && !v199) ? (float(v229 - v230)) : float(0);
 const P v232 = (((true && v20) && !v188) && !v199) ? (P(v228 / v231)) : P(0);
 const P v233 = (((true && v20) && !v188) && !v199) ? (P(v232)) : P(0);
 const float v234 = (((true && v20) && !v188) && !v199) ? (float(1.f)) : float(0);
 const P v235 = (((true && v20) && !v188) && !v199) ? (P(v234 - v233)) : P(0);
 const P v236 = (((true && v20) && !v188) && !v199) ? (P(v235 * v217)) : P(0);
 const P v237 = (((true && v20) && !v188) && !v199) ? (P(v233 * v226)) : P(0);
 const P v238 = (((true && v20) && !v188) && !v199) ? (P(v236 + v237)) : P(0);
 const P v239 = (((true && v20) && !v188) && !v199) ? (P(v238)) : P(0);
 const P v242 = ((true && v20) && !v188) ? (v199 ? v208 : v239) : P(0);
 const P v249 = (true && v20) ? (v188 ? v197 : v242) : P(0);
 const P v254 = (true && v20) ? (P(v249)) : P(0);
 const P v255 = (true && v20) ? (P(v254)) : P(0);
 const float v256 = (true && v20) ? (float(.611583699e03f)) : float(0);
 const P v257 = (true && v20) ? (P(v256)) : P(0);
 const float v258 = (true && v20) ? (float(.444606896e02f)) : float(0);
 const P v259 = (true && v20) ? (P(v258)) : P(0);
 const float v260 = (true && v20) ? (float(.143177157e01f)) : float(0);
 const P v261 = (true && v20) ? (P(v260)) : P(0);
 const float v262 = (true && v20) ? (float(.264224321e-1f)) : float(0);
 const P v263 = (true && v20) ? (P(v262)) : P(0);
 const float v264 = (true && v20) ? (float(.299291081e-3f)) : float(0);
 const P v265 = (true && v20) ? (P(v264)) : P(0);
 const float v266 = (true && v20) ? (float(.203154182e-5f)) : float(0);
 const P v267 = (true && v20) ? (P(v266)) : P(0);
 const float v268 = (true && v20) ? (float(.702620698e-8f)) : float(0);
 const P v269 = (true && v20) ? (P(v268)) : P(0);
 const float v270 = (true && v20) ? (float(.379534310e-11f)) : float(0);
 const P v271 = (true && v20) ? (P(v270)) : P(0);
 const float v272 = (true && v20) ? (float(.321582393e-13f)) : float(0);
 const float v273 = (true && v20) ? (-v272) : float(0);
 const P v274 = (true && v20) ? (P(v273)) : P(0);
 const float v275 = (true && v20) ? (float(.609868993e03f)) : float(0);
 const P v276 = (true && v20) ? (P(v275)) : P(0);
 const float v277 = (true && v20) ? (float(.499320233e02f)) : float(0);
 const P v278 = (true && v20) ? (P(v277)) : P(0);
 const float v279 = (true && v20) ? (float(.184672631e01f)) : float(0);
 const P v280 = (true && v20) ? (P(v279)) : P(0);
 const float v281 = (true && v20) ? (float(.402737184e-1f)) : float(0);
 const P v282 = (true && v20) ? (P(v281)) : P(0);
 const float v283 = (true && v20) ? (float(.565392987e-3f)) : float(0);
 const P v284 = (true && v20) ? (P(v283)) : P(0);
 const float v285 = (true && v20) ? (float(.521693933e-5f)) : float(0);
 const P v286 = (true && v20) ? (P(v285)) : P(0);
 const float v287 = (true && v20) ? (float(.307839583e-7f)) : float(0);
 const P v288 = (true && v20) ? (P(v287)) : P(0);
 const float v289 = (true && v20) ? (float(.105785160e-9f)) : float(0);
 const P v290 = (true && v20) ? (P(v289)) : P(0);
 const float v291 = (true && v20) ? (float(.161444444e-12f)) : float(0);
 const P v292 = (true && v20) ? (P(v291)) : P(0);
 const float v293 = (true && v20) ? (float(80.f)) : float(0);
 const float v294 = (true && v20) ? (-v293) : float(0);
 const float v295 = (true && v20) ? (float(t0c)) : float(0);
 const P v296 = (true && v20) ? (P(v8 - v295)) : P(0);
 const P v297 = (true && v20) ? (v294 > v296 ? v294 : v296) : P(0);
 const P v298 = (true && v20) ? (P(v297)) : P(0);
 const float v299 = (true && v20) ? (float(t0c)) : float(0);
 const float v300 = (true && v20) ? (float(6.f)) : float(0);
 const float v301 = (true && v20) ? (float(v299 - v300)) : float(0);
 const bool v302 = (true && v20) ? (v8 >= v301) : bool(0);
 const P v303 = ((true && v20) && v302) ? (P(v298 * v274)) : P(0);
 const P v304 = ((true && v20) && v302) ? (P(v271 + v303)) : P(0);
 const P v305 = ((true && v20) && v302) ? (P(v298 * v304)) : P(0);
 const P v306 = ((true && v20) && v302) ? (P(v269 + v305)) : P(0);
 const P v307 = ((true && v20) && v302) ? (P(v298 * v306)) : P(0);
 const P v308 = ((true && v20) && v302) ? (P(v267 + v307)) : P(0);
 const P v309 = ((true && v20) && v302) ? (P(v298 * v308)) : P(0);
 const P v310 = ((true && v20) && v302) ? (P(v265 + v309)) : P(0);
 const P v311 = ((true && v20) && v302) ? (P(v298 * v310)) : P(0);
 const P v312 = ((true && v20) && v302) ? (P(v263 + v311)) : P(0);
 const P v313 = ((true && v20) && v302) ? (P(v298 * v312)) : P(0);
 const P v314 = ((true && v20) && v302) ? (P(v261 + v313)) : P(0);
 const P v315 = ((true && v20) && v302) ? (P(v298 * v314)) : P(0);
 const P v316 = ((true && v20) && v302) ? (P(v259 + v315)) : P(0);
 const P v317 = ((true && v20) && v302) ? (P(v298 * v316)) : P(0);
 const P v318 = ((true && v20) && v302) ? (P(v257 + v317)) : P(0);
 const P v319 = ((true && v20) && v302) ? (P(v318)) : P(0);
 const float v320 = ((true && v20) && v302) ? (float(0.15f)) : float(0);
 const P v321 = ((true && v20) && v302) ? (P(v9 * v320)) : P(0);
 const P v322 = ((true && v20) && v302) ? (v319 < v321 ? v319 : v321) : P(0);
 const P v323 = ((true && v20) && v302) ? (P(v322)) : P(0);
 const float v324 = ((true && v20) && v302) ? (float(0.622f)) : float(0);
 const P v325 = ((true && v20) && v302) ? (P(v324 * v323)) : P(0);
 const P v326 = ((true && v20) && v302) ? (P(v9 - v323)) : P(0);
 const float v327 = ((true && v20) && v302) ? (float(1e-5f)) : float(0);
 const P v328 = ((true && v20) && v302) ? (v326 > v327 ? v326 : v327) : P(0);
 const P v329 = ((true && v20) && v302) ? (P(v325 / v328)) : P(0);
 const P v330 = ((true && v20) && v302) ? (P(v329)) : P(0);
 const float v331 = ((true && v20) && !v302) ? (float(tice)) : float(0);
 const bool v332 = ((true && v20) && !v302) ? (v8 <= v331) : bool(0);
 const P v333 = (((true && v20) && !v302) && v332) ? (P(v298 * v292)) : P(0);
 const P v334 = (((true && v20) && !v302) && v332) ? (P(v290 + v333)) : P(0);
 const P v335 = (((true && v20) && !v302) && v332) ? (P(v298 * v334)) : P(0);
 const P v336 = (((true && v20) && !v302) && v332) ? (P(v288 + v335)) : P(0);
 const P v337 = (((true && v20) && !v302) && v332) ? (P(v298 * v336)) : P(0);
 const P v338 = (((true && v20) && !v302) && v332) ? (P(v286 + v337)) : P(0);
 const P v339 = (((true && v20) && !v302) && v332) ? (P(v298 * v338)) : P(0);
 const P v340 = (((true && v20) && !v302) && v332) ? (P(v284 + v339)) : P(0);
 const P v341 = (((true && v20) && !v302) && v332) ? (P(v298 * v340)) : P(0);
 const P v342 = (((true && v20) && !v302) && v332) ? (P(v282 + v341)) : P(0);
 const P v343 = (((true && v20) && !v302) && v332) ? (P(v298 * v342)) : P(0);
 const P v344 = (((true && v20) && !v302) && v332) ? (P(v280 + v343)) : P(0);
 const P v345 = (((true && v20) && !v302) && v332) ? (P(v298 * v344)) : P(0);
 const P v346 = (((true && v20) && !v302) && v332) ? (P(v278 + v345)) : P(0);
 const P v347 = (((true && v20) && !v302) && v332) ? (P(v298 * v346)) : P(0);
 const P v348 = (((true && v20) && !v302) && v332) ? (P(v276 + v347)) : P(0);
 const P v349 = (((true && v20) && !v302) && v332) ? (P(v348)) : P(0);
 const float v350 = (((true && v20) && !v302) && v332) ? (float(0.15f)) : float(0);
 const P v351 = (((true && v20) && !v302) && v332) ? (P(v9 * v350)) : P(0);
 const P v352 = (((true && v20) && !v302) && v332) ? (v349 < v351 ? v349 : v351) : P(0);
 const P v353 = (((true && v20) && !v302) && v332) ? (P(v352)) : P(0);
 const float v354 = (((true && v20) && !v302) && v332) ? (float(0.622f)) : float(0);
 const P v355 = (((true && v20) && !v302) && v332) ? (P(v354 * v353)) : P(0);
 const P v356 = (((true && v20) && !v302) && v332) ? (P(v9 - v353)) : P(0);
 const float v357 = (((true && v20) && !v302) && v332) ? (float(1e-5f)) : float(0);
 const P v358 = (((true && v20) && !v302) && v332) ? (v356 > v357 ? v356 : v357) : P(0);
 const P v359 = (((true && v20) && !v302) && v332) ? (P(v355 / v358)) : P(0);
 const P v360 = (((true && v20) && !v302) && v332) ? (P(v359)) : P(0);
 const P v361 = (((true && v20) && !v302) && !v332) ? (P(v298 * v274)) : P(0);
 const P v362 = (((true && v20) && !v302) && !v332) ? (P(v271 + v361)) : P(0);
 const P v363 = (((true && v20) && !v302) && !v332) ? (P(v298 * v362)) : P(0);
 const P v364 = (((true && v20) && !v302) && !v332) ? (P(v269 + v363)) : P(0);
 const P v365 = (((true && v20) && !v302) && !v332) ? (P(v298 * v364)) : P(0);
 const P v366 = (((true && v20) && !v302) && !v332) ? (P(v267 + v365)) : P(0);
 const P v367 = (((true && v20) && !v302) && !v332) ? (P(v298 * v366)) : P(0);
 const P v368 = (((true && v20) && !v302) && !v332) ? (P(v265 + v367)) : P(0);
 const P v369 = (((true && v20) && !v302) && !v332) ? (P(v298 * v368)) : P(0);
 const P v370 = (((true && v20) && !v302) && !v332) ? (P(v263 + v369)) : P(0);
 const P v371 = (((true && v20) && !v302) && !v332) ? (P(v298 * v370)) : P(0);
 const P v372 = (((true && v20) && !v302) && !v332) ? (P(v261 + v371)) : P(0);
 const P v373 = (((true && v20) && !v302) && !v332) ? (P(v298 * v372)) : P(0);
 const P v374 = (((true && v20) && !v302) && !v332) ? (P(v259 + v373)) : P(0);
 const P v375 = (((true && v20) && !v302) && !v332) ? (P(v298 * v374)) : P(0);
 const P v376 = (((true && v20) && !v302) && !v332) ? (P(v257 + v375)) : P(0);
 const P v377 = (((true && v20) && !v302) && !v332) ? (P(v376)) : P(0);
 const float v378 = (((true && v20) && !v302) && !v332) ? (float(0.15f)) : float(0);
 const P v379 = (((true && v20) && !v302) && !v332) ? (P(v9 * v378)) : P(0);
 const P v380 = (((true && v20) && !v302) && !v332) ? (v377 < v379 ? v377 : v379) : P(0);
 const P v381 = (((true && v20) && !v302) && !v332) ? (P(v380)) : P(0);
 const P v382 = (((true && v20) && !v302) && !v332) ? (P(v298 * v292)) : P(0);
 const P v383 = (((true && v20) && !v302) && !v332) ? (P(v290 + v382)) : P(0);
 const P v384 = (((true && v20) && !v302) && !v332) ? (P(v298 * v383)) : P(0);
 const P v385 = (((true && v20) && !v302) && !v332) ? (P(v288 + v384)) : P(0);
 const P v386 = (((true && v20) && !v302) && !v332) ? (P(v298 * v385)) : P(0);
 const P v387 = (((true && v20) && !v302) && !v332) ? (P(v286 + v386)) : P(0);
 const P v388 = (((true && v20) && !v302) && !v332) ? (P(v298 * v387)) : P(0);
 const P v389 = (((true && v20) && !v302) && !v332) ? (P(v284 + v388)) : P(0);
 const P v390 = (((true && v20) && !v302) && !v332) ? (P(v298 * v389)) : P(0);
 const P v391 = (((true && v20) && !v302) && !v332) ? (P(v282 + v390)) : P(0);
 const P v392 = (((true && v20) && !v302) && !v332) ? (P(v298 * v391)) : P(0);
 const P v393 = (((true && v20) && !v302) && !v332) ? (P(v280 + v392)) : P(0);
 const P v394 = (((true && v20) && !v302) && !v332) ? (P(v298 * v393)) : P(0);
 const P v395 = (((true && v20) && !v302) && !v332) ? (P(v278 + v394)) : P(0);
 const P v396 = (((true && v20) && !v302) && !v332) ? (P(v298 * v395)) : P(0);
 const P v397 = (((true && v20) && !v302) && !v332) ? (P(v276 + v396)) : P(0);
 const P v398 = (((true && v20) && !v302) && !v332) ? (P(v397)) : P(0);
 const float v399 = (((true && v20) && !v302) && !v332) ? (float(0.15f)) : float(0);
 const P v400 = (((true && v20) && !v302) && !v332) ? (P(v9 * v399)) : P(0);
 const P v401 = (((true && v20) && !v302) && !v332) ? (v398 < v400 ? v398 : v400) : P(0);
 const P v402 = (((true && v20) && !v302) && !v332) ? (P(v401)) : P(0);
 const float v403 = (((true && v20) && !v302) && !v332) ? (float(0.622f)) : float(0);
 const P v404 = (((true && v20) && !v302) && !v332) ? (P(v403 * v381)) : P(0);
 const P v405 = (((true && v20) && !v302) && !v332) ? (P(v9 - v381)) : P(0);
 const float v406 = (((true && v20) && !v302) && !v332) ? (float(1e-5f)) : float(0);
 const P v407 = (((true && v20) && !v302) && !v332) ? (v405 > v406 ? v405 : v406) : P(0);
 const P v408 = (((true && v20) && !v302) && !v332) ? (P(v404 / v407)) : P(0);
 const P v409 = (((true && v20) && !v302) && !v332) ? (P(v408)) : P(0);
 const float v410 = (((true && v20) && !v302) && !v332) ? (float(0.622f)) : float(0);
 const P v411 = (((true && v20) && !v302) && !v332) ? (P(v410 * v402)) : P(0);
 const P v412 = (((true && v20) && !v302) && !v332) ? (P(v9 - v402)) : P(0);
 const float v413 = (((true && v20) && !v302) && !v332) ? (float(1e-5f)) : float(0);
 const P v414 = (((true && v20) && !v302) && !v332) ? (v412 > v413 ? v412 : v413) : P(0);
 const P v415 = (((true && v20) && !v302) && !v332) ? (P(v411 / v414)) : P(0);
 const P v416 = (((true && v20) && !v302) && !v332) ? (P(v415)) : P(0);
 const float v417 = (((true && v20) && !v302) && !v332) ? (float(t0c)) : float(0);
 const float v418 = (((true && v20) && !v302) && !v332) ? (float(6.f)) : float(0);
 const float v419 = (((true && v20) && !v302) && !v332) ? (float(v417 - v418)) : float(0);
 const P v420 = (((true && v20) && !v302) && !v332) ? (P(v419 - v8)) : P(0);
 const float v421 = (((true && v20) && !v302) && !v332) ? (float(t0c)) : float(0);
 const float v422 = (((true && v20) && !v302) && !v332) ? (float(6.f)) : float(0);
 const float v423 = (((true && v20) && !v302) && !v332) ? (float(v421 - v422)) : float(0);
 const float v424 = (((true && v20) && !v302) && !v332) ? (float(tice)) : float(0);
 const float v425 = (((true && v20) && !v302) && !v332) ? (float(v423 - v424)) : float(0);
 const P v426 = (((true && v20) && !v302) && !v332) ? (P(v420 / v425)) : P(0);
 const P v427 = (((true && v20) && !v302) && !v332) ? (P(v426)) : P(0);
 const float v428 = (((true && v20) && !v302) && !v332) ? (float(1.f)) : float(0);
 const P v429 = (((true && v20) && !v302) && !v332) ? (P(v428 - v427)) : P(0);
 const P v430 = (((true && v20) && !v302) && !v332) ? (P(v429 * v409)) : P(0);
 const P v431 = (((true && v20) && !v302) && !v332) ? (P(v427 * v416)) : P(0);
 const P v432 = (((true && v20) && !v302) && !v332) ? (P(v430 + v431)) : P(0);
 const P v433 = (((true && v20) && !v302) && !v332) ? (P(v432)) : P(0);
 const P v439 = ((true && v20) && !v302) ? (v332 ? v360 : v433) : P(0);
 const P v449 = (true && v20) ? (v302 ? v330 : v439) : P(0);
 const P v454 = (true && v20) ? (P(v449)) : P(0);
 const P v455 = (true && v20) ? (P(v454)) : P(0);
 const P v456 = (true && v20) ? (P(v255 * v455)) : P(0);
 const float v457 = (true && v20) ? (float(r_v)) : float(0);
 const P v458 = (true && v20) ? (P(v8*v8)) : P(0);
 const P v459 = (true && v20) ? (P(v457 * v458)) : P(0);
 const P v460 = (true && v20) ? (P(v456 / v459)) : P(0);
 const P v461 = (true && v20) ? (P(v460)) : P(0);
 const float v462 = (true && v20) ? (float(cp)) : float(0);
 const float v463 = (true && v20) ? (float(cpv)) : float(0);
 const P v464 = (true && v20) ? (P(v10 * v463)) : P(0);
 const P v465 = (true && v20) ? (P(v462 + v464)) : P(0);
 const P v466 = (true && v20) ? (P(v465)) : P(0);
 const float v467 = (true && v20) ? (float(one)) : float(0);
 const float v468 = (true && v20) ? (float(one)) : float(0);
 const P v469 = (true && v20) ? (P(v255 * v461)) : P(0);
 const P v470 = (true && v20) ? (P(v469 / v466)) : P(0);
 const P v471 = (true && v20) ? (P(v468 + v470)) : P(0);
 const P v472 = (true && v20) ? (P(v467 / v471)) : P(0);
 const P v473 = (true && v20) ? (P(v472)) : P(0);
 const P v474 = (true && v20) ? (P(v473 * v461)) : P(0);
 const P v475 = (true && v20) ? (P(v474)) : P(0);
 const float v476 = (true && v20) ? (float(xlvcp)) : float(0);
 const P v477 = (true && v20) ? (P(v476 / v11)) : P(0);
 const P v478 = (true && v20) ? (P(v477)) : P(0);
 const P v479 = (true && v20) ? (P(v478 * v24)) : P(0);
 const P v480 = (true && v20) ? (P(v479 * v21)) : P(0);
 const P v481 = (true && v20) ? (P(v12 + v480)) : P(0);
 const P v482 = (true && v20) ? (P(v481)) : P(0);
 const P v483 = (true && v20) ? (P(v475 * v8)) : P(0);
 const P v484 = (true && v20) ? (P(v483 / v482)) : P(0);
 const P v485 = (true && v20) ? (P(v484)) : P(0);
 const float v486 = (true && v20) ? (float(one)) : float(0);
 const float v487 = (true && v20) ? (float(p608)) : float(0);
 const P v488 = (true && v20) ? (P(v487 * v10)) : P(0);
 const P v489 = (true && v20) ? (P(v486 + v488)) : P(0);
 const P v490 = (true && v20) ? (P(v489)) : P(0);
 const float v491 = (true && v20) ? (float(p608)) : float(0);
 const P v492 = (true && v20) ? (P(v491 * v482)) : P(0);
 const P v493 = (true && v20) ? (P(v492)) : P(0);
 const P v494 = (true && v20) ? (P(v482 * v255)) : P(0);
 const float v495 = (true && v20) ? (float(cp)) : float(0);
 const P v496 = (true && v20) ? (P(v8 * v495)) : P(0);
 const P v497 = (true && v20) ? (P(v494 / v496)) : P(0);
 const float v498 = (true && v20) ? (float(1.61f)) : float(0);
 const P v499 = (true && v20) ? (P(v498 * v482)) : P(0);
 const P v500 = (true && v20) ? (P(v497 - v499)) : P(0);
 const P v501 = (true && v20) ? (P(v500)) : P(0);
 const float v512 = (true && v20) ? (float(1800.f)) : float(0);
 const P v513 = (true && v20) ? (P(v512)) : P(0);
 const float v514 = (true && v20) ? (float(five)) : float(0);
 const P v515 = (true && v20) ? (P(v514 * v21)) : P(0);
 const P v516 = (true && v20) ? (P(v515 * v4)) : P(0);
 const float v517 = (true && v20) ? (float(zero)) : float(0);
 const P v518 = (true && v20) ? (v517 > v27 ? v517 : v27) : P(0);
 const P v519 = (true && v20) ? (P(v513 * v518)) : P(0);
 const float v520 = (true && v20) ? (float(4e-3f)) : float(0);
 const P v521 = (true && v20) ? (v519 > v520 ? v519 : v520) : P(0);
 const P v522 = (true && v20) ? (P(v516 * v521)) : P(0);
 const P v523 = (true && v20) ? (P(v522)) : P(0);
 const float v524 = (true && v20) ? (float(three)) : float(0);
 const P v525 = (true && v20) ? (P(v524 * v21)) : P(0);
 const P v526 = (true && v20) ? (P(v23 - v10)) : P(0);
 const P v527 = (true && v20) ? (P(v525 * v526)) : P(0);
 const P v528 = (true && v20) ? (v523 > v527 ? v523 : v527) : P(0);
 const P v529 = (true && v20) ? (P(v528)) : P(0);
 const P v530 = (true && v20) ? (P(v10 - v455)) : P(0);
 const P v531 = (true && v20) ? (P(v530)) : P(0);
 const float v532 = (true && v20) ? (float(zero)) : float(0);
 const bool v533 = (true && v20) ? (v531 > v532) : bool(0);
 const float v534 = ((true && v20) && v533) ? (float(0.015f)) : float(0);
 const P v535 = ((true && v20) && v533) ? (P(v455 * v534)) : P(0);
 const P v536 = ((true && v20) && v533) ? (v535 < v529 ? v535 : v529) : P(0);
 const P v537 = ((true && v20) && v533) ? (P(v536)) : P(0);
 const P v538 = (true && v20) ? (v533 ? v537 : v529) : P(0);
 const P v539 = (true && v20) ? (P(v531 / v538)) : P(0);
 const P v540 = (true && v20) ? (P(v539)) : P(0);
 const float v541 = (true && v20) ? (float(1.5f)) : float(0);
 const P v542 = (true && v20) ? (P(v13 - v541)) : P(0);
 const float v543 = (true && v20) ? (float(zero)) : float(0);
 const bool v544 = (true && v20) ? (v542 >= v543) : bool(0);
 const float v545 = ((true && v20) && v544) ? (float(0.05f)) : float(0);
 const P v546 = ((true && v20) && v544) ? (P(v14 - v545)) : P(0);
 const float v547 = ((true && v20) && v544) ? (float(zero)) : float(0);
 const P v548 = ((true && v20) && v544) ? (v546 > v547 ? v546 : v547) : P(0);
 const float v549 = ((true && v20) && v544) ? (float(0.05f)) : float(0);
 const P v550 = ((true && v20) && v544) ? (P(v548 / v549)) : P(0);
 const float v551 = ((true && v20) && v544) ? (float(one)) : float(0);
 const P v552 = ((true && v20) && v544) ? (v550 < v551 ? v550 : v551) : P(0);
 const P v553 = ((true && v20) && v544) ? (P(v552)) : P(0);
 const float v554 = ((true && v20) && v544) ? (float(p5)) : float(0);
 const float v555 = ((true && v20) && v544) ? (float(0.36f)) : float(0);
 const float v556 = ((true && v20) && v544) ? (float(1.8f)) : float(0);
 const float v557 = ((true && v20) && v544) ? (float(p2)) : float(0);
 const P v558 = ((true && v20) && v544) ? (P(v540 + v557)) : P(0);
 const P v559 = ((true && v20) && v544) ? (P(v558 + v553)) : P(0);
 const P v560 = ((true && v20) && v544) ? (P(v556 * v559)) : P(0);
 const P v561 = ((true && v20) && v544) ? (P(std::atan(v560))) : P(0);
 const P v562 = ((true && v20) && v544) ? (P(v555 * v561)) : P(0);
 const P v563 = ((true && v20) && v544) ? (P(v554 + v562)) : P(0);
 const float v564 = ((true && v20) && v544) ? (float(0.01f)) : float(0);
 const P v565 = ((true && v20) && v544) ? (v563 > v564 ? v563 : v564) : P(0);
 const float v566 = ((true && v20) && v544) ? (float(one)) : float(0);
 const P v567 = ((true && v20) && v544) ? (v565 < v566 ? v565 : v566) : P(0);
 const P v568 = ((true && v20) && v544) ? (P(v567)) : P(0);
 const float v569 = ((true && v20) && v544) ? (float(1.2f)) : float(0);
 const P v570 = ((true && v20) && v544) ? (P(v569 * v21)) : P(0);
 const P v571 = ((true && v20) && v544) ? (v568 > v570 ? v568 : v570) : P(0);
 const P v572 = ((true && v20) && v544) ? (P(v571)) : P(0);
 const float v573 = ((true && v20) && !v544) ? (float(p5)) : float(0);
 const float v574 = ((true && v20) && !v544) ? (float(0.36f)) : float(0);
 const float v575 = ((true && v20) && !v544) ? (float(1.8f)) : float(0);
 const float v576 = ((true && v20) && !v544) ? (float(p2)) : float(0);
 const P v577 = ((true && v20) && !v544) ? (P(v540 + v576)) : P(0);
 const P v578 = ((true && v20) && !v544) ? (P(v575 * v577)) : P(0);
 const P v579 = ((true && v20) && !v544) ? (P(std::atan(v578))) : P(0);
 const P v580 = ((true && v20) && !v544) ? (P(v574 * v579)) : P(0);
 const P v581 = ((true && v20) && !v544) ? (P(v573 + v580)) : P(0);
 const float v582 = ((true && v20) && !v544) ? (float(0.01f)) : float(0);
 const P v583 = ((true && v20) && !v544) ? (v581 > v582 ? v581 : v582) : P(0);
 const float v584 = ((true && v20) && !v544) ? (float(one)) : float(0);
 const P v585 = ((true && v20) && !v544) ? (v583 < v584 ? v583 : v584) : P(0);
 const P v586 = ((true && v20) && !v544) ? (P(v585)) : P(0);
 const float v587 = ((true && v20) && !v544) ? (float(1.8f)) : float(0);
 const P v588 = ((true && v20) && !v544) ? (P(v587 * v21)) : P(0);
 const P v589 = ((true && v20) && !v544) ? (v586 > v588 ? v586 : v588) : P(0);
 const P v590 = ((true && v20) && !v544) ? (P(v589)) : P(0);
 const P v591 = (true && v20) ? (v544 ? v572 : v590) : P(0);
 const float v594 = (true && v20) ? (float(1.20f)) : float(0);
 const float v595 = (true && v20) ? (float(1.0f)) : float(0);
 const float v596 = (true && v20) ? (float(0.04f)) : float(0);
 const P v597 = (true && v20) ? (P(v14 - v596)) : P(0);
 const float v598 = (true && v20) ? (float(zero)) : float(0);
 const P v599 = (true && v20) ? (v597 > v598 ? v597 : v598) : P(0);
 const float v600 = (true && v20) ? (float(0.05f)) : float(0);
 const P v601 = (true && v20) ? (P(v599 / v600)) : P(0);
 const float v602 = (true && v20) ? (float(one)) : float(0);
 const P v603 = (true && v20) ? (v601 < v602 ? v601 : v602) : P(0);
 const P v604 = (true && v20) ? (P(v595 * v603)) : P(0);
 const P v605 = (true && v20) ? (P(v594 + v604)) : P(0);
 const P v606 = (true && v20) ? (P(v605)) : P(0);
 const float v607 = (true && v20) ? (float(1.5f)) : float(0);
 const P v608 = (true && v20) ? (P(v13 - v607)) : P(0);
 const float v609 = (true && v20) ? (float(zero)) : float(0);
 const bool v610 = (true && v20) ? (v608 >= v609) : bool(0);
 const P v611 = ((true && v20) && v610) ? (P(v24 * v21)) : P(0);
 const P v612 = ((true && v20) && v610) ? (P(v606 * v611)) : P(0);
 const P v613 = ((true && v20) && v610) ? (P(v612)) : P(0);
 const P v614 = ((true && v20) && !v610) ? (P(v24 * v21)) : P(0);
 const float v615 = ((true && v20) && !v610) ? (float(5e-5f)) : float(0);
 const bool v616 = ((true && v20) && !v610) ? (v614 > v615) : bool(0);
 const float v617 = (((true && v20) && !v610) && v616) ? (float(1.86f)) : float(0);
 const P v618 = (((true && v20) && !v610) && v616) ? (P(v24 * v21)) : P(0);
 const P v619 = (((true && v20) && !v610) && v616) ? (P(v617 * v618)) : P(0);
 const float v620 = (((true && v20) && !v610) && v616) ? (float(2.2e-5f)) : float(0);
 const P v621 = (((true && v20) && !v610) && v616) ? (P(v619 - v620)) : P(0);
 const P v622 = (((true && v20) && !v610) && v616) ? (P(v621)) : P(0);
 const float v623 = (((true && v20) && !v610) && !v616) ? (float(1.20f)) : float(0);
 const P v624 = (((true && v20) && !v610) && !v616) ? (P(v24 * v21)) : P(0);
 const P v625 = (((true && v20) && !v610) && !v616) ? (P(v623 * v624)) : P(0);
 const P v626 = (((true && v20) && !v610) && !v616) ? (P(v625)) : P(0);
 const P v627 = ((true && v20) && !v610) ? (v616 ? v622 : v626) : P(0);
 const P v628 = (true && v20) ? (v610 ? v613 : v627) : P(0);
 const float v629 = (true && v20) ? (float(one)) : float(0);
 const float v630 = (true && v20) ? (float(thirty)) : float(0);
 const P v631 = (true && v20) ? (P(v591 - v630)) : P(0);
 const float v632 = (true && v20) ? (float(zero)) : float(0);
 const P v633 = (true && v20) ? (v631 > v632 ? v631 : v632) : P(0);
 const float v634 = (true && v20) ? (float(hundred)) : float(0);
 const float v635 = (true && v20) ? (float(thirty)) : float(0);
 const float v636 = (true && v20) ? (float(v634 - v635)) : float(0);
 const P v637 = (true && v20) ? (P(v633 / v636)) : P(0);
 const float v638 = (true && v20) ? (float(one)) : float(0);
 const P v639 = (true && v20) ? (v637 < v638 ? v637 : v638) : P(0);
 const P v640 = (true && v20) ? (P(v629 - v639)) : P(0);
 const P v641 = (true && v20) ? (P(v640)) : P(0);
 const P v642 = (true && v20) ? (P(v641 * v455)) : P(0);
 const float v643 = (true && v20) ? (float(0.01f)) : float(0);
 const P v644 = (true && v20) ? (P(v642 * v643)) : P(0);
 const P v645 = (true && v20) ? (P(v644 * v591)) : P(0);
 const float v646 = (true && v20) ? (float(one)) : float(0);
 const P v647 = (true && v20) ? (P(v646 - v641)) : P(0);
 const P v648 = (true && v20) ? (P(v647 * v455)) : P(0);
 const float v649 = (true && v20) ? (float(0.025f)) : float(0);
 const P v650 = (true && v20) ? (P(v648 * v649)) : P(0);
 const P v651 = (true && v20) ? (P(v650 * v591)) : P(0);
 const P v652 = (true && v20) ? (P(v645 + v651)) : P(0);
 const P v653 = (true && v20) ? (P(v652)) : P(0);
 const P v654 = (true && v20) ? (v628 > v653 ? v628 : v653) : P(0);
 const P v655 = (true && v20) ? (P(v654)) : P(0);
 const P v656 = (true && v20) ? (P(v15)) : P(0);
 const P v657 = (true && v20) ? (P(v16)) : P(0);
 const P v658 = (true && v20) ? (v591 > v656 ? v591 : v656) : P(0);
 const P v659 = (true && v20) ? (P(v658)) : P(0);
 const P v660 = (true && v20) ? (P(v591 / v659)) : P(0);
 const P v661 = (true && v20) ? (P(v660)) : P(0);
 const P v662 = (true && v20) ? (P(v655 * v661)) : P(0);
 const float v663 = (true && v20) ? (float(one)) : float(0);
 const P v664 = (true && v20) ? (P(v663 - v661)) : P(0);
 const P v665 = (true && v20) ? (P(v664 * v657)) : P(0);
 const P v666 = (true && v20) ? (P(v662 + v665)) : P(0);
 const P v667 = (true && v20) ? (P(v666)) : P(0);
 const float v668 = (true && v20) ? (float(2.25f)) : float(0);
 const float v669 = (true && v20) ? (-v668) : float(0);
 const P v670 = (true && v20) ? (v540 > v669 ? v540 : v669) : P(0);
 const P v671 = (true && v20) ? (P(v670)) : P(0);
 const float v672 = (true && v20) ? (float(one)) : float(0);
 const bool v673 = (true && v20) ? (v671 >= v672) : bool(0);
 const float v674 = ((true && v20) && v673) ? (float(one)) : float(0);
 const P v675 = ((true && v20) && v673) ? (P(v674)) : P(0);
 const float v676 = ((true && v20) && !v673) ? (float(1.7f)) : float(0);
 const float v677 = ((true && v20) && !v673) ? (-v676) : float(0);
 const bool v678 = ((true && v20) && !v673) ? (v671 >= v677) : bool(0);
 const float v679 = ((true && v20) && !v673) ? (float(one)) : float(0);
 const bool v680 = ((true && v20) && !v673) ? (v671 < v679) : bool(0);
 const bool v681 = ((true && v20) && !v673) ? (v678 && v680) : bool(0);
 const float v682 = (((true && v20) && !v673) && v681) ? (float(p4)) : float(0);
 const float v683 = (((true && v20) && !v673) && v681) ? (float(one)) : float(0);
 const P v684 = (((true && v20) && !v673) && v681) ? (P(v671 - v683)) : P(0);
 const P v685 = (((true && v20) && !v673) && v681) ? (P(v682 * v684)) : P(0);
 const P v686 = (((true && v20) && !v673) && v681) ? (-v685) : P(0);
 const P v687 = (((true && v20) && !v673) && v681) ? (P(std::exp(v686))) : P(0);
 const P v688 = (((true && v20) && !v673) && v681) ? (P(v687)) : P(0);
 const float v689 = (((true && v20) && !v673) && !v681) ? (float(2.5f)) : float(0);
 const float v690 = (((true && v20) && !v673) && !v681) ? (-v689) : float(0);
 const bool v691 = (((true && v20) && !v673) && !v681) ? (v671 >= v690) : bool(0);
 const float v692 = (((true && v20) && !v673) && !v681) ? (float(1.7f)) : float(0);
 const float v693 = (((true && v20) && !v673) && !v681) ? (-v692) : float(0);
 const bool v694 = (((true && v20) && !v673) && !v681) ? (v671 < v693) : bool(0);
 const bool v695 = (((true && v20) && !v673) && !v681) ? (v691 && v694) : bool(0);
 const float v696 = ((((true && v20) && !v673) && !v681) && v695) ? (float(three)) : float(0);
 const float v697 = ((((true && v20) && !v673) && !v681) && v695) ? (float(3.8f)) : float(0);
 const float v698 = ((((true && v20) && !v673) && !v681) && v695) ? (float(1.7f)) : float(0);
 const P v699 = ((((true && v20) && !v673) && !v681) && v695) ? (P(v671 + v698)) : P(0);
 const P v700 = ((((true && v20) && !v673) && !v681) && v695) ? (P(v697 * v699)) : P(0);
 const P v701 = ((((true && v20) && !v673) && !v681) && v695) ? (-v700) : P(0);
 const P v702 = ((((true && v20) && !v673) && !v681) && v695) ? (P(std::exp(v701))) : P(0);
 const P v703 = ((((true && v20) && !v673) && !v681) && v695) ? (P(v696 + v702)) : P(0);
 const P v704 = ((((true && v20) && !v673) && !v681) && v695) ? (P(v703)) : P(0);
 const float v705 = ((((true && v20) && !v673) && !v681) && !v695) ? (float(23.9f)) : float(0);
 const float v706 = ((((true && v20) && !v673) && !v681) && !v695) ? (float(1.6f)) : float(0);
 const float v707 = ((((true && v20) && !v673) && !v681) && !v695) ? (float(2.5f)) : float(0);
 const P v708 = ((((true && v20) && !v673) && !v681) && !v695) ? (P(v671 + v707)) : P(0);
 const P v709 = ((((true && v20) && !v673) && !v681) && !v695) ? (P(v706 * v708)) : P(0);
 const P v710 = ((((true && v20) && !v673) && !v681) && !v695) ? (-v709) : P(0);
 const P v711 = ((((true && v20) && !v673) && !v681) && !v695) ? (P(std::exp(v710))) : P(0);
 const P v712 = ((((true && v20) && !v673) && !v681) && !v695) ? (P(v705 + v711)) : P(0);
 const float v713 = ((((true && v20) && !v673) && !v681) && !v695) ? (float(60.f)) : float(0);
 const P v714 = ((((true && v20) && !v673) && !v681) && !v695) ? (v712 < v713 ? v712 : v713) : P(0);
 const P v715 = ((((true && v20) && !v673) && !v681) && !v695) ? (P(v714)) : P(0);
 const P v716 = (((true && v20) && !v673) && !v681) ? (v695 ? v704 : v715) : P(0);
 const P v717 = ((true && v20) && !v673) ? (v681 ? v688 : v716) : P(0);
 const P v718 = (true && v20) ? (v673 ? v675 : v717) : P(0);
 const float v719 = (true && v20) ? (float(1.5f)) : float(0);
 const P v720 = (true && v20) ? (P(v719 * v21)) : P(0);
 const P v721 = (true && v20) ? (P(v720 * v501)) : P(0);
 const P v722 = (true && v20) ? (P(v721 * v485)) : P(0);
 const P v723 = (true && v20) ? (P(v722 * v718)) : P(0);
 const P v724 = (true && v20) ? (P(v490 - v723)) : P(0);
 const float v725 = (true && v20) ? (float(one)) : float(0);
 const P v726 = (true && v20) ? (P(v724 - v725)) : P(0);
 const P v727 = (true && v20) ? (P(v726)) : P(0);
 const float v728 = (true && v20) ? (float(1.5f)) : float(0);
 const P v729 = (true && v20) ? (P(v728 * v21)) : P(0);
 const P v730 = (true && v20) ? (P(v729 * v501)) : P(0);
 const P v731 = (true && v20) ? (P(v730 * v473)) : P(0);
 const P v732 = (true && v20) ? (P(v731 * v718)) : P(0);
 const P v733 = (true && v20) ? (P(v493 + v732)) : P(0);
 const float v734 = (true && v20) ? (float(tv0)) : float(0);
 const P v735 = (true && v20) ? (P(v733 - v734)) : P(0);
 const P v736 = (true && v20) ? (P(v735)) : P(0);
 const P v755 = v20 ? v659 : v15;
 const P v774 = v20 ? v667 : v16;
 const P v803 = v20 ? v736 : v18;
 const P v804 = v20 ? v727 : v17;
 double b0=0;
 double b2=0;
 double b3=0;
 double b4=0;
 double b5=0;
 double b6=0;
 double b7=0;
 double b8=0;
 double b9=0;
 double b10=0;
 double b11=0;
 double b12=0;
 double b14=0;
 double b15=0;
 double b16=0;
 double b17=0;
 double b18=0;
 double b21=0;
 double b23=0;
 double b24=0;
 double b25=0;
 double b26=0;
 double b27=0;
 double b192=0;
 double b194=0;
 double b195=0;
 double b196=0;
 double b197=0;
 double b203=0;
 double b205=0;
 double b206=0;
 double b207=0;
 double b208=0;
 double b212=0;
 double b214=0;
 double b215=0;
 double b216=0;
 double b217=0;
 double b221=0;
 double b223=0;
 double b224=0;
 double b225=0;
 double b226=0;
 double b228=0;
 double b231=0;
 double b232=0;
 double b233=0;
 double b235=0;
 double b236=0;
 double b237=0;
 double b238=0;
 double b239=0;
 double b242=0;
 double b249=0;
 double b254=0;
 double b255=0;
 double b257=0;
 double b259=0;
 double b261=0;
 double b263=0;
 double b265=0;
 double b267=0;
 double b269=0;
 double b271=0;
 double b273=0;
 double b274=0;
 double b276=0;
 double b278=0;
 double b280=0;
 double b282=0;
 double b284=0;
 double b286=0;
 double b288=0;
 double b290=0;
 double b292=0;
 double b294=0;
 double b296=0;
 double b297=0;
 double b298=0;
 double b303=0;
 double b304=0;
 double b305=0;
 double b306=0;
 double b307=0;
 double b308=0;
 double b309=0;
 double b310=0;
 double b311=0;
 double b312=0;
 double b313=0;
 double b314=0;
 double b315=0;
 double b316=0;
 double b317=0;
 double b318=0;
 double b319=0;
 double b321=0;
 double b322=0;
 double b323=0;
 double b325=0;
 double b326=0;
 double b328=0;
 double b329=0;
 double b330=0;
 double b333=0;
 double b334=0;
 double b335=0;
 double b336=0;
 double b337=0;
 double b338=0;
 double b339=0;
 double b340=0;
 double b341=0;
 double b342=0;
 double b343=0;
 double b344=0;
 double b345=0;
 double b346=0;
 double b347=0;
 double b348=0;
 double b349=0;
 double b351=0;
 double b352=0;
 double b353=0;
 double b355=0;
 double b356=0;
 double b358=0;
 double b359=0;
 double b360=0;
 double b361=0;
 double b362=0;
 double b363=0;
 double b364=0;
 double b365=0;
 double b366=0;
 double b367=0;
 double b368=0;
 double b369=0;
 double b370=0;
 double b371=0;
 double b372=0;
 double b373=0;
 double b374=0;
 double b375=0;
 double b376=0;
 double b377=0;
 double b379=0;
 double b380=0;
 double b381=0;
 double b382=0;
 double b383=0;
 double b384=0;
 double b385=0;
 double b386=0;
 double b387=0;
 double b388=0;
 double b389=0;
 double b390=0;
 double b391=0;
 double b392=0;
 double b393=0;
 double b394=0;
 double b395=0;
 double b396=0;
 double b397=0;
 double b398=0;
 double b400=0;
 double b401=0;
 double b402=0;
 double b404=0;
 double b405=0;
 double b407=0;
 double b408=0;
 double b409=0;
 double b411=0;
 double b412=0;
 double b414=0;
 double b415=0;
 double b416=0;
 double b419=0;
 double b420=0;
 double b423=0;
 double b425=0;
 double b426=0;
 double b427=0;
 double b429=0;
 double b430=0;
 double b431=0;
 double b432=0;
 double b433=0;
 double b439=0;
 double b449=0;
 double b454=0;
 double b455=0;
 double b456=0;
 double b458=0;
 double b459=0;
 double b460=0;
 double b461=0;
 double b464=0;
 double b465=0;
 double b466=0;
 double b469=0;
 double b470=0;
 double b471=0;
 double b472=0;
 double b473=0;
 double b474=0;
 double b475=0;
 double b477=0;
 double b478=0;
 double b479=0;
 double b480=0;
 double b481=0;
 double b482=0;
 double b483=0;
 double b484=0;
 double b485=0;
 double b488=0;
 double b489=0;
 double b490=0;
 double b492=0;
 double b493=0;
 double b494=0;
 double b496=0;
 double b497=0;
 double b499=0;
 double b500=0;
 double b501=0;
 double b513=0;
 double b515=0;
 double b516=0;
 double b518=0;
 double b519=0;
 double b521=0;
 double b522=0;
 double b523=0;
 double b525=0;
 double b526=0;
 double b527=0;
 double b528=0;
 double b529=0;
 double b530=0;
 double b531=0;
 double b535=0;
 double b536=0;
 double b537=0;
 double b538=0;
 double b539=0;
 double b540=0;
 double b546=0;
 double b548=0;
 double b550=0;
 double b552=0;
 double b553=0;
 double b558=0;
 double b559=0;
 double b560=0;
 double b561=0;
 double b562=0;
 double b563=0;
 double b565=0;
 double b567=0;
 double b568=0;
 double b570=0;
 double b571=0;
 double b572=0;
 double b577=0;
 double b578=0;
 double b579=0;
 double b580=0;
 double b581=0;
 double b583=0;
 double b585=0;
 double b586=0;
 double b588=0;
 double b589=0;
 double b590=0;
 double b591=0;
 double b597=0;
 double b599=0;
 double b601=0;
 double b603=0;
 double b604=0;
 double b605=0;
 double b606=0;
 double b611=0;
 double b612=0;
 double b613=0;
 double b618=0;
 double b619=0;
 double b621=0;
 double b622=0;
 double b624=0;
 double b625=0;
 double b626=0;
 double b627=0;
 double b628=0;
 double b631=0;
 double b633=0;
 double b636=0;
 double b637=0;
 double b639=0;
 double b640=0;
 double b641=0;
 double b642=0;
 double b644=0;
 double b645=0;
 double b647=0;
 double b648=0;
 double b650=0;
 double b651=0;
 double b652=0;
 double b653=0;
 double b654=0;
 double b655=0;
 double b656=0;
 double b657=0;
 double b658=0;
 double b659=0;
 double b660=0;
 double b661=0;
 double b662=0;
 double b664=0;
 double b665=0;
 double b666=0;
 double b667=0;
 double b669=0;
 double b670=0;
 double b671=0;
 double b675=0;
 double b684=0;
 double b685=0;
 double b686=0;
 double b687=0;
 double b688=0;
 double b699=0;
 double b700=0;
 double b701=0;
 double b702=0;
 double b703=0;
 double b704=0;
 double b708=0;
 double b709=0;
 double b710=0;
 double b711=0;
 double b712=0;
 double b714=0;
 double b715=0;
 double b716=0;
 double b717=0;
 double b718=0;
 double b720=0;
 double b721=0;
 double b722=0;
 double b723=0;
 double b724=0;
 double b726=0;
 double b727=0;
 double b729=0;
 double b730=0;
 double b731=0;
 double b732=0;
 double b733=0;
 double b735=0;
 double b736=0;
 double b755=0;
 double b774=0;
 double b803=0;
 double b804=0;
 b755 += seed[0]; if(y)y[0]=v755;
 b774 += seed[1]; if(y)y[1]=v774;
 b804 += seed[2]; if(y)y[2]=v804;
 b803 += seed[3]; if(y)y[3]=v803;
 if(v20){acc(b727, 1.0, b804);}else{acc(b17, 1.0, b804);}
 if(v20){acc(b736, 1.0, b803);}else{acc(b18, 1.0, b803);}
 if(v20){acc(b667, 1.0, b774);}else{acc(b16, 1.0, b774);}
 if(v20){acc(b659, 1.0, b755);}else{acc(b15, 1.0, b755);}
 if((true && v20)){acc(b735, 1.0, b736);}
 if((true && v20)){acc(b733, 1.0, b735);}
 if((true && v20)){acc(b493, 1.0, b733);acc(b732, 1.0, b733);}
 if((true && v20)){acc(b731, double(v718), b732);acc(b718, double(v731), b732);}
 if((true && v20)){acc(b730, double(v473), b731);acc(b473, double(v730), b731);}
 if((true && v20)){acc(b729, double(v501), b730);acc(b501, double(v729), b730);}
 if((true && v20)){acc(b21, double(v728), b729);}
 if((true && v20)){acc(b726, 1.0, b727);}
 if((true && v20)){acc(b724, 1.0, b726);}
 if((true && v20)){acc(b490, 1.0, b724);acc(b723, -1.0, b724);}
 if((true && v20)){acc(b722, double(v718), b723);acc(b718, double(v722), b723);}
 if((true && v20)){acc(b721, double(v485), b722);acc(b485, double(v721), b722);}
 if((true && v20)){acc(b720, double(v501), b721);acc(b501, double(v720), b721);}
 if((true && v20)){acc(b21, double(v719), b720);}
 if((true && v20)){if(v673){acc(b675, 1.0, b718);}else{acc(b717, 1.0, b718);}}
 if(((true && v20) && !v673)){if(v681){acc(b688, 1.0, b717);}else{acc(b716, 1.0, b717);}}
 if((((true && v20) && !v673) && !v681)){if(v695){acc(b704, 1.0, b716);}else{acc(b715, 1.0, b716);}}
 if(((((true && v20) && !v673) && !v681) && !v695)){acc(b714, 1.0, b715);}
 if(((((true && v20) && !v673) && !v681) && !v695)){if(v712 < v713){acc(b712, 1.0, b714);}else{}}
 if(((((true && v20) && !v673) && !v681) && !v695)){acc(b711, 1.0, b712);}
 if(((((true && v20) && !v673) && !v681) && !v695)){acc(b710, double(v711), b711);}
 if(((((true && v20) && !v673) && !v681) && !v695)){acc(b709, -1.0, b710);}
 if(((((true && v20) && !v673) && !v681) && !v695)){acc(b708, double(v706), b709);}
 if(((((true && v20) && !v673) && !v681) && !v695)){acc(b671, 1.0, b708);}
 if(((((true && v20) && !v673) && !v681) && v695)){acc(b703, 1.0, b704);}
 if(((((true && v20) && !v673) && !v681) && v695)){acc(b702, 1.0, b703);}
 if(((((true && v20) && !v673) && !v681) && v695)){acc(b701, double(v702), b702);}
 if(((((true && v20) && !v673) && !v681) && v695)){acc(b700, -1.0, b701);}
 if(((((true && v20) && !v673) && !v681) && v695)){acc(b699, double(v697), b700);}
 if(((((true && v20) && !v673) && !v681) && v695)){acc(b671, 1.0, b699);}
 if((((true && v20) && !v673) && v681)){acc(b687, 1.0, b688);}
 if((((true && v20) && !v673) && v681)){acc(b686, double(v687), b687);}
 if((((true && v20) && !v673) && v681)){acc(b685, -1.0, b686);}
 if((((true && v20) && !v673) && v681)){acc(b684, double(v682), b685);}
 if((((true && v20) && !v673) && v681)){acc(b671, 1.0, b684);}
 if(((true && v20) && v673)){}
 if((true && v20)){acc(b670, 1.0, b671);}
 if((true && v20)){if(v540 > v669){acc(b540, 1.0, b670);}else{acc(b669, 1.0, b670);}}
 if((true && v20)){}
 if((true && v20)){acc(b666, 1.0, b667);}
 if((true && v20)){acc(b662, 1.0, b666);acc(b665, 1.0, b666);}
 if((true && v20)){acc(b664, double(v657), b665);acc(b657, double(v664), b665);}
 if((true && v20)){acc(b661, -1.0, b664);}
 if((true && v20)){acc(b655, double(v661), b662);acc(b661, double(v655), b662);}
 if((true && v20)){acc(b660, 1.0, b661);}
 if((true && v20)){acc(b591, 1.0/double(v659), b660);acc(b659, -(double(v591)/double(v659))/double(v659), b660);}
 if((true && v20)){acc(b658, 1.0, b659);}
 if((true && v20)){if(v591 > v656){acc(b591, 1.0, b658);}else{acc(b656, 1.0, b658);}}
 if((true && v20)){acc(b16, 1.0, b657);}
 if((true && v20)){acc(b15, 1.0, b656);}
 if((true && v20)){acc(b654, 1.0, b655);}
 if((true && v20)){if(v628 > v653){acc(b628, 1.0, b654);}else{acc(b653, 1.0, b654);}}
 if((true && v20)){acc(b652, 1.0, b653);}
 if((true && v20)){acc(b645, 1.0, b652);acc(b651, 1.0, b652);}
 if((true && v20)){acc(b650, double(v591), b651);acc(b591, double(v650), b651);}
 if((true && v20)){acc(b648, double(v649), b650);}
 if((true && v20)){acc(b647, double(v455), b648);acc(b455, double(v647), b648);}
 if((true && v20)){acc(b641, -1.0, b647);}
 if((true && v20)){acc(b644, double(v591), b645);acc(b591, double(v644), b645);}
 if((true && v20)){acc(b642, double(v643), b644);}
 if((true && v20)){acc(b641, double(v455), b642);acc(b455, double(v641), b642);}
 if((true && v20)){acc(b640, 1.0, b641);}
 if((true && v20)){acc(b639, -1.0, b640);}
 if((true && v20)){if(v637 < v638){acc(b637, 1.0, b639);}else{}}
 if((true && v20)){acc(b633, 1.0/double(v636), b637);acc(b636, -(double(v633)/double(v636))/double(v636), b637);}
 if((true && v20)){}
 if((true && v20)){if(v631 > v632){acc(b631, 1.0, b633);}else{}}
 if((true && v20)){acc(b591, 1.0, b631);}
 if((true && v20)){if(v610){acc(b613, 1.0, b628);}else{acc(b627, 1.0, b628);}}
 if(((true && v20) && !v610)){if(v616){acc(b622, 1.0, b627);}else{acc(b626, 1.0, b627);}}
 if((((true && v20) && !v610) && !v616)){acc(b625, 1.0, b626);}
 if((((true && v20) && !v610) && !v616)){acc(b624, double(v623), b625);}
 if((((true && v20) && !v610) && !v616)){acc(b24, double(v21), b624);acc(b21, double(v24), b624);}
 if((((true && v20) && !v610) && v616)){acc(b621, 1.0, b622);}
 if((((true && v20) && !v610) && v616)){acc(b619, 1.0, b621);}
 if((((true && v20) && !v610) && v616)){acc(b618, double(v617), b619);}
 if((((true && v20) && !v610) && v616)){acc(b24, double(v21), b618);acc(b21, double(v24), b618);}
 if(((true && v20) && v610)){acc(b612, 1.0, b613);}
 if(((true && v20) && v610)){acc(b606, double(v611), b612);acc(b611, double(v606), b612);}
 if(((true && v20) && v610)){acc(b24, double(v21), b611);acc(b21, double(v24), b611);}
 if((true && v20)){acc(b605, 1.0, b606);}
 if((true && v20)){acc(b604, 1.0, b605);}
 if((true && v20)){acc(b603, double(v595), b604);}
 if((true && v20)){if(v601 < v602){acc(b601, 1.0, b603);}else{}}
 if((true && v20)){acc(b599, 1.0/double(v600), b601);}
 if((true && v20)){if(v597 > v598){acc(b597, 1.0, b599);}else{}}
 if((true && v20)){acc(b14, 1.0, b597);}
 if((true && v20)){if(v544){acc(b572, 1.0, b591);}else{acc(b590, 1.0, b591);}}
 if(((true && v20) && !v544)){acc(b589, 1.0, b590);}
 if(((true && v20) && !v544)){if(v586 > v588){acc(b586, 1.0, b589);}else{acc(b588, 1.0, b589);}}
 if(((true && v20) && !v544)){acc(b21, double(v587), b588);}
 if(((true && v20) && !v544)){acc(b585, 1.0, b586);}
 if(((true && v20) && !v544)){if(v583 < v584){acc(b583, 1.0, b585);}else{}}
 if(((true && v20) && !v544)){if(v581 > v582){acc(b581, 1.0, b583);}else{}}
 if(((true && v20) && !v544)){acc(b580, 1.0, b581);}
 if(((true && v20) && !v544)){acc(b579, double(v574), b580);}
 if(((true && v20) && !v544)){acc(b578, 1.0/(1.0+double(v578)*double(v578)), b579);}
 if(((true && v20) && !v544)){acc(b577, double(v575), b578);}
 if(((true && v20) && !v544)){acc(b540, 1.0, b577);}
 if(((true && v20) && v544)){acc(b571, 1.0, b572);}
 if(((true && v20) && v544)){if(v568 > v570){acc(b568, 1.0, b571);}else{acc(b570, 1.0, b571);}}
 if(((true && v20) && v544)){acc(b21, double(v569), b570);}
 if(((true && v20) && v544)){acc(b567, 1.0, b568);}
 if(((true && v20) && v544)){if(v565 < v566){acc(b565, 1.0, b567);}else{}}
 if(((true && v20) && v544)){if(v563 > v564){acc(b563, 1.0, b565);}else{}}
 if(((true && v20) && v544)){acc(b562, 1.0, b563);}
 if(((true && v20) && v544)){acc(b561, double(v555), b562);}
 if(((true && v20) && v544)){acc(b560, 1.0/(1.0+double(v560)*double(v560)), b561);}
 if(((true && v20) && v544)){acc(b559, double(v556), b560);}
 if(((true && v20) && v544)){acc(b558, 1.0, b559);acc(b553, 1.0, b559);}
 if(((true && v20) && v544)){acc(b540, 1.0, b558);}
 if(((true && v20) && v544)){acc(b552, 1.0, b553);}
 if(((true && v20) && v544)){if(v550 < v551){acc(b550, 1.0, b552);}else{}}
 if(((true && v20) && v544)){acc(b548, 1.0/double(v549), b550);}
 if(((true && v20) && v544)){if(v546 > v547){acc(b546, 1.0, b548);}else{}}
 if(((true && v20) && v544)){acc(b14, 1.0, b546);}
 if((true && v20)){acc(b539, 1.0, b540);}
 if((true && v20)){acc(b531, 1.0/double(v538), b539);acc(b538, -(double(v531)/double(v538))/double(v538), b539);}
 if((true && v20)){if(v533){acc(b537, 1.0, b538);}else{acc(b529, 1.0, b538);}}
 if(((true && v20) && v533)){acc(b536, 1.0, b537);}
 if(((true && v20) && v533)){if(v535 < v529){acc(b535, 1.0, b536);}else{acc(b529, 1.0, b536);}}
 if(((true && v20) && v533)){acc(b455, double(v534), b535);}
 if((true && v20)){acc(b530, 1.0, b531);}
 if((true && v20)){acc(b10, 1.0, b530);acc(b455, -1.0, b530);}
 if((true && v20)){acc(b528, 1.0, b529);}
 if((true && v20)){if(v523 > v527){acc(b523, 1.0, b528);}else{acc(b527, 1.0, b528);}}
 if((true && v20)){acc(b525, double(v526), b527);acc(b526, double(v525), b527);}
 if((true && v20)){acc(b23, 1.0, b526);acc(b10, -1.0, b526);}
 if((true && v20)){acc(b21, double(v524), b525);}
 if((true && v20)){acc(b522, 1.0, b523);}
 if((true && v20)){acc(b516, double(v521), b522);acc(b521, double(v516), b522);}
 if((true && v20)){if(v519 > v520){acc(b519, 1.0, b521);}else{}}
 if((true && v20)){acc(b513, double(v518), b519);acc(b518, double(v513), b519);}
 if((true && v20)){if(v517 > v27){}else{acc(b27, 1.0, b518);}}
 if((true && v20)){acc(b515, double(v4), b516);acc(b4, double(v515), b516);}
 if((true && v20)){acc(b21, double(v514), b515);}
 if((true && v20)){}
 if((true && v20)){acc(b500, 1.0, b501);}
 if((true && v20)){acc(b497, 1.0, b500);acc(b499, -1.0, b500);}
 if((true && v20)){acc(b482, double(v498), b499);}
 if((true && v20)){acc(b494, 1.0/double(v496), b497);acc(b496, -(double(v494)/double(v496))/double(v496), b497);}
 if((true && v20)){acc(b8, double(v495), b496);}
 if((true && v20)){acc(b482, double(v255), b494);acc(b255, double(v482), b494);}
 if((true && v20)){acc(b492, 1.0, b493);}
 if((true && v20)){acc(b482, double(v491), b492);}
 if((true && v20)){acc(b489, 1.0, b490);}
 if((true && v20)){acc(b488, 1.0, b489);}
 if((true && v20)){acc(b10, double(v487), b488);}
 if((true && v20)){acc(b484, 1.0, b485);}
 if((true && v20)){acc(b483, 1.0/double(v482), b484);acc(b482, -(double(v483)/double(v482))/double(v482), b484);}
 if((true && v20)){acc(b475, double(v8), b483);acc(b8, double(v475), b483);}
 if((true && v20)){acc(b481, 1.0, b482);}
 if((true && v20)){acc(b12, 1.0, b481);acc(b480, 1.0, b481);}
 if((true && v20)){acc(b479, double(v21), b480);acc(b21, double(v479), b480);}
 if((true && v20)){acc(b478, double(v24), b479);acc(b24, double(v478), b479);}
 if((true && v20)){acc(b477, 1.0, b478);}
 if((true && v20)){acc(b11, -(double(v476)/double(v11))/double(v11), b477);}
 if((true && v20)){acc(b474, 1.0, b475);}
 if((true && v20)){acc(b473, double(v461), b474);acc(b461, double(v473), b474);}
 if((true && v20)){acc(b472, 1.0, b473);}
 if((true && v20)){acc(b471, -(double(v467)/double(v471))/double(v471), b472);}
 if((true && v20)){acc(b470, 1.0, b471);}
 if((true && v20)){acc(b469, 1.0/double(v466), b470);acc(b466, -(double(v469)/double(v466))/double(v466), b470);}
 if((true && v20)){acc(b255, double(v461), b469);acc(b461, double(v255), b469);}
 if((true && v20)){acc(b465, 1.0, b466);}
 if((true && v20)){acc(b464, 1.0, b465);}
 if((true && v20)){acc(b10, double(v463), b464);}
 if((true && v20)){acc(b460, 1.0, b461);}
 if((true && v20)){acc(b456, 1.0/double(v459), b460);acc(b459, -(double(v456)/double(v459))/double(v459), b460);}
 if((true && v20)){acc(b458, double(v457), b459);}
 if((true && v20)){acc(b8, 2.0*double(v8), b458);}
 if((true && v20)){acc(b255, double(v455), b456);acc(b455, double(v255), b456);}
 if((true && v20)){acc(b454, 1.0, b455);}
 if((true && v20)){acc(b449, 1.0, b454);}
 if((true && v20)){if(v302){acc(b330, 1.0, b449);}else{acc(b439, 1.0, b449);}}
 if(((true && v20) && !v302)){if(v332){acc(b360, 1.0, b439);}else{acc(b433, 1.0, b439);}}
 if((((true && v20) && !v302) && !v332)){acc(b432, 1.0, b433);}
 if((((true && v20) && !v302) && !v332)){acc(b430, 1.0, b432);acc(b431, 1.0, b432);}
 if((((true && v20) && !v302) && !v332)){acc(b427, double(v416), b431);acc(b416, double(v427), b431);}
 if((((true && v20) && !v302) && !v332)){acc(b429, double(v409), b430);acc(b409, double(v429), b430);}
 if((((true && v20) && !v302) && !v332)){acc(b427, -1.0, b429);}
 if((((true && v20) && !v302) && !v332)){acc(b426, 1.0, b427);}
 if((((true && v20) && !v302) && !v332)){acc(b420, 1.0/double(v425), b426);acc(b425, -(double(v420)/double(v425))/double(v425), b426);}
 if((((true && v20) && !v302) && !v332)){acc(b423, 1.0, b425);}
 if((((true && v20) && !v302) && !v332)){}
 if((((true && v20) && !v302) && !v332)){acc(b419, 1.0, b420);acc(b8, -1.0, b420);}
 if((((true && v20) && !v302) && !v332)){}
 if((((true && v20) && !v302) && !v332)){acc(b415, 1.0, b416);}
 if((((true && v20) && !v302) && !v332)){acc(b411, 1.0/double(v414), b415);acc(b414, -(double(v411)/double(v414))/double(v414), b415);}
 if((((true && v20) && !v302) && !v332)){if(v412 > v413){acc(b412, 1.0, b414);}else{}}
 if((((true && v20) && !v302) && !v332)){acc(b9, 1.0, b412);acc(b402, -1.0, b412);}
 if((((true && v20) && !v302) && !v332)){acc(b402, double(v410), b411);}
 if((((true && v20) && !v302) && !v332)){acc(b408, 1.0, b409);}
 if((((true && v20) && !v302) && !v332)){acc(b404, 1.0/double(v407), b408);acc(b407, -(double(v404)/double(v407))/double(v407), b408);}
 if((((true && v20) && !v302) && !v332)){if(v405 > v406){acc(b405, 1.0, b407);}else{}}
 if((((true && v20) && !v302) && !v332)){acc(b9, 1.0, b405);acc(b381, -1.0, b405);}
 if((((true && v20) && !v302) && !v332)){acc(b381, double(v403), b404);}
 if((((true && v20) && !v302) && !v332)){acc(b401, 1.0, b402);}
 if((((true && v20) && !v302) && !v332)){if(v398 < v400){acc(b398, 1.0, b401);}else{acc(b400, 1.0, b401);}}
 if((((true && v20) && !v302) && !v332)){acc(b9, double(v399), b400);}
 if((((true && v20) && !v302) && !v332)){acc(b397, 1.0, b398);}
 if((((true && v20) && !v302) && !v332)){acc(b276, 1.0, b397);acc(b396, 1.0, b397);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v395), b396);acc(b395, double(v298), b396);}
 if((((true && v20) && !v302) && !v332)){acc(b278, 1.0, b395);acc(b394, 1.0, b395);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v393), b394);acc(b393, double(v298), b394);}
 if((((true && v20) && !v302) && !v332)){acc(b280, 1.0, b393);acc(b392, 1.0, b393);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v391), b392);acc(b391, double(v298), b392);}
 if((((true && v20) && !v302) && !v332)){acc(b282, 1.0, b391);acc(b390, 1.0, b391);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v389), b390);acc(b389, double(v298), b390);}
 if((((true && v20) && !v302) && !v332)){acc(b284, 1.0, b389);acc(b388, 1.0, b389);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v387), b388);acc(b387, double(v298), b388);}
 if((((true && v20) && !v302) && !v332)){acc(b286, 1.0, b387);acc(b386, 1.0, b387);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v385), b386);acc(b385, double(v298), b386);}
 if((((true && v20) && !v302) && !v332)){acc(b288, 1.0, b385);acc(b384, 1.0, b385);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v383), b384);acc(b383, double(v298), b384);}
 if((((true && v20) && !v302) && !v332)){acc(b290, 1.0, b383);acc(b382, 1.0, b383);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v292), b382);acc(b292, double(v298), b382);}
 if((((true && v20) && !v302) && !v332)){acc(b380, 1.0, b381);}
 if((((true && v20) && !v302) && !v332)){if(v377 < v379){acc(b377, 1.0, b380);}else{acc(b379, 1.0, b380);}}
 if((((true && v20) && !v302) && !v332)){acc(b9, double(v378), b379);}
 if((((true && v20) && !v302) && !v332)){acc(b376, 1.0, b377);}
 if((((true && v20) && !v302) && !v332)){acc(b257, 1.0, b376);acc(b375, 1.0, b376);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v374), b375);acc(b374, double(v298), b375);}
 if((((true && v20) && !v302) && !v332)){acc(b259, 1.0, b374);acc(b373, 1.0, b374);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v372), b373);acc(b372, double(v298), b373);}
 if((((true && v20) && !v302) && !v332)){acc(b261, 1.0, b372);acc(b371, 1.0, b372);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v370), b371);acc(b370, double(v298), b371);}
 if((((true && v20) && !v302) && !v332)){acc(b263, 1.0, b370);acc(b369, 1.0, b370);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v368), b369);acc(b368, double(v298), b369);}
 if((((true && v20) && !v302) && !v332)){acc(b265, 1.0, b368);acc(b367, 1.0, b368);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v366), b367);acc(b366, double(v298), b367);}
 if((((true && v20) && !v302) && !v332)){acc(b267, 1.0, b366);acc(b365, 1.0, b366);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v364), b365);acc(b364, double(v298), b365);}
 if((((true && v20) && !v302) && !v332)){acc(b269, 1.0, b364);acc(b363, 1.0, b364);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v362), b363);acc(b362, double(v298), b363);}
 if((((true && v20) && !v302) && !v332)){acc(b271, 1.0, b362);acc(b361, 1.0, b362);}
 if((((true && v20) && !v302) && !v332)){acc(b298, double(v274), b361);acc(b274, double(v298), b361);}
 if((((true && v20) && !v302) && v332)){acc(b359, 1.0, b360);}
 if((((true && v20) && !v302) && v332)){acc(b355, 1.0/double(v358), b359);acc(b358, -(double(v355)/double(v358))/double(v358), b359);}
 if((((true && v20) && !v302) && v332)){if(v356 > v357){acc(b356, 1.0, b358);}else{}}
 if((((true && v20) && !v302) && v332)){acc(b9, 1.0, b356);acc(b353, -1.0, b356);}
 if((((true && v20) && !v302) && v332)){acc(b353, double(v354), b355);}
 if((((true && v20) && !v302) && v332)){acc(b352, 1.0, b353);}
 if((((true && v20) && !v302) && v332)){if(v349 < v351){acc(b349, 1.0, b352);}else{acc(b351, 1.0, b352);}}
 if((((true && v20) && !v302) && v332)){acc(b9, double(v350), b351);}
 if((((true && v20) && !v302) && v332)){acc(b348, 1.0, b349);}
 if((((true && v20) && !v302) && v332)){acc(b276, 1.0, b348);acc(b347, 1.0, b348);}
 if((((true && v20) && !v302) && v332)){acc(b298, double(v346), b347);acc(b346, double(v298), b347);}
 if((((true && v20) && !v302) && v332)){acc(b278, 1.0, b346);acc(b345, 1.0, b346);}
 if((((true && v20) && !v302) && v332)){acc(b298, double(v344), b345);acc(b344, double(v298), b345);}
 if((((true && v20) && !v302) && v332)){acc(b280, 1.0, b344);acc(b343, 1.0, b344);}
 if((((true && v20) && !v302) && v332)){acc(b298, double(v342), b343);acc(b342, double(v298), b343);}
 if((((true && v20) && !v302) && v332)){acc(b282, 1.0, b342);acc(b341, 1.0, b342);}
 if((((true && v20) && !v302) && v332)){acc(b298, double(v340), b341);acc(b340, double(v298), b341);}
 if((((true && v20) && !v302) && v332)){acc(b284, 1.0, b340);acc(b339, 1.0, b340);}
 if((((true && v20) && !v302) && v332)){acc(b298, double(v338), b339);acc(b338, double(v298), b339);}
 if((((true && v20) && !v302) && v332)){acc(b286, 1.0, b338);acc(b337, 1.0, b338);}
 if((((true && v20) && !v302) && v332)){acc(b298, double(v336), b337);acc(b336, double(v298), b337);}
 if((((true && v20) && !v302) && v332)){acc(b288, 1.0, b336);acc(b335, 1.0, b336);}
 if((((true && v20) && !v302) && v332)){acc(b298, double(v334), b335);acc(b334, double(v298), b335);}
 if((((true && v20) && !v302) && v332)){acc(b290, 1.0, b334);acc(b333, 1.0, b334);}
 if((((true && v20) && !v302) && v332)){acc(b298, double(v292), b333);acc(b292, double(v298), b333);}
 if(((true && v20) && v302)){acc(b329, 1.0, b330);}
 if(((true && v20) && v302)){acc(b325, 1.0/double(v328), b329);acc(b328, -(double(v325)/double(v328))/double(v328), b329);}
 if(((true && v20) && v302)){if(v326 > v327){acc(b326, 1.0, b328);}else{}}
 if(((true && v20) && v302)){acc(b9, 1.0, b326);acc(b323, -1.0, b326);}
 if(((true && v20) && v302)){acc(b323, double(v324), b325);}
 if(((true && v20) && v302)){acc(b322, 1.0, b323);}
 if(((true && v20) && v302)){if(v319 < v321){acc(b319, 1.0, b322);}else{acc(b321, 1.0, b322);}}
 if(((true && v20) && v302)){acc(b9, double(v320), b321);}
 if(((true && v20) && v302)){acc(b318, 1.0, b319);}
 if(((true && v20) && v302)){acc(b257, 1.0, b318);acc(b317, 1.0, b318);}
 if(((true && v20) && v302)){acc(b298, double(v316), b317);acc(b316, double(v298), b317);}
 if(((true && v20) && v302)){acc(b259, 1.0, b316);acc(b315, 1.0, b316);}
 if(((true && v20) && v302)){acc(b298, double(v314), b315);acc(b314, double(v298), b315);}
 if(((true && v20) && v302)){acc(b261, 1.0, b314);acc(b313, 1.0, b314);}
 if(((true && v20) && v302)){acc(b298, double(v312), b313);acc(b312, double(v298), b313);}
 if(((true && v20) && v302)){acc(b263, 1.0, b312);acc(b311, 1.0, b312);}
 if(((true && v20) && v302)){acc(b298, double(v310), b311);acc(b310, double(v298), b311);}
 if(((true && v20) && v302)){acc(b265, 1.0, b310);acc(b309, 1.0, b310);}
 if(((true && v20) && v302)){acc(b298, double(v308), b309);acc(b308, double(v298), b309);}
 if(((true && v20) && v302)){acc(b267, 1.0, b308);acc(b307, 1.0, b308);}
 if(((true && v20) && v302)){acc(b298, double(v306), b307);acc(b306, double(v298), b307);}
 if(((true && v20) && v302)){acc(b269, 1.0, b306);acc(b305, 1.0, b306);}
 if(((true && v20) && v302)){acc(b298, double(v304), b305);acc(b304, double(v298), b305);}
 if(((true && v20) && v302)){acc(b271, 1.0, b304);acc(b303, 1.0, b304);}
 if(((true && v20) && v302)){acc(b298, double(v274), b303);acc(b274, double(v298), b303);}
 if((true && v20)){acc(b297, 1.0, b298);}
 if((true && v20)){if(v294 > v296){acc(b294, 1.0, b297);}else{acc(b296, 1.0, b297);}}
 if((true && v20)){acc(b8, 1.0, b296);}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){acc(b273, 1.0, b274);}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){}
 if((true && v20)){acc(b254, 1.0, b255);}
 if((true && v20)){acc(b249, 1.0, b254);}
 if((true && v20)){if(v188){acc(b197, 1.0, b249);}else{acc(b242, 1.0, b249);}}
 if(((true && v20) && !v188)){if(v199){acc(b208, 1.0, b242);}else{acc(b239, 1.0, b242);}}
 if((((true && v20) && !v188) && !v199)){acc(b238, 1.0, b239);}
 if((((true && v20) && !v188) && !v199)){acc(b236, 1.0, b238);acc(b237, 1.0, b238);}
 if((((true && v20) && !v188) && !v199)){acc(b233, double(v226), b237);acc(b226, double(v233), b237);}
 if((((true && v20) && !v188) && !v199)){acc(b235, double(v217), b236);acc(b217, double(v235), b236);}
 if((((true && v20) && !v188) && !v199)){acc(b233, -1.0, b235);}
 if((((true && v20) && !v188) && !v199)){acc(b232, 1.0, b233);}
 if((((true && v20) && !v188) && !v199)){acc(b228, 1.0/double(v231), b232);acc(b231, -(double(v228)/double(v231))/double(v231), b232);}
 if((((true && v20) && !v188) && !v199)){}
 if((((true && v20) && !v188) && !v199)){acc(b8, -1.0, b228);}
 if((((true && v20) && !v188) && !v199)){acc(b225, 1.0, b226);}
 if((((true && v20) && !v188) && !v199)){acc(b224, 1.0, b225);}
 if((((true && v20) && !v188) && !v199)){acc(b221, double(v223), b224);acc(b223, double(v221), b224);}
 if((((true && v20) && !v188) && !v199)){acc(b8, 1.0, b223);}
 if((((true && v20) && !v188) && !v199)){}
 if((((true && v20) && !v188) && !v199)){acc(b216, 1.0, b217);}
 if((((true && v20) && !v188) && !v199)){acc(b215, 1.0, b216);}
 if((((true && v20) && !v188) && !v199)){acc(b212, double(v214), b215);acc(b214, double(v212), b215);}
 if((((true && v20) && !v188) && !v199)){acc(b8, 1.0, b214);}
 if((((true && v20) && !v188) && !v199)){}
 if((((true && v20) && !v188) && v199)){acc(b207, 1.0, b208);}
 if((((true && v20) && !v188) && v199)){acc(b206, 1.0, b207);}
 if((((true && v20) && !v188) && v199)){acc(b203, double(v205), b206);acc(b205, double(v203), b206);}
 if((((true && v20) && !v188) && v199)){acc(b8, 1.0, b205);}
 if((((true && v20) && !v188) && v199)){}
 if(((true && v20) && v188)){acc(b196, 1.0, b197);}
 if(((true && v20) && v188)){acc(b195, 1.0, b196);}
 if(((true && v20) && v188)){acc(b192, double(v194), b195);acc(b194, double(v192), b195);}
 if(((true && v20) && v188)){acc(b8, 1.0, b194);}
 if(((true && v20) && v188)){}
 if((true && v20)){acc(b26, 1.0, b27);}
 if((true && v20)){acc(b25, 1.0/double(v7), b26);acc(b7, -(double(v25)/double(v7))/double(v7), b26);}
 if((true && v20)){acc(b5, 1.0, b25);acc(b6, -1.0, b25);}
 if((true && v20)){acc(b3, 1.0, b24);}
 if((true && v20)){acc(b2, 1.0, b23);}
 if((true && v20)){acc(b0, 1.0, b21);}
 bar[18] += b18;
 bar[17] += b17;
 bar[16] += b16;
 bar[15] += b15;
 bar[14] += b14;
 bar[12] += b12;
 bar[11] += b11;
 bar[10] += b10;
 bar[9] += b9;
 bar[8] += b8;
 bar[7] += b7;
 bar[6] += b6;
 bar[5] += b5;
 bar[4] += b4;
 bar[3] += b3;
 bar[2] += b2;
 bar[0] += b0*(planted?1.001:1.0);
}
template<class P> inline void blend_forward(const P*x,P*y){
 const P v0 = x[0];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const P v6 = x[6];
 const P v7 = x[7];
 const P v8 = x[8];
 const P v9 = x[9];
 const P v10 = x[10];
 const P v11 = x[11];
 const P v12 = x[12];
 const P v13 = x[13];
 const P v14 = x[14];
 const P v15 = x[15];
 const P v16 = x[16];
 const P v17 = x[17];
 const P v18 = x[18];
 const float v19 = float(zero);
 const bool v20 = v3 > v19;
 const P v21 = (true && v20) ? (P(v0)) : P(0);
 const P v23 = (true && v20) ? (P(v2)) : P(0);
 const P v24 = (true && v20) ? (P(v3)) : P(0);
 const P v25 = (true && v20) ? (P(v5 - v6)) : P(0);
 const P v26 = (true && v20) ? (P(v25 / v7)) : P(0);
 const P v27 = (true && v20) ? (P(v26)) : P(0);
 const float v187 = (true && v20) ? (float(t0c)) : float(0);
 const bool v188 = (true && v20) ? (v8 >= v187) : bool(0);
 const float v189 = ((true && v20) && v188) ? (float(xlv)) : float(0);
 const float v190 = ((true && v20) && v188) ? (float(cpv)) : float(0);
 const float v191 = ((true && v20) && v188) ? (float(cliq)) : float(0);
 const float v192 = ((true && v20) && v188) ? (float(v190 - v191)) : float(0);
 const float v193 = ((true && v20) && v188) ? (float(t0c)) : float(0);
 const P v194 = ((true && v20) && v188) ? (P(v8 - v193)) : P(0);
 const P v195 = ((true && v20) && v188) ? (P(v192 * v194)) : P(0);
 const P v196 = ((true && v20) && v188) ? (P(v189 + v195)) : P(0);
 const P v197 = ((true && v20) && v188) ? (P(v196)) : P(0);
 const float v198 = ((true && v20) && !v188) ? (float(tice)) : float(0);
 const bool v199 = ((true && v20) && !v188) ? (v8 <= v198) : bool(0);
 const float v200 = (((true && v20) && !v188) && v199) ? (float(xls)) : float(0);
 const float v201 = (((true && v20) && !v188) && v199) ? (float(cpv)) : float(0);
 const float v202 = (((true && v20) && !v188) && v199) ? (float(cice)) : float(0);
 const float v203 = (((true && v20) && !v188) && v199) ? (float(v201 - v202)) : float(0);
 const float v204 = (((true && v20) && !v188) && v199) ? (float(t0c)) : float(0);
 const P v205 = (((true && v20) && !v188) && v199) ? (P(v8 - v204)) : P(0);
 const P v206 = (((true && v20) && !v188) && v199) ? (P(v203 * v205)) : P(0);
 const P v207 = (((true && v20) && !v188) && v199) ? (P(v200 + v206)) : P(0);
 const P v208 = (((true && v20) && !v188) && v199) ? (P(v207)) : P(0);
 const float v209 = (((true && v20) && !v188) && !v199) ? (float(xlv)) : float(0);
 const float v210 = (((true && v20) && !v188) && !v199) ? (float(cpv)) : float(0);
 const float v211 = (((true && v20) && !v188) && !v199) ? (float(cliq)) : float(0);
 const float v212 = (((true && v20) && !v188) && !v199) ? (float(v210 - v211)) : float(0);
 const float v213 = (((true && v20) && !v188) && !v199) ? (float(t0c)) : float(0);
 const P v214 = (((true && v20) && !v188) && !v199) ? (P(v8 - v213)) : P(0);
 const P v215 = (((true && v20) && !v188) && !v199) ? (P(v212 * v214)) : P(0);
 const P v216 = (((true && v20) && !v188) && !v199) ? (P(v209 + v215)) : P(0);
 const P v217 = (((true && v20) && !v188) && !v199) ? (P(v216)) : P(0);
 const float v218 = (((true && v20) && !v188) && !v199) ? (float(xls)) : float(0);
 const float v219 = (((true && v20) && !v188) && !v199) ? (float(cpv)) : float(0);
 const float v220 = (((true && v20) && !v188) && !v199) ? (float(cice)) : float(0);
 const float v221 = (((true && v20) && !v188) && !v199) ? (float(v219 - v220)) : float(0);
 const float v222 = (((true && v20) && !v188) && !v199) ? (float(t0c)) : float(0);
 const P v223 = (((true && v20) && !v188) && !v199) ? (P(v8 - v222)) : P(0);
 const P v224 = (((true && v20) && !v188) && !v199) ? (P(v221 * v223)) : P(0);
 const P v225 = (((true && v20) && !v188) && !v199) ? (P(v218 + v224)) : P(0);
 const P v226 = (((true && v20) && !v188) && !v199) ? (P(v225)) : P(0);
 const float v227 = (((true && v20) && !v188) && !v199) ? (float(t0c)) : float(0);
 const P v228 = (((true && v20) && !v188) && !v199) ? (P(v227 - v8)) : P(0);
 const float v229 = (((true && v20) && !v188) && !v199) ? (float(t0c)) : float(0);
 const float v230 = (((true && v20) && !v188) && !v199) ? (float(tice)) : float(0);
 const float v231 = (((true && v20) && !v188) && !v199) ? (float(v229 - v230)) : float(0);
 const P v232 = (((true && v20) && !v188) && !v199) ? (P(v228 / v231)) : P(0);
 const P v233 = (((true && v20) && !v188) && !v199) ? (P(v232)) : P(0);
 const float v234 = (((true && v20) && !v188) && !v199) ? (float(1.f)) : float(0);
 const P v235 = (((true && v20) && !v188) && !v199) ? (P(v234 - v233)) : P(0);
 const P v236 = (((true && v20) && !v188) && !v199) ? (P(v235 * v217)) : P(0);
 const P v237 = (((true && v20) && !v188) && !v199) ? (P(v233 * v226)) : P(0);
 const P v238 = (((true && v20) && !v188) && !v199) ? (P(v236 + v237)) : P(0);
 const P v239 = (((true && v20) && !v188) && !v199) ? (P(v238)) : P(0);
 const P v242 = ((true && v20) && !v188) ? (v199 ? v208 : v239) : P(0);
 const P v249 = (true && v20) ? (v188 ? v197 : v242) : P(0);
 const P v254 = (true && v20) ? (P(v249)) : P(0);
 const P v255 = (true && v20) ? (P(v254)) : P(0);
 const float v256 = (true && v20) ? (float(.611583699e03f)) : float(0);
 const P v257 = (true && v20) ? (P(v256)) : P(0);
 const float v258 = (true && v20) ? (float(.444606896e02f)) : float(0);
 const P v259 = (true && v20) ? (P(v258)) : P(0);
 const float v260 = (true && v20) ? (float(.143177157e01f)) : float(0);
 const P v261 = (true && v20) ? (P(v260)) : P(0);
 const float v262 = (true && v20) ? (float(.264224321e-1f)) : float(0);
 const P v263 = (true && v20) ? (P(v262)) : P(0);
 const float v264 = (true && v20) ? (float(.299291081e-3f)) : float(0);
 const P v265 = (true && v20) ? (P(v264)) : P(0);
 const float v266 = (true && v20) ? (float(.203154182e-5f)) : float(0);
 const P v267 = (true && v20) ? (P(v266)) : P(0);
 const float v268 = (true && v20) ? (float(.702620698e-8f)) : float(0);
 const P v269 = (true && v20) ? (P(v268)) : P(0);
 const float v270 = (true && v20) ? (float(.379534310e-11f)) : float(0);
 const P v271 = (true && v20) ? (P(v270)) : P(0);
 const float v272 = (true && v20) ? (float(.321582393e-13f)) : float(0);
 const float v273 = (true && v20) ? (-v272) : float(0);
 const P v274 = (true && v20) ? (P(v273)) : P(0);
 const float v275 = (true && v20) ? (float(.609868993e03f)) : float(0);
 const P v276 = (true && v20) ? (P(v275)) : P(0);
 const float v277 = (true && v20) ? (float(.499320233e02f)) : float(0);
 const P v278 = (true && v20) ? (P(v277)) : P(0);
 const float v279 = (true && v20) ? (float(.184672631e01f)) : float(0);
 const P v280 = (true && v20) ? (P(v279)) : P(0);
 const float v281 = (true && v20) ? (float(.402737184e-1f)) : float(0);
 const P v282 = (true && v20) ? (P(v281)) : P(0);
 const float v283 = (true && v20) ? (float(.565392987e-3f)) : float(0);
 const P v284 = (true && v20) ? (P(v283)) : P(0);
 const float v285 = (true && v20) ? (float(.521693933e-5f)) : float(0);
 const P v286 = (true && v20) ? (P(v285)) : P(0);
 const float v287 = (true && v20) ? (float(.307839583e-7f)) : float(0);
 const P v288 = (true && v20) ? (P(v287)) : P(0);
 const float v289 = (true && v20) ? (float(.105785160e-9f)) : float(0);
 const P v290 = (true && v20) ? (P(v289)) : P(0);
 const float v291 = (true && v20) ? (float(.161444444e-12f)) : float(0);
 const P v292 = (true && v20) ? (P(v291)) : P(0);
 const float v293 = (true && v20) ? (float(80.f)) : float(0);
 const float v294 = (true && v20) ? (-v293) : float(0);
 const float v295 = (true && v20) ? (float(t0c)) : float(0);
 const P v296 = (true && v20) ? (P(v8 - v295)) : P(0);
 const P v297 = (true && v20) ? (v294 > v296 ? v294 : v296) : P(0);
 const P v298 = (true && v20) ? (P(v297)) : P(0);
 const float v299 = (true && v20) ? (float(t0c)) : float(0);
 const float v300 = (true && v20) ? (float(6.f)) : float(0);
 const float v301 = (true && v20) ? (float(v299 - v300)) : float(0);
 const bool v302 = (true && v20) ? (v8 >= v301) : bool(0);
 const P v303 = ((true && v20) && v302) ? (P(v298 * v274)) : P(0);
 const P v304 = ((true && v20) && v302) ? (P(v271 + v303)) : P(0);
 const P v305 = ((true && v20) && v302) ? (P(v298 * v304)) : P(0);
 const P v306 = ((true && v20) && v302) ? (P(v269 + v305)) : P(0);
 const P v307 = ((true && v20) && v302) ? (P(v298 * v306)) : P(0);
 const P v308 = ((true && v20) && v302) ? (P(v267 + v307)) : P(0);
 const P v309 = ((true && v20) && v302) ? (P(v298 * v308)) : P(0);
 const P v310 = ((true && v20) && v302) ? (P(v265 + v309)) : P(0);
 const P v311 = ((true && v20) && v302) ? (P(v298 * v310)) : P(0);
 const P v312 = ((true && v20) && v302) ? (P(v263 + v311)) : P(0);
 const P v313 = ((true && v20) && v302) ? (P(v298 * v312)) : P(0);
 const P v314 = ((true && v20) && v302) ? (P(v261 + v313)) : P(0);
 const P v315 = ((true && v20) && v302) ? (P(v298 * v314)) : P(0);
 const P v316 = ((true && v20) && v302) ? (P(v259 + v315)) : P(0);
 const P v317 = ((true && v20) && v302) ? (P(v298 * v316)) : P(0);
 const P v318 = ((true && v20) && v302) ? (P(v257 + v317)) : P(0);
 const P v319 = ((true && v20) && v302) ? (P(v318)) : P(0);
 const float v320 = ((true && v20) && v302) ? (float(0.15f)) : float(0);
 const P v321 = ((true && v20) && v302) ? (P(v9 * v320)) : P(0);
 const P v322 = ((true && v20) && v302) ? (v319 < v321 ? v319 : v321) : P(0);
 const P v323 = ((true && v20) && v302) ? (P(v322)) : P(0);
 const float v324 = ((true && v20) && v302) ? (float(0.622f)) : float(0);
 const P v325 = ((true && v20) && v302) ? (P(v324 * v323)) : P(0);
 const P v326 = ((true && v20) && v302) ? (P(v9 - v323)) : P(0);
 const float v327 = ((true && v20) && v302) ? (float(1e-5f)) : float(0);
 const P v328 = ((true && v20) && v302) ? (v326 > v327 ? v326 : v327) : P(0);
 const P v329 = ((true && v20) && v302) ? (P(v325 / v328)) : P(0);
 const P v330 = ((true && v20) && v302) ? (P(v329)) : P(0);
 const float v331 = ((true && v20) && !v302) ? (float(tice)) : float(0);
 const bool v332 = ((true && v20) && !v302) ? (v8 <= v331) : bool(0);
 const P v333 = (((true && v20) && !v302) && v332) ? (P(v298 * v292)) : P(0);
 const P v334 = (((true && v20) && !v302) && v332) ? (P(v290 + v333)) : P(0);
 const P v335 = (((true && v20) && !v302) && v332) ? (P(v298 * v334)) : P(0);
 const P v336 = (((true && v20) && !v302) && v332) ? (P(v288 + v335)) : P(0);
 const P v337 = (((true && v20) && !v302) && v332) ? (P(v298 * v336)) : P(0);
 const P v338 = (((true && v20) && !v302) && v332) ? (P(v286 + v337)) : P(0);
 const P v339 = (((true && v20) && !v302) && v332) ? (P(v298 * v338)) : P(0);
 const P v340 = (((true && v20) && !v302) && v332) ? (P(v284 + v339)) : P(0);
 const P v341 = (((true && v20) && !v302) && v332) ? (P(v298 * v340)) : P(0);
 const P v342 = (((true && v20) && !v302) && v332) ? (P(v282 + v341)) : P(0);
 const P v343 = (((true && v20) && !v302) && v332) ? (P(v298 * v342)) : P(0);
 const P v344 = (((true && v20) && !v302) && v332) ? (P(v280 + v343)) : P(0);
 const P v345 = (((true && v20) && !v302) && v332) ? (P(v298 * v344)) : P(0);
 const P v346 = (((true && v20) && !v302) && v332) ? (P(v278 + v345)) : P(0);
 const P v347 = (((true && v20) && !v302) && v332) ? (P(v298 * v346)) : P(0);
 const P v348 = (((true && v20) && !v302) && v332) ? (P(v276 + v347)) : P(0);
 const P v349 = (((true && v20) && !v302) && v332) ? (P(v348)) : P(0);
 const float v350 = (((true && v20) && !v302) && v332) ? (float(0.15f)) : float(0);
 const P v351 = (((true && v20) && !v302) && v332) ? (P(v9 * v350)) : P(0);
 const P v352 = (((true && v20) && !v302) && v332) ? (v349 < v351 ? v349 : v351) : P(0);
 const P v353 = (((true && v20) && !v302) && v332) ? (P(v352)) : P(0);
 const float v354 = (((true && v20) && !v302) && v332) ? (float(0.622f)) : float(0);
 const P v355 = (((true && v20) && !v302) && v332) ? (P(v354 * v353)) : P(0);
 const P v356 = (((true && v20) && !v302) && v332) ? (P(v9 - v353)) : P(0);
 const float v357 = (((true && v20) && !v302) && v332) ? (float(1e-5f)) : float(0);
 const P v358 = (((true && v20) && !v302) && v332) ? (v356 > v357 ? v356 : v357) : P(0);
 const P v359 = (((true && v20) && !v302) && v332) ? (P(v355 / v358)) : P(0);
 const P v360 = (((true && v20) && !v302) && v332) ? (P(v359)) : P(0);
 const P v361 = (((true && v20) && !v302) && !v332) ? (P(v298 * v274)) : P(0);
 const P v362 = (((true && v20) && !v302) && !v332) ? (P(v271 + v361)) : P(0);
 const P v363 = (((true && v20) && !v302) && !v332) ? (P(v298 * v362)) : P(0);
 const P v364 = (((true && v20) && !v302) && !v332) ? (P(v269 + v363)) : P(0);
 const P v365 = (((true && v20) && !v302) && !v332) ? (P(v298 * v364)) : P(0);
 const P v366 = (((true && v20) && !v302) && !v332) ? (P(v267 + v365)) : P(0);
 const P v367 = (((true && v20) && !v302) && !v332) ? (P(v298 * v366)) : P(0);
 const P v368 = (((true && v20) && !v302) && !v332) ? (P(v265 + v367)) : P(0);
 const P v369 = (((true && v20) && !v302) && !v332) ? (P(v298 * v368)) : P(0);
 const P v370 = (((true && v20) && !v302) && !v332) ? (P(v263 + v369)) : P(0);
 const P v371 = (((true && v20) && !v302) && !v332) ? (P(v298 * v370)) : P(0);
 const P v372 = (((true && v20) && !v302) && !v332) ? (P(v261 + v371)) : P(0);
 const P v373 = (((true && v20) && !v302) && !v332) ? (P(v298 * v372)) : P(0);
 const P v374 = (((true && v20) && !v302) && !v332) ? (P(v259 + v373)) : P(0);
 const P v375 = (((true && v20) && !v302) && !v332) ? (P(v298 * v374)) : P(0);
 const P v376 = (((true && v20) && !v302) && !v332) ? (P(v257 + v375)) : P(0);
 const P v377 = (((true && v20) && !v302) && !v332) ? (P(v376)) : P(0);
 const float v378 = (((true && v20) && !v302) && !v332) ? (float(0.15f)) : float(0);
 const P v379 = (((true && v20) && !v302) && !v332) ? (P(v9 * v378)) : P(0);
 const P v380 = (((true && v20) && !v302) && !v332) ? (v377 < v379 ? v377 : v379) : P(0);
 const P v381 = (((true && v20) && !v302) && !v332) ? (P(v380)) : P(0);
 const P v382 = (((true && v20) && !v302) && !v332) ? (P(v298 * v292)) : P(0);
 const P v383 = (((true && v20) && !v302) && !v332) ? (P(v290 + v382)) : P(0);
 const P v384 = (((true && v20) && !v302) && !v332) ? (P(v298 * v383)) : P(0);
 const P v385 = (((true && v20) && !v302) && !v332) ? (P(v288 + v384)) : P(0);
 const P v386 = (((true && v20) && !v302) && !v332) ? (P(v298 * v385)) : P(0);
 const P v387 = (((true && v20) && !v302) && !v332) ? (P(v286 + v386)) : P(0);
 const P v388 = (((true && v20) && !v302) && !v332) ? (P(v298 * v387)) : P(0);
 const P v389 = (((true && v20) && !v302) && !v332) ? (P(v284 + v388)) : P(0);
 const P v390 = (((true && v20) && !v302) && !v332) ? (P(v298 * v389)) : P(0);
 const P v391 = (((true && v20) && !v302) && !v332) ? (P(v282 + v390)) : P(0);
 const P v392 = (((true && v20) && !v302) && !v332) ? (P(v298 * v391)) : P(0);
 const P v393 = (((true && v20) && !v302) && !v332) ? (P(v280 + v392)) : P(0);
 const P v394 = (((true && v20) && !v302) && !v332) ? (P(v298 * v393)) : P(0);
 const P v395 = (((true && v20) && !v302) && !v332) ? (P(v278 + v394)) : P(0);
 const P v396 = (((true && v20) && !v302) && !v332) ? (P(v298 * v395)) : P(0);
 const P v397 = (((true && v20) && !v302) && !v332) ? (P(v276 + v396)) : P(0);
 const P v398 = (((true && v20) && !v302) && !v332) ? (P(v397)) : P(0);
 const float v399 = (((true && v20) && !v302) && !v332) ? (float(0.15f)) : float(0);
 const P v400 = (((true && v20) && !v302) && !v332) ? (P(v9 * v399)) : P(0);
 const P v401 = (((true && v20) && !v302) && !v332) ? (v398 < v400 ? v398 : v400) : P(0);
 const P v402 = (((true && v20) && !v302) && !v332) ? (P(v401)) : P(0);
 const float v403 = (((true && v20) && !v302) && !v332) ? (float(0.622f)) : float(0);
 const P v404 = (((true && v20) && !v302) && !v332) ? (P(v403 * v381)) : P(0);
 const P v405 = (((true && v20) && !v302) && !v332) ? (P(v9 - v381)) : P(0);
 const float v406 = (((true && v20) && !v302) && !v332) ? (float(1e-5f)) : float(0);
 const P v407 = (((true && v20) && !v302) && !v332) ? (v405 > v406 ? v405 : v406) : P(0);
 const P v408 = (((true && v20) && !v302) && !v332) ? (P(v404 / v407)) : P(0);
 const P v409 = (((true && v20) && !v302) && !v332) ? (P(v408)) : P(0);
 const float v410 = (((true && v20) && !v302) && !v332) ? (float(0.622f)) : float(0);
 const P v411 = (((true && v20) && !v302) && !v332) ? (P(v410 * v402)) : P(0);
 const P v412 = (((true && v20) && !v302) && !v332) ? (P(v9 - v402)) : P(0);
 const float v413 = (((true && v20) && !v302) && !v332) ? (float(1e-5f)) : float(0);
 const P v414 = (((true && v20) && !v302) && !v332) ? (v412 > v413 ? v412 : v413) : P(0);
 const P v415 = (((true && v20) && !v302) && !v332) ? (P(v411 / v414)) : P(0);
 const P v416 = (((true && v20) && !v302) && !v332) ? (P(v415)) : P(0);
 const float v417 = (((true && v20) && !v302) && !v332) ? (float(t0c)) : float(0);
 const float v418 = (((true && v20) && !v302) && !v332) ? (float(6.f)) : float(0);
 const float v419 = (((true && v20) && !v302) && !v332) ? (float(v417 - v418)) : float(0);
 const P v420 = (((true && v20) && !v302) && !v332) ? (P(v419 - v8)) : P(0);
 const float v421 = (((true && v20) && !v302) && !v332) ? (float(t0c)) : float(0);
 const float v422 = (((true && v20) && !v302) && !v332) ? (float(6.f)) : float(0);
 const float v423 = (((true && v20) && !v302) && !v332) ? (float(v421 - v422)) : float(0);
 const float v424 = (((true && v20) && !v302) && !v332) ? (float(tice)) : float(0);
 const float v425 = (((true && v20) && !v302) && !v332) ? (float(v423 - v424)) : float(0);
 const P v426 = (((true && v20) && !v302) && !v332) ? (P(v420 / v425)) : P(0);
 const P v427 = (((true && v20) && !v302) && !v332) ? (P(v426)) : P(0);
 const float v428 = (((true && v20) && !v302) && !v332) ? (float(1.f)) : float(0);
 const P v429 = (((true && v20) && !v302) && !v332) ? (P(v428 - v427)) : P(0);
 const P v430 = (((true && v20) && !v302) && !v332) ? (P(v429 * v409)) : P(0);
 const P v431 = (((true && v20) && !v302) && !v332) ? (P(v427 * v416)) : P(0);
 const P v432 = (((true && v20) && !v302) && !v332) ? (P(v430 + v431)) : P(0);
 const P v433 = (((true && v20) && !v302) && !v332) ? (P(v432)) : P(0);
 const P v439 = ((true && v20) && !v302) ? (v332 ? v360 : v433) : P(0);
 const P v449 = (true && v20) ? (v302 ? v330 : v439) : P(0);
 const P v454 = (true && v20) ? (P(v449)) : P(0);
 const P v455 = (true && v20) ? (P(v454)) : P(0);
 const P v456 = (true && v20) ? (P(v255 * v455)) : P(0);
 const float v457 = (true && v20) ? (float(r_v)) : float(0);
 const P v458 = (true && v20) ? (P(v8*v8)) : P(0);
 const P v459 = (true && v20) ? (P(v457 * v458)) : P(0);
 const P v460 = (true && v20) ? (P(v456 / v459)) : P(0);
 const P v461 = (true && v20) ? (P(v460)) : P(0);
 const float v462 = (true && v20) ? (float(cp)) : float(0);
 const float v463 = (true && v20) ? (float(cpv)) : float(0);
 const P v464 = (true && v20) ? (P(v10 * v463)) : P(0);
 const P v465 = (true && v20) ? (P(v462 + v464)) : P(0);
 const P v466 = (true && v20) ? (P(v465)) : P(0);
 const float v467 = (true && v20) ? (float(one)) : float(0);
 const float v468 = (true && v20) ? (float(one)) : float(0);
 const P v469 = (true && v20) ? (P(v255 * v461)) : P(0);
 const P v470 = (true && v20) ? (P(v469 / v466)) : P(0);
 const P v471 = (true && v20) ? (P(v468 + v470)) : P(0);
 const P v472 = (true && v20) ? (P(v467 / v471)) : P(0);
 const P v473 = (true && v20) ? (P(v472)) : P(0);
 const P v474 = (true && v20) ? (P(v473 * v461)) : P(0);
 const P v475 = (true && v20) ? (P(v474)) : P(0);
 const float v476 = (true && v20) ? (float(xlvcp)) : float(0);
 const P v477 = (true && v20) ? (P(v476 / v11)) : P(0);
 const P v478 = (true && v20) ? (P(v477)) : P(0);
 const P v479 = (true && v20) ? (P(v478 * v24)) : P(0);
 const P v480 = (true && v20) ? (P(v479 * v21)) : P(0);
 const P v481 = (true && v20) ? (P(v12 + v480)) : P(0);
 const P v482 = (true && v20) ? (P(v481)) : P(0);
 const P v483 = (true && v20) ? (P(v475 * v8)) : P(0);
 const P v484 = (true && v20) ? (P(v483 / v482)) : P(0);
 const P v485 = (true && v20) ? (P(v484)) : P(0);
 const float v486 = (true && v20) ? (float(one)) : float(0);
 const float v487 = (true && v20) ? (float(p608)) : float(0);
 const P v488 = (true && v20) ? (P(v487 * v10)) : P(0);
 const P v489 = (true && v20) ? (P(v486 + v488)) : P(0);
 const P v490 = (true && v20) ? (P(v489)) : P(0);
 const float v491 = (true && v20) ? (float(p608)) : float(0);
 const P v492 = (true && v20) ? (P(v491 * v482)) : P(0);
 const P v493 = (true && v20) ? (P(v492)) : P(0);
 const P v494 = (true && v20) ? (P(v482 * v255)) : P(0);
 const float v495 = (true && v20) ? (float(cp)) : float(0);
 const P v496 = (true && v20) ? (P(v8 * v495)) : P(0);
 const P v497 = (true && v20) ? (P(v494 / v496)) : P(0);
 const float v498 = (true && v20) ? (float(1.61f)) : float(0);
 const P v499 = (true && v20) ? (P(v498 * v482)) : P(0);
 const P v500 = (true && v20) ? (P(v497 - v499)) : P(0);
 const P v501 = (true && v20) ? (P(v500)) : P(0);
 const float v512 = (true && v20) ? (float(1800.f)) : float(0);
 const P v513 = (true && v20) ? (P(v512)) : P(0);
 const float v514 = (true && v20) ? (float(five)) : float(0);
 const P v515 = (true && v20) ? (P(v514 * v21)) : P(0);
 const P v516 = (true && v20) ? (P(v515 * v4)) : P(0);
 const float v517 = (true && v20) ? (float(zero)) : float(0);
 const P v518 = (true && v20) ? (v517 > v27 ? v517 : v27) : P(0);
 const P v519 = (true && v20) ? (P(v513 * v518)) : P(0);
 const float v520 = (true && v20) ? (float(4e-3f)) : float(0);
 const P v521 = (true && v20) ? (v519 > v520 ? v519 : v520) : P(0);
 const P v522 = (true && v20) ? (P(v516 * v521)) : P(0);
 const P v523 = (true && v20) ? (P(v522)) : P(0);
 const float v524 = (true && v20) ? (float(three)) : float(0);
 const P v525 = (true && v20) ? (P(v524 * v21)) : P(0);
 const P v526 = (true && v20) ? (P(v23 - v10)) : P(0);
 const P v527 = (true && v20) ? (P(v525 * v526)) : P(0);
 const P v528 = (true && v20) ? (v523 > v527 ? v523 : v527) : P(0);
 const P v529 = (true && v20) ? (P(v528)) : P(0);
 const P v530 = (true && v20) ? (P(v10 - v455)) : P(0);
 const P v531 = (true && v20) ? (P(v530)) : P(0);
 const float v532 = (true && v20) ? (float(zero)) : float(0);
 const bool v533 = (true && v20) ? (v531 > v532) : bool(0);
 const float v534 = ((true && v20) && v533) ? (float(0.015f)) : float(0);
 const P v535 = ((true && v20) && v533) ? (P(v455 * v534)) : P(0);
 const P v536 = ((true && v20) && v533) ? (v535 < v529 ? v535 : v529) : P(0);
 const P v537 = ((true && v20) && v533) ? (P(v536)) : P(0);
 const P v538 = (true && v20) ? (v533 ? v537 : v529) : P(0);
 const P v539 = (true && v20) ? (P(v531 / v538)) : P(0);
 const P v540 = (true && v20) ? (P(v539)) : P(0);
 const float v541 = (true && v20) ? (float(1.5f)) : float(0);
 const P v542 = (true && v20) ? (P(v13 - v541)) : P(0);
 const float v543 = (true && v20) ? (float(zero)) : float(0);
 const bool v544 = (true && v20) ? (v542 >= v543) : bool(0);
 const float v545 = ((true && v20) && v544) ? (float(0.05f)) : float(0);
 const P v546 = ((true && v20) && v544) ? (P(v14 - v545)) : P(0);
 const float v547 = ((true && v20) && v544) ? (float(zero)) : float(0);
 const P v548 = ((true && v20) && v544) ? (v546 > v547 ? v546 : v547) : P(0);
 const float v549 = ((true && v20) && v544) ? (float(0.05f)) : float(0);
 const P v550 = ((true && v20) && v544) ? (P(v548 / v549)) : P(0);
 const float v551 = ((true && v20) && v544) ? (float(one)) : float(0);
 const P v552 = ((true && v20) && v544) ? (v550 < v551 ? v550 : v551) : P(0);
 const P v553 = ((true && v20) && v544) ? (P(v552)) : P(0);
 const float v554 = ((true && v20) && v544) ? (float(p5)) : float(0);
 const float v555 = ((true && v20) && v544) ? (float(0.36f)) : float(0);
 const float v556 = ((true && v20) && v544) ? (float(1.8f)) : float(0);
 const float v557 = ((true && v20) && v544) ? (float(p2)) : float(0);
 const P v558 = ((true && v20) && v544) ? (P(v540 + v557)) : P(0);
 const P v559 = ((true && v20) && v544) ? (P(v558 + v553)) : P(0);
 const P v560 = ((true && v20) && v544) ? (P(v556 * v559)) : P(0);
 const P v561 = ((true && v20) && v544) ? (P(std::atan(v560))) : P(0);
 const P v562 = ((true && v20) && v544) ? (P(v555 * v561)) : P(0);
 const P v563 = ((true && v20) && v544) ? (P(v554 + v562)) : P(0);
 const float v564 = ((true && v20) && v544) ? (float(0.01f)) : float(0);
 const P v565 = ((true && v20) && v544) ? (v563 > v564 ? v563 : v564) : P(0);
 const float v566 = ((true && v20) && v544) ? (float(one)) : float(0);
 const P v567 = ((true && v20) && v544) ? (v565 < v566 ? v565 : v566) : P(0);
 const P v568 = ((true && v20) && v544) ? (P(v567)) : P(0);
 const float v569 = ((true && v20) && v544) ? (float(1.2f)) : float(0);
 const P v570 = ((true && v20) && v544) ? (P(v569 * v21)) : P(0);
 const P v571 = ((true && v20) && v544) ? (v568 > v570 ? v568 : v570) : P(0);
 const P v572 = ((true && v20) && v544) ? (P(v571)) : P(0);
 const float v573 = ((true && v20) && !v544) ? (float(p5)) : float(0);
 const float v574 = ((true && v20) && !v544) ? (float(0.36f)) : float(0);
 const float v575 = ((true && v20) && !v544) ? (float(1.8f)) : float(0);
 const float v576 = ((true && v20) && !v544) ? (float(p2)) : float(0);
 const P v577 = ((true && v20) && !v544) ? (P(v540 + v576)) : P(0);
 const P v578 = ((true && v20) && !v544) ? (P(v575 * v577)) : P(0);
 const P v579 = ((true && v20) && !v544) ? (P(std::atan(v578))) : P(0);
 const P v580 = ((true && v20) && !v544) ? (P(v574 * v579)) : P(0);
 const P v581 = ((true && v20) && !v544) ? (P(v573 + v580)) : P(0);
 const float v582 = ((true && v20) && !v544) ? (float(0.01f)) : float(0);
 const P v583 = ((true && v20) && !v544) ? (v581 > v582 ? v581 : v582) : P(0);
 const float v584 = ((true && v20) && !v544) ? (float(one)) : float(0);
 const P v585 = ((true && v20) && !v544) ? (v583 < v584 ? v583 : v584) : P(0);
 const P v586 = ((true && v20) && !v544) ? (P(v585)) : P(0);
 const float v587 = ((true && v20) && !v544) ? (float(1.8f)) : float(0);
 const P v588 = ((true && v20) && !v544) ? (P(v587 * v21)) : P(0);
 const P v589 = ((true && v20) && !v544) ? (v586 > v588 ? v586 : v588) : P(0);
 const P v590 = ((true && v20) && !v544) ? (P(v589)) : P(0);
 const P v591 = (true && v20) ? (v544 ? v572 : v590) : P(0);
 const float v594 = (true && v20) ? (float(1.20f)) : float(0);
 const float v595 = (true && v20) ? (float(1.0f)) : float(0);
 const float v596 = (true && v20) ? (float(0.04f)) : float(0);
 const P v597 = (true && v20) ? (P(v14 - v596)) : P(0);
 const float v598 = (true && v20) ? (float(zero)) : float(0);
 const P v599 = (true && v20) ? (v597 > v598 ? v597 : v598) : P(0);
 const float v600 = (true && v20) ? (float(0.05f)) : float(0);
 const P v601 = (true && v20) ? (P(v599 / v600)) : P(0);
 const float v602 = (true && v20) ? (float(one)) : float(0);
 const P v603 = (true && v20) ? (v601 < v602 ? v601 : v602) : P(0);
 const P v604 = (true && v20) ? (P(v595 * v603)) : P(0);
 const P v605 = (true && v20) ? (P(v594 + v604)) : P(0);
 const P v606 = (true && v20) ? (P(v605)) : P(0);
 const float v607 = (true && v20) ? (float(1.5f)) : float(0);
 const P v608 = (true && v20) ? (P(v13 - v607)) : P(0);
 const float v609 = (true && v20) ? (float(zero)) : float(0);
 const bool v610 = (true && v20) ? (v608 >= v609) : bool(0);
 const P v611 = ((true && v20) && v610) ? (P(v24 * v21)) : P(0);
 const P v612 = ((true && v20) && v610) ? (P(v606 * v611)) : P(0);
 const P v613 = ((true && v20) && v610) ? (P(v612)) : P(0);
 const P v614 = ((true && v20) && !v610) ? (P(v24 * v21)) : P(0);
 const float v615 = ((true && v20) && !v610) ? (float(5e-5f)) : float(0);
 const bool v616 = ((true && v20) && !v610) ? (v614 > v615) : bool(0);
 const float v617 = (((true && v20) && !v610) && v616) ? (float(1.86f)) : float(0);
 const P v618 = (((true && v20) && !v610) && v616) ? (P(v24 * v21)) : P(0);
 const P v619 = (((true && v20) && !v610) && v616) ? (P(v617 * v618)) : P(0);
 const float v620 = (((true && v20) && !v610) && v616) ? (float(2.2e-5f)) : float(0);
 const P v621 = (((true && v20) && !v610) && v616) ? (P(v619 - v620)) : P(0);
 const P v622 = (((true && v20) && !v610) && v616) ? (P(v621)) : P(0);
 const float v623 = (((true && v20) && !v610) && !v616) ? (float(1.20f)) : float(0);
 const P v624 = (((true && v20) && !v610) && !v616) ? (P(v24 * v21)) : P(0);
 const P v625 = (((true && v20) && !v610) && !v616) ? (P(v623 * v624)) : P(0);
 const P v626 = (((true && v20) && !v610) && !v616) ? (P(v625)) : P(0);
 const P v627 = ((true && v20) && !v610) ? (v616 ? v622 : v626) : P(0);
 const P v628 = (true && v20) ? (v610 ? v613 : v627) : P(0);
 const float v629 = (true && v20) ? (float(one)) : float(0);
 const float v630 = (true && v20) ? (float(thirty)) : float(0);
 const P v631 = (true && v20) ? (P(v591 - v630)) : P(0);
 const float v632 = (true && v20) ? (float(zero)) : float(0);
 const P v633 = (true && v20) ? (v631 > v632 ? v631 : v632) : P(0);
 const float v634 = (true && v20) ? (float(hundred)) : float(0);
 const float v635 = (true && v20) ? (float(thirty)) : float(0);
 const float v636 = (true && v20) ? (float(v634 - v635)) : float(0);
 const P v637 = (true && v20) ? (P(v633 / v636)) : P(0);
 const float v638 = (true && v20) ? (float(one)) : float(0);
 const P v639 = (true && v20) ? (v637 < v638 ? v637 : v638) : P(0);
 const P v640 = (true && v20) ? (P(v629 - v639)) : P(0);
 const P v641 = (true && v20) ? (P(v640)) : P(0);
 const P v642 = (true && v20) ? (P(v641 * v455)) : P(0);
 const float v643 = (true && v20) ? (float(0.01f)) : float(0);
 const P v644 = (true && v20) ? (P(v642 * v643)) : P(0);
 const P v645 = (true && v20) ? (P(v644 * v591)) : P(0);
 const float v646 = (true && v20) ? (float(one)) : float(0);
 const P v647 = (true && v20) ? (P(v646 - v641)) : P(0);
 const P v648 = (true && v20) ? (P(v647 * v455)) : P(0);
 const float v649 = (true && v20) ? (float(0.025f)) : float(0);
 const P v650 = (true && v20) ? (P(v648 * v649)) : P(0);
 const P v651 = (true && v20) ? (P(v650 * v591)) : P(0);
 const P v652 = (true && v20) ? (P(v645 + v651)) : P(0);
 const P v653 = (true && v20) ? (P(v652)) : P(0);
 const P v654 = (true && v20) ? (v628 > v653 ? v628 : v653) : P(0);
 const P v655 = (true && v20) ? (P(v654)) : P(0);
 const P v656 = (true && v20) ? (P(v15)) : P(0);
 const P v657 = (true && v20) ? (P(v16)) : P(0);
 const P v658 = (true && v20) ? (v591 > v656 ? v591 : v656) : P(0);
 const P v659 = (true && v20) ? (P(v658)) : P(0);
 const P v660 = (true && v20) ? (P(v591 / v659)) : P(0);
 const P v661 = (true && v20) ? (P(v660)) : P(0);
 const P v662 = (true && v20) ? (P(v655 * v661)) : P(0);
 const float v663 = (true && v20) ? (float(one)) : float(0);
 const P v664 = (true && v20) ? (P(v663 - v661)) : P(0);
 const P v665 = (true && v20) ? (P(v664 * v657)) : P(0);
 const P v666 = (true && v20) ? (P(v662 + v665)) : P(0);
 const P v667 = (true && v20) ? (P(v666)) : P(0);
 const float v668 = (true && v20) ? (float(2.25f)) : float(0);
 const float v669 = (true && v20) ? (-v668) : float(0);
 const P v670 = (true && v20) ? (v540 > v669 ? v540 : v669) : P(0);
 const P v671 = (true && v20) ? (P(v670)) : P(0);
 const float v672 = (true && v20) ? (float(one)) : float(0);
 const bool v673 = (true && v20) ? (v671 >= v672) : bool(0);
 const float v674 = ((true && v20) && v673) ? (float(one)) : float(0);
 const P v675 = ((true && v20) && v673) ? (P(v674)) : P(0);
 const float v676 = ((true && v20) && !v673) ? (float(1.7f)) : float(0);
 const float v677 = ((true && v20) && !v673) ? (-v676) : float(0);
 const bool v678 = ((true && v20) && !v673) ? (v671 >= v677) : bool(0);
 const float v679 = ((true && v20) && !v673) ? (float(one)) : float(0);
 const bool v680 = ((true && v20) && !v673) ? (v671 < v679) : bool(0);
 const bool v681 = ((true && v20) && !v673) ? (v678 && v680) : bool(0);
 const float v682 = (((true && v20) && !v673) && v681) ? (float(p4)) : float(0);
 const float v683 = (((true && v20) && !v673) && v681) ? (float(one)) : float(0);
 const P v684 = (((true && v20) && !v673) && v681) ? (P(v671 - v683)) : P(0);
 const P v685 = (((true && v20) && !v673) && v681) ? (P(v682 * v684)) : P(0);
 const P v686 = (((true && v20) && !v673) && v681) ? (-v685) : P(0);
 const P v687 = (((true && v20) && !v673) && v681) ? (P(std::exp(v686))) : P(0);
 const P v688 = (((true && v20) && !v673) && v681) ? (P(v687)) : P(0);
 const float v689 = (((true && v20) && !v673) && !v681) ? (float(2.5f)) : float(0);
 const float v690 = (((true && v20) && !v673) && !v681) ? (-v689) : float(0);
 const bool v691 = (((true && v20) && !v673) && !v681) ? (v671 >= v690) : bool(0);
 const float v692 = (((true && v20) && !v673) && !v681) ? (float(1.7f)) : float(0);
 const float v693 = (((true && v20) && !v673) && !v681) ? (-v692) : float(0);
 const bool v694 = (((true && v20) && !v673) && !v681) ? (v671 < v693) : bool(0);
 const bool v695 = (((true && v20) && !v673) && !v681) ? (v691 && v694) : bool(0);
 const float v696 = ((((true && v20) && !v673) && !v681) && v695) ? (float(three)) : float(0);
 const float v697 = ((((true && v20) && !v673) && !v681) && v695) ? (float(3.8f)) : float(0);
 const float v698 = ((((true && v20) && !v673) && !v681) && v695) ? (float(1.7f)) : float(0);
 const P v699 = ((((true && v20) && !v673) && !v681) && v695) ? (P(v671 + v698)) : P(0);
 const P v700 = ((((true && v20) && !v673) && !v681) && v695) ? (P(v697 * v699)) : P(0);
 const P v701 = ((((true && v20) && !v673) && !v681) && v695) ? (-v700) : P(0);
 const P v702 = ((((true && v20) && !v673) && !v681) && v695) ? (P(std::exp(v701))) : P(0);
 const P v703 = ((((true && v20) && !v673) && !v681) && v695) ? (P(v696 + v702)) : P(0);
 const P v704 = ((((true && v20) && !v673) && !v681) && v695) ? (P(v703)) : P(0);
 const float v705 = ((((true && v20) && !v673) && !v681) && !v695) ? (float(23.9f)) : float(0);
 const float v706 = ((((true && v20) && !v673) && !v681) && !v695) ? (float(1.6f)) : float(0);
 const float v707 = ((((true && v20) && !v673) && !v681) && !v695) ? (float(2.5f)) : float(0);
 const P v708 = ((((true && v20) && !v673) && !v681) && !v695) ? (P(v671 + v707)) : P(0);
 const P v709 = ((((true && v20) && !v673) && !v681) && !v695) ? (P(v706 * v708)) : P(0);
 const P v710 = ((((true && v20) && !v673) && !v681) && !v695) ? (-v709) : P(0);
 const P v711 = ((((true && v20) && !v673) && !v681) && !v695) ? (P(std::exp(v710))) : P(0);
 const P v712 = ((((true && v20) && !v673) && !v681) && !v695) ? (P(v705 + v711)) : P(0);
 const float v713 = ((((true && v20) && !v673) && !v681) && !v695) ? (float(60.f)) : float(0);
 const P v714 = ((((true && v20) && !v673) && !v681) && !v695) ? (v712 < v713 ? v712 : v713) : P(0);
 const P v715 = ((((true && v20) && !v673) && !v681) && !v695) ? (P(v714)) : P(0);
 const P v716 = (((true && v20) && !v673) && !v681) ? (v695 ? v704 : v715) : P(0);
 const P v717 = ((true && v20) && !v673) ? (v681 ? v688 : v716) : P(0);
 const P v718 = (true && v20) ? (v673 ? v675 : v717) : P(0);
 const float v719 = (true && v20) ? (float(1.5f)) : float(0);
 const P v720 = (true && v20) ? (P(v719 * v21)) : P(0);
 const P v721 = (true && v20) ? (P(v720 * v501)) : P(0);
 const P v722 = (true && v20) ? (P(v721 * v485)) : P(0);
 const P v723 = (true && v20) ? (P(v722 * v718)) : P(0);
 const P v724 = (true && v20) ? (P(v490 - v723)) : P(0);
 const float v725 = (true && v20) ? (float(one)) : float(0);
 const P v726 = (true && v20) ? (P(v724 - v725)) : P(0);
 const P v727 = (true && v20) ? (P(v726)) : P(0);
 const float v728 = (true && v20) ? (float(1.5f)) : float(0);
 const P v729 = (true && v20) ? (P(v728 * v21)) : P(0);
 const P v730 = (true && v20) ? (P(v729 * v501)) : P(0);
 const P v731 = (true && v20) ? (P(v730 * v473)) : P(0);
 const P v732 = (true && v20) ? (P(v731 * v718)) : P(0);
 const P v733 = (true && v20) ? (P(v493 + v732)) : P(0);
 const float v734 = (true && v20) ? (float(tv0)) : float(0);
 const P v735 = (true && v20) ? (P(v733 - v734)) : P(0);
 const P v736 = (true && v20) ? (P(v735)) : P(0);
 const P v755 = v20 ? v659 : v15;
 const P v774 = v20 ? v667 : v16;
 const P v803 = v20 ? v736 : v18;
 const P v804 = v20 ? v727 : v17;
y[0]=v755;
y[1]=v774;
y[2]=v804;
y[3]=v803;
}
template<class P> inline void speed_reverse(const P*x,const double*seed,double*bar,P*y=nullptr,bool planted=false){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const float v6 = float(grav);
 const P v7 = P(v0 / v1);
 const float v8 = float(one);
 const P v9 = P(v7 - v8);
 const P v10 = P(v6 * v9);
 const P v11 = P(v10);
 const float v12 = float(0.f);
 const bool v13 = v11 > v12;
 const float v14 = (true && v13) ? (float(0.15f)) : float(0);
 const P v15 = (true && v13) ? (P(v14)) : P(0);
 const float v16 = (true && !v13) ? (float(0.2f)) : float(0);
 const P v17 = (true && !v13) ? (P(v16)) : P(0);
 const P v18 = v13 ? v15 : v17;
 const float v19 = float(0.2f);
 const bool v20 = v2 < v19;
 const float v21 = (true && v20) ? (float(2.f)) : float(0);
 const P v22 = (true && v20) ? (P(v21 * v4)) : P(0);
 const P v23 = (true && v20) ? (-v22) : P(0);
 const P v24 = (true && v20) ? (P(v23 * v3)) : P(0);
 const P v25 = (true && v20) ? (P(v18 * v11)) : P(0);
 const float v26 = (true && v20) ? (float(0.2f)) : float(0);
 const P v27 = (true && v20) ? (v3 > v26 ? v3 : v26) : P(0);
 const P v28 = (true && v20) ? (P(v25 / v27)) : P(0);
 const P v29 = (true && v20) ? (P(v24 + v28)) : P(0);
 const float v30 = (true && v20) ? (float(250.f)) : float(0);
 const P v31 = (true && v20) ? (v5 < v30 ? v5 : v30) : P(0);
 const P v32 = (true && v20) ? (P(v29 * v31)) : P(0);
 const P v33 = (true && v20) ? (P(v3 + v32)) : P(0);
 const P v34 = (true && v20) ? (P(v33)) : P(0);
 const float v35 = (true && !v20) ? (float(2.f)) : float(0);
 const P v36 = (true && !v20) ? (P(v35 * v4)) : P(0);
 const P v37 = (true && !v20) ? (-v36) : P(0);
 const P v38 = (true && !v20) ? (P(v37 * v3)) : P(0);
 const P v39 = (true && !v20) ? (P(v18 * v11)) : P(0);
 const P v40 = (true && !v20) ? (P(v39 / v3)) : P(0);
 const P v41 = (true && !v20) ? (P(v38 + v40)) : P(0);
 const float v42 = (true && !v20) ? (float(250.f)) : float(0);
 const P v43 = (true && !v20) ? (v5 < v42 ? v5 : v42) : P(0);
 const P v44 = (true && !v20) ? (P(v41 * v43)) : P(0);
 const P v45 = (true && !v20) ? (P(v3 + v44)) : P(0);
 const P v46 = (true && !v20) ? (P(v45)) : P(0);
 const P v47 = v20 ? v34 : v46;
 const float v48 = float(1.25f);
 const P v49 = P(v48 * v5);
 const float v50 = float(200.f);
 const P v51 = P(v49 / v50);
 const float v52 = float(two);
 const P v53 = v51 < v52 ? v51 : v52;
 const P v54 = P(v3 + v53);
 const bool v55 = v47 > v54;
 const float v56 = (true && v55) ? (float(1.25f)) : float(0);
 const P v57 = (true && v55) ? (P(v56 * v5)) : P(0);
 const float v58 = (true && v55) ? (float(200.f)) : float(0);
 const P v59 = (true && v55) ? (P(v57 / v58)) : P(0);
 const float v60 = (true && v55) ? (float(two)) : float(0);
 const P v61 = (true && v55) ? (v59 < v60 ? v59 : v60) : P(0);
 const P v62 = (true && v55) ? (P(v3 + v61)) : P(0);
 const P v63 = (true && v55) ? (P(v62)) : P(0);
 const P v64 = v55 ? v63 : v47;
 const float v65 = float(1.25f);
 const P v66 = P(v65 * v5);
 const float v67 = float(200.f);
 const P v68 = P(v66 / v67);
 const float v69 = float(two);
 const P v70 = v68 < v69 ? v68 : v69;
 const P v71 = P(v3 - v70);
 const bool v72 = v64 < v71;
 const float v73 = (true && v72) ? (float(1.25f)) : float(0);
 const P v74 = (true && v72) ? (P(v73 * v5)) : P(0);
 const float v75 = (true && v72) ? (float(200.f)) : float(0);
 const P v76 = (true && v72) ? (P(v74 / v75)) : P(0);
 const float v77 = (true && v72) ? (float(two)) : float(0);
 const P v78 = (true && v72) ? (v76 < v77 ? v76 : v77) : P(0);
 const P v79 = (true && v72) ? (P(v3 - v78)) : P(0);
 const P v80 = (true && v72) ? (P(v79)) : P(0);
 const P v81 = v72 ? v80 : v64;
 const float v82 = float(zero);
 const P v83 = v82 > v81 ? v82 : v81;
 const float v84 = float(3.0f);
 const P v85 = v83 < v84 ? v83 : v84;
 const P v86 = P(v85);
 const P v87 = v86;
 const P v88 = P(v87);
 double b0=0;
 double b1=0;
 double b3=0;
 double b4=0;
 double b5=0;
 double b7=0;
 double b9=0;
 double b10=0;
 double b11=0;
 double b15=0;
 double b17=0;
 double b18=0;
 double b22=0;
 double b23=0;
 double b24=0;
 double b25=0;
 double b27=0;
 double b28=0;
 double b29=0;
 double b31=0;
 double b32=0;
 double b33=0;
 double b34=0;
 double b36=0;
 double b37=0;
 double b38=0;
 double b39=0;
 double b40=0;
 double b41=0;
 double b43=0;
 double b44=0;
 double b45=0;
 double b46=0;
 double b47=0;
 double b57=0;
 double b59=0;
 double b61=0;
 double b62=0;
 double b63=0;
 double b64=0;
 double b74=0;
 double b76=0;
 double b78=0;
 double b79=0;
 double b80=0;
 double b81=0;
 double b83=0;
 double b85=0;
 double b86=0;
 double b87=0;
 double b88=0;
 b88 += seed[0]; if(y)y[0]=v88;
 acc(b87, 1.0, b88);
 acc(b86, plume_ramp(double(v86),double(v3)), b87);
 acc(b85, 1.0, b86);
 if(v83 < v84){acc(b83, 1.0, b85);}else{}
 if(v82 > v81){}else{acc(b81, 1.0, b83);}
 if(v72){acc(b80, 1.0, b81);}else{acc(b64, 1.0, b81);}
 if((true && v72)){acc(b79, 1.0, b80);}
 if((true && v72)){acc(b3, 1.0, b79);acc(b78, -1.0, b79);}
 if((true && v72)){if(v76 < v77){acc(b76, 1.0, b78);}else{}}
 if((true && v72)){acc(b74, 1.0/double(v75), b76);}
 if((true && v72)){acc(b5, double(v73), b74);}
 if(v55){acc(b63, 1.0, b64);}else{acc(b47, 1.0, b64);}
 if((true && v55)){acc(b62, 1.0, b63);}
 if((true && v55)){acc(b3, 1.0, b62);acc(b61, 1.0, b62);}
 if((true && v55)){if(v59 < v60){acc(b59, 1.0, b61);}else{}}
 if((true && v55)){acc(b57, 1.0/double(v58), b59);}
 if((true && v55)){acc(b5, double(v56), b57);}
 if(v20){acc(b34, 1.0, b47);}else{acc(b46, 1.0, b47);}
 if((true && !v20)){acc(b45, 1.0, b46);}
 if((true && !v20)){acc(b3, 1.0, b45);acc(b44, 1.0, b45);}
 if((true && !v20)){acc(b41, double(v43), b44);acc(b43, double(v41), b44);}
 if((true && !v20)){if(v5 < v42){acc(b5, 1.0, b43);}else{}}
 if((true && !v20)){acc(b38, 1.0, b41);acc(b40, 1.0, b41);}
 if((true && !v20)){acc(b39, 1.0/double(v3), b40);acc(b3, -(double(v39)/double(v3))/double(v3), b40);}
 if((true && !v20)){acc(b18, double(v11), b39);acc(b11, double(v18), b39);}
 if((true && !v20)){acc(b37, double(v3), b38);acc(b3, double(v37), b38);}
 if((true && !v20)){acc(b36, -1.0, b37);}
 if((true && !v20)){acc(b4, double(v35), b36);}
 if((true && v20)){acc(b33, 1.0, b34);}
 if((true && v20)){acc(b3, 1.0, b33);acc(b32, 1.0, b33);}
 if((true && v20)){acc(b29, double(v31), b32);acc(b31, double(v29), b32);}
 if((true && v20)){if(v5 < v30){acc(b5, 1.0, b31);}else{}}
 if((true && v20)){acc(b24, 1.0, b29);acc(b28, 1.0, b29);}
 if((true && v20)){acc(b25, 1.0/double(v27), b28);acc(b27, -(double(v25)/double(v27))/double(v27), b28);}
 if((true && v20)){if(v3 > v26){acc(b3, 1.0, b27);}else{}}
 if((true && v20)){acc(b18, double(v11), b25);acc(b11, double(v18), b25);}
 if((true && v20)){acc(b23, double(v3), b24);acc(b3, double(v23), b24);}
 if((true && v20)){acc(b22, -1.0, b23);}
 if((true && v20)){acc(b4, double(v21), b22);}
 if(v13){acc(b15, 1.0, b18);}else{acc(b17, 1.0, b18);}
 if((true && !v13)){}
 if((true && v13)){}
 acc(b10, 1.0, b11);
 acc(b9, double(v6), b10);
 acc(b7, 1.0, b9);
 acc(b0, 1.0/double(v1), b7);
 acc(b1, -(double(v0)/double(v1))/double(v1), b7);
 bar[5] += b5;
 bar[4] += b4;
 bar[3] += b3;
 bar[1] += b1;
 bar[0] += b0*(planted?1.001:1.0);
}
template<class P> inline void speed_forward(const P*x,P*y){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const float v6 = float(grav);
 const P v7 = P(v0 / v1);
 const float v8 = float(one);
 const P v9 = P(v7 - v8);
 const P v10 = P(v6 * v9);
 const P v11 = P(v10);
 const float v12 = float(0.f);
 const bool v13 = v11 > v12;
 const float v14 = (true && v13) ? (float(0.15f)) : float(0);
 const P v15 = (true && v13) ? (P(v14)) : P(0);
 const float v16 = (true && !v13) ? (float(0.2f)) : float(0);
 const P v17 = (true && !v13) ? (P(v16)) : P(0);
 const P v18 = v13 ? v15 : v17;
 const float v19 = float(0.2f);
 const bool v20 = v2 < v19;
 const float v21 = (true && v20) ? (float(2.f)) : float(0);
 const P v22 = (true && v20) ? (P(v21 * v4)) : P(0);
 const P v23 = (true && v20) ? (-v22) : P(0);
 const P v24 = (true && v20) ? (P(v23 * v3)) : P(0);
 const P v25 = (true && v20) ? (P(v18 * v11)) : P(0);
 const float v26 = (true && v20) ? (float(0.2f)) : float(0);
 const P v27 = (true && v20) ? (v3 > v26 ? v3 : v26) : P(0);
 const P v28 = (true && v20) ? (P(v25 / v27)) : P(0);
 const P v29 = (true && v20) ? (P(v24 + v28)) : P(0);
 const float v30 = (true && v20) ? (float(250.f)) : float(0);
 const P v31 = (true && v20) ? (v5 < v30 ? v5 : v30) : P(0);
 const P v32 = (true && v20) ? (P(v29 * v31)) : P(0);
 const P v33 = (true && v20) ? (P(v3 + v32)) : P(0);
 const P v34 = (true && v20) ? (P(v33)) : P(0);
 const float v35 = (true && !v20) ? (float(2.f)) : float(0);
 const P v36 = (true && !v20) ? (P(v35 * v4)) : P(0);
 const P v37 = (true && !v20) ? (-v36) : P(0);
 const P v38 = (true && !v20) ? (P(v37 * v3)) : P(0);
 const P v39 = (true && !v20) ? (P(v18 * v11)) : P(0);
 const P v40 = (true && !v20) ? (P(v39 / v3)) : P(0);
 const P v41 = (true && !v20) ? (P(v38 + v40)) : P(0);
 const float v42 = (true && !v20) ? (float(250.f)) : float(0);
 const P v43 = (true && !v20) ? (v5 < v42 ? v5 : v42) : P(0);
 const P v44 = (true && !v20) ? (P(v41 * v43)) : P(0);
 const P v45 = (true && !v20) ? (P(v3 + v44)) : P(0);
 const P v46 = (true && !v20) ? (P(v45)) : P(0);
 const P v47 = v20 ? v34 : v46;
 const float v48 = float(1.25f);
 const P v49 = P(v48 * v5);
 const float v50 = float(200.f);
 const P v51 = P(v49 / v50);
 const float v52 = float(two);
 const P v53 = v51 < v52 ? v51 : v52;
 const P v54 = P(v3 + v53);
 const bool v55 = v47 > v54;
 const float v56 = (true && v55) ? (float(1.25f)) : float(0);
 const P v57 = (true && v55) ? (P(v56 * v5)) : P(0);
 const float v58 = (true && v55) ? (float(200.f)) : float(0);
 const P v59 = (true && v55) ? (P(v57 / v58)) : P(0);
 const float v60 = (true && v55) ? (float(two)) : float(0);
 const P v61 = (true && v55) ? (v59 < v60 ? v59 : v60) : P(0);
 const P v62 = (true && v55) ? (P(v3 + v61)) : P(0);
 const P v63 = (true && v55) ? (P(v62)) : P(0);
 const P v64 = v55 ? v63 : v47;
 const float v65 = float(1.25f);
 const P v66 = P(v65 * v5);
 const float v67 = float(200.f);
 const P v68 = P(v66 / v67);
 const float v69 = float(two);
 const P v70 = v68 < v69 ? v68 : v69;
 const P v71 = P(v3 - v70);
 const bool v72 = v64 < v71;
 const float v73 = (true && v72) ? (float(1.25f)) : float(0);
 const P v74 = (true && v72) ? (P(v73 * v5)) : P(0);
 const float v75 = (true && v72) ? (float(200.f)) : float(0);
 const P v76 = (true && v72) ? (P(v74 / v75)) : P(0);
 const float v77 = (true && v72) ? (float(two)) : float(0);
 const P v78 = (true && v72) ? (v76 < v77 ? v76 : v77) : P(0);
 const P v79 = (true && v72) ? (P(v3 - v78)) : P(0);
 const P v80 = (true && v72) ? (P(v79)) : P(0);
 const P v81 = v72 ? v80 : v64;
 const float v82 = float(zero);
 const P v83 = v82 > v81 ? v82 : v81;
 const float v84 = float(3.0f);
 const P v85 = v83 < v84 ? v83 : v84;
 const P v86 = P(v85);
 const P v87 = v86;
 const P v88 = P(v87);
y[0]=v88;
}
template<class P> inline void entrainment_reverse(const P*x,const double*seed,double*bar,P*y=nullptr,bool planted=false){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const P v6 = x[6];
 const float v7 = float(p5);
 const P v8 = P(v7 * v1);
 const P v9 = P(v0 + v8);
 const P v10 = P(v9);
 const float v11 = float(p3);
 const float v12 = float(0.0005f);
 const P v13 = P(v2 * v12);
 const P v14 = P(v11 + v13);
 const P v15 = P(v14);
 const float v16 = float(0.21f);
 const P v17 = P(std::sqrt(v3));
 const float v18 = float(1.30f);
 const P v19 = v17 > v18 ? v17 : v18;
 const float v20 = float(1.57f);
 const P v21 = v19 < v20 ? v19 : v20;
 const P v22 = P(v16 * v21);
 const P v23 = P(v22);
 const P v24 = P(v10 - v4);
 const float v25 = float(zero);
 const P v26 = v24 > v25 ? v24 : v25;
 const float v27 = float(500.f);
 const P v28 = P(v26 / v27);
 const float v29 = float(one);
 const P v30 = v28 < v29 ? v28 : v29;
 const P v31 = P(v30);
 const float v32 = float(one);
 const P v33 = P(v32 - v31);
 const P v34 = P(v23 * v33);
 const float v35 = float(0.33f);
 const P v36 = P(v31 * v35);
 const P v37 = P(v34 + v36);
 const P v38 = P(v37);
 const P v39 = v5 > v15 ? v5 : v15;
 const float v40 = float(one);
 const P v41 = v39 < v40 ? v39 : v40;
 const P v42 = P(v41 * v2);
 const P v43 = P(v38 / v42);
 const P v44 = P(v43);
 const float v45 = float(0.0003f);
 const P v46 = v44 > v45 ? v44 : v45;
 const P v47 = P(v46);
 const float v48 = float(1500.f);
 const P v49 = P(v4 + v48);
 const float v50 = float(4000.f);
 const P v51 = v49 < v50 ? v49 : v50;
 const bool v52 = v0 >= v51;
 const float v53 = (true && v52) ? (float(1500.f)) : float(0);
 const P v54 = (true && v52) ? (P(v4 + v53)) : P(0);
 const float v55 = (true && v52) ? (float(4000.f)) : float(0);
 const P v56 = (true && v52) ? (v54 < v55 ? v54 : v55) : P(0);
 const P v57 = (true && v52) ? (P(v0 - v56)) : P(0);
 const float v58 = (true && v52) ? (float(5.0e-6f)) : float(0);
 const P v59 = (true && v52) ? (P(v57 * v58)) : P(0);
 const P v60 = (true && v52) ? (P(v47 + v59)) : P(0);
 const P v61 = (true && v52) ? (P(v60)) : P(0);
 const P v62 = v52 ? v61 : v47;
 const float v63 = float(one);
 const P v64 = P(v63 - v6);
 const P v65 = P(v62 * v64);
 const P v66 = P(v65);
 const float v67 = float(0.9f);
 const P v68 = P(v67 / v1);
 const P v69 = v66 < v68 ? v66 : v68;
 const P v70 = P(v69);
 double b0=0;
 double b1=0;
 double b2=0;
 double b3=0;
 double b4=0;
 double b5=0;
 double b6=0;
 double b8=0;
 double b9=0;
 double b10=0;
 double b13=0;
 double b14=0;
 double b15=0;
 double b17=0;
 double b19=0;
 double b21=0;
 double b22=0;
 double b23=0;
 double b24=0;
 double b26=0;
 double b28=0;
 double b30=0;
 double b31=0;
 double b33=0;
 double b34=0;
 double b36=0;
 double b37=0;
 double b38=0;
 double b39=0;
 double b41=0;
 double b42=0;
 double b43=0;
 double b44=0;
 double b46=0;
 double b47=0;
 double b54=0;
 double b56=0;
 double b57=0;
 double b59=0;
 double b60=0;
 double b61=0;
 double b62=0;
 double b64=0;
 double b65=0;
 double b66=0;
 double b68=0;
 double b69=0;
 double b70=0;
 b70 += seed[0]; if(y)y[0]=v70;
 acc(b69, 1.0, b70);
 if(v66 < v68){acc(b66, 1.0, b69);}else{acc(b68, 1.0, b69);}
 acc(b1, -(double(v67)/double(v1))/double(v1), b68);
 acc(b65, 1.0, b66);
 acc(b62, double(v64), b65);
 acc(b64, double(v62), b65);
 acc(b6, -1.0, b64);
 if(v52){acc(b61, 1.0, b62);}else{acc(b47, 1.0, b62);}
 if((true && v52)){acc(b60, 1.0, b61);}
 if((true && v52)){acc(b47, 1.0, b60);acc(b59, 1.0, b60);}
 if((true && v52)){acc(b57, double(v58), b59);}
 if((true && v52)){acc(b0, 1.0, b57);acc(b56, -1.0, b57);}
 if((true && v52)){if(v54 < v55){acc(b54, 1.0, b56);}else{}}
 if((true && v52)){acc(b4, 1.0, b54);}
 acc(b46, 1.0, b47);
 if(v44 > v45){acc(b44, 1.0, b46);}else{}
 acc(b43, 1.0, b44);
 acc(b38, 1.0/double(v42), b43);
 acc(b42, -(double(v38)/double(v42))/double(v42), b43);
 acc(b41, double(v2), b42);
 acc(b2, double(v41), b42);
 if(v39 < v40){acc(b39, 1.0, b41);}else{}
 if(v5 > v15){acc(b5, 1.0, b39);}else{acc(b15, 1.0, b39);}
 acc(b37, 1.0, b38);
 acc(b34, 1.0, b37);
 acc(b36, 1.0, b37);
 acc(b31, double(v35), b36);
 acc(b23, double(v33), b34);
 acc(b33, double(v23), b34);
 acc(b31, -1.0, b33);
 acc(b30, 1.0, b31);
 if(v28 < v29){acc(b28, 1.0, b30);}else{}
 acc(b26, 1.0/double(v27), b28);
 if(v24 > v25){acc(b24, 1.0, b26);}else{}
 acc(b10, 1.0, b24);
 acc(b4, -1.0, b24);
 acc(b22, 1.0, b23);
 acc(b21, double(v16), b22);
 if(v19 < v20){acc(b19, 1.0, b21);}else{}
 if(v17 > v18){acc(b17, 1.0, b19);}else{}
 acc(b3, 1.0/(2.0*double(v17)), b17);
 acc(b14, 1.0, b15);
 acc(b13, 1.0, b14);
 acc(b2, double(v12), b13);
 acc(b9, 1.0, b10);
 acc(b0, 1.0, b9);
 acc(b8, 1.0, b9);
 acc(b1, double(v7), b8);
 bar[6] += b6;
 bar[5] += b5;
 bar[4] += b4;
 bar[3] += b3;
 bar[2] += b2;
 bar[1] += b1;
 bar[0] += b0*(planted?1.001:1.0);
}
template<class P> inline void entrainment_forward(const P*x,P*y){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const P v6 = x[6];
 const float v7 = float(p5);
 const P v8 = P(v7 * v1);
 const P v9 = P(v0 + v8);
 const P v10 = P(v9);
 const float v11 = float(p3);
 const float v12 = float(0.0005f);
 const P v13 = P(v2 * v12);
 const P v14 = P(v11 + v13);
 const P v15 = P(v14);
 const float v16 = float(0.21f);
 const P v17 = P(std::sqrt(v3));
 const float v18 = float(1.30f);
 const P v19 = v17 > v18 ? v17 : v18;
 const float v20 = float(1.57f);
 const P v21 = v19 < v20 ? v19 : v20;
 const P v22 = P(v16 * v21);
 const P v23 = P(v22);
 const P v24 = P(v10 - v4);
 const float v25 = float(zero);
 const P v26 = v24 > v25 ? v24 : v25;
 const float v27 = float(500.f);
 const P v28 = P(v26 / v27);
 const float v29 = float(one);
 const P v30 = v28 < v29 ? v28 : v29;
 const P v31 = P(v30);
 const float v32 = float(one);
 const P v33 = P(v32 - v31);
 const P v34 = P(v23 * v33);
 const float v35 = float(0.33f);
 const P v36 = P(v31 * v35);
 const P v37 = P(v34 + v36);
 const P v38 = P(v37);
 const P v39 = v5 > v15 ? v5 : v15;
 const float v40 = float(one);
 const P v41 = v39 < v40 ? v39 : v40;
 const P v42 = P(v41 * v2);
 const P v43 = P(v38 / v42);
 const P v44 = P(v43);
 const float v45 = float(0.0003f);
 const P v46 = v44 > v45 ? v44 : v45;
 const P v47 = P(v46);
 const float v48 = float(1500.f);
 const P v49 = P(v4 + v48);
 const float v50 = float(4000.f);
 const P v51 = v49 < v50 ? v49 : v50;
 const bool v52 = v0 >= v51;
 const float v53 = (true && v52) ? (float(1500.f)) : float(0);
 const P v54 = (true && v52) ? (P(v4 + v53)) : P(0);
 const float v55 = (true && v52) ? (float(4000.f)) : float(0);
 const P v56 = (true && v52) ? (v54 < v55 ? v54 : v55) : P(0);
 const P v57 = (true && v52) ? (P(v0 - v56)) : P(0);
 const float v58 = (true && v52) ? (float(5.0e-6f)) : float(0);
 const P v59 = (true && v52) ? (P(v57 * v58)) : P(0);
 const P v60 = (true && v52) ? (P(v47 + v59)) : P(0);
 const P v61 = (true && v52) ? (P(v60)) : P(0);
 const P v62 = v52 ? v61 : v47;
 const float v63 = float(one);
 const P v64 = P(v63 - v6);
 const P v65 = P(v62 * v64);
 const P v66 = P(v65);
 const float v67 = float(0.9f);
 const P v68 = P(v67 / v1);
 const P v69 = v66 < v68 ? v66 : v68;
 const P v70 = P(v69);
y[0]=v70;
}
template<class P> inline void mix_reverse(const P*x,const double*seed,double*bar,P*y=nullptr,bool planted=false){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = P(v2 * v3);
 const P v5 = P(v4);
 const float v6 = float(one);
 const P v7 = P(v6 - v5);
 const P v8 = P(v0 * v7);
 const P v9 = P(v1 * v5);
 const P v10 = P(v8 + v9);
 const P v11 = P(v10);
 double b0=0;
 double b1=0;
 double b2=0;
 double b3=0;
 double b4=0;
 double b5=0;
 double b7=0;
 double b8=0;
 double b9=0;
 double b10=0;
 double b11=0;
 b11 += seed[0]; if(y)y[0]=v11;
 acc(b10, 1.0, b11);
 acc(b8, 1.0, b10);
 acc(b9, 1.0, b10);
 acc(b1, double(v5), b9);
 acc(b5, double(v1), b9);
 acc(b0, double(v7), b8);
 acc(b7, double(v0), b8);
 acc(b5, -1.0, b7);
 acc(b4, 1.0, b5);
 acc(b2, double(v3), b4);
 acc(b3, double(v2), b4);
 bar[3] += b3;
 bar[2] += b2;
 bar[1] += b1;
 bar[0] += b0*(planted?1.001:1.0);
}
template<class P> inline void mix_forward(const P*x,P*y){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = P(v2 * v3);
 const P v5 = P(v4);
 const float v6 = float(one);
 const P v7 = P(v6 - v5);
 const P v8 = P(v0 * v7);
 const P v9 = P(v1 * v5);
 const P v10 = P(v8 + v9);
 const P v11 = P(v10);
y[0]=v11;
}
template<class P> inline void flux_reverse(const P*x,const double*seed,double*bar,P*y=nullptr,bool planted=false){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const P v6 = P(v1 * v2);
 const P v7 = P(v6 * v3);
 const P v8 = P(v7 * v4);
 const P v9 = P(v8 * v5);
 const P v10 = P(v0 + v9);
 const P v11 = P(v10);
 double b0=0;
 double b1=0;
 double b2=0;
 double b3=0;
 double b4=0;
 double b5=0;
 double b6=0;
 double b7=0;
 double b8=0;
 double b9=0;
 double b10=0;
 double b11=0;
 b11 += seed[0]; if(y)y[0]=v11;
 acc(b10, 1.0, b11);
 acc(b0, 1.0, b10);
 acc(b9, 1.0, b10);
 acc(b8, double(v5), b9);
 acc(b5, double(v8), b9);
 acc(b7, double(v4), b8);
 acc(b4, double(v7), b8);
 acc(b6, double(v3), b7);
 acc(b3, double(v6), b7);
 acc(b1, double(v2), b6);
 acc(b2, double(v1), b6);
 bar[5] += b5;
 bar[4] += b4;
 bar[3] += b3;
 bar[2] += b2;
 bar[1] += b1;
 bar[0] += b0*(planted?1.001:1.0);
}
template<class P> inline void flux_forward(const P*x,P*y){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const P v6 = P(v1 * v2);
 const P v7 = P(v6 * v3);
 const P v8 = P(v7 * v4);
 const P v9 = P(v8 * v5);
 const P v10 = P(v0 + v9);
 const P v11 = P(v10);
y[0]=v11;
}
template<class P> inline void interp_reverse(const P*x,const double*seed,double*bar,P*y=nullptr,bool planted=false){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = P(v0 * v3);
 const P v5 = P(v1 * v2);
 const P v6 = P(v4 + v5);
 const P v7 = P(v3 + v2);
 const P v8 = P(v6 / v7);
 const P v9 = P(v8);
 double b0=0;
 double b1=0;
 double b2=0;
 double b3=0;
 double b4=0;
 double b5=0;
 double b6=0;
 double b7=0;
 double b8=0;
 double b9=0;
 b9 += seed[0]; if(y)y[0]=v9;
 acc(b8, 1.0, b9);
 acc(b6, 1.0/double(v7), b8);
 acc(b7, -(double(v6)/double(v7))/double(v7), b8);
 acc(b3, 1.0, b7);
 acc(b2, 1.0, b7);
 acc(b4, 1.0, b6);
 acc(b5, 1.0, b6);
 acc(b1, double(v2), b5);
 acc(b2, double(v1), b5);
 acc(b0, double(v3), b4);
 acc(b3, double(v0), b4);
 bar[3] += b3;
 bar[2] += b2;
 bar[1] += b1;
 bar[0] += b0*(planted?1.001:1.0);
}
template<class P> inline void interp_forward(const P*x,P*y){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = P(v0 * v3);
 const P v5 = P(v1 * v2);
 const P v6 = P(v4 + v5);
 const P v7 = P(v3 + v2);
 const P v8 = P(v6 / v7);
 const P v9 = P(v8);
y[0]=v9;
}
template<class P> inline void cond_step_reverse(const P*x,const double*seed,double*bar,P*y=nullptr,bool planted=false){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = P(v0 * v1);
 const float v6 = float(xlvcp);
 const P v7 = P(v6 / v0);
 const P v8 = P(v7 * v4);
 const P v9 = P(v5 + v8);
 const P v10 = P(v9);
 const float v11 = float(.611583699e03f);
 const P v12 = P(v11);
 const float v13 = float(.444606896e02f);
 const P v14 = P(v13);
 const float v15 = float(.143177157e01f);
 const P v16 = P(v15);
 const float v17 = float(.264224321e-1f);
 const P v18 = P(v17);
 const float v19 = float(.299291081e-3f);
 const P v20 = P(v19);
 const float v21 = float(.203154182e-5f);
 const P v22 = P(v21);
 const float v23 = float(.702620698e-8f);
 const P v24 = P(v23);
 const float v25 = float(.379534310e-11f);
 const P v26 = P(v25);
 const float v27 = float(.321582393e-13f);
 const float v28 = -v27;
 const P v29 = P(v28);
 const float v30 = float(.609868993e03f);
 const P v31 = P(v30);
 const float v32 = float(.499320233e02f);
 const P v33 = P(v32);
 const float v34 = float(.184672631e01f);
 const P v35 = P(v34);
 const float v36 = float(.402737184e-1f);
 const P v37 = P(v36);
 const float v38 = float(.565392987e-3f);
 const P v39 = P(v38);
 const float v40 = float(.521693933e-5f);
 const P v41 = P(v40);
 const float v42 = float(.307839583e-7f);
 const P v43 = P(v42);
 const float v44 = float(.105785160e-9f);
 const P v45 = P(v44);
 const float v46 = float(.161444444e-12f);
 const P v47 = P(v46);
 const float v48 = float(80.f);
 const float v49 = -v48;
 const float v50 = float(t0c);
 const P v51 = P(v10 - v50);
 const P v52 = v49 > v51 ? v49 : v51;
 const P v53 = P(v52);
 const float v54 = float(t0c);
 const float v55 = float(6.f);
 const float v56 = float(v54 - v55);
 const bool v57 = v10 >= v56;
 const P v58 = (true && v57) ? (P(v53 * v29)) : P(0);
 const P v59 = (true && v57) ? (P(v26 + v58)) : P(0);
 const P v60 = (true && v57) ? (P(v53 * v59)) : P(0);
 const P v61 = (true && v57) ? (P(v24 + v60)) : P(0);
 const P v62 = (true && v57) ? (P(v53 * v61)) : P(0);
 const P v63 = (true && v57) ? (P(v22 + v62)) : P(0);
 const P v64 = (true && v57) ? (P(v53 * v63)) : P(0);
 const P v65 = (true && v57) ? (P(v20 + v64)) : P(0);
 const P v66 = (true && v57) ? (P(v53 * v65)) : P(0);
 const P v67 = (true && v57) ? (P(v18 + v66)) : P(0);
 const P v68 = (true && v57) ? (P(v53 * v67)) : P(0);
 const P v69 = (true && v57) ? (P(v16 + v68)) : P(0);
 const P v70 = (true && v57) ? (P(v53 * v69)) : P(0);
 const P v71 = (true && v57) ? (P(v14 + v70)) : P(0);
 const P v72 = (true && v57) ? (P(v53 * v71)) : P(0);
 const P v73 = (true && v57) ? (P(v12 + v72)) : P(0);
 const P v74 = (true && v57) ? (P(v73)) : P(0);
 const float v75 = (true && v57) ? (float(0.15f)) : float(0);
 const P v76 = (true && v57) ? (P(v3 * v75)) : P(0);
 const P v77 = (true && v57) ? (v74 < v76 ? v74 : v76) : P(0);
 const P v78 = (true && v57) ? (P(v77)) : P(0);
 const float v79 = (true && v57) ? (float(0.622f)) : float(0);
 const P v80 = (true && v57) ? (P(v79 * v78)) : P(0);
 const P v81 = (true && v57) ? (P(v3 - v78)) : P(0);
 const float v82 = (true && v57) ? (float(1e-5f)) : float(0);
 const P v83 = (true && v57) ? (v81 > v82 ? v81 : v82) : P(0);
 const P v84 = (true && v57) ? (P(v80 / v83)) : P(0);
 const P v85 = (true && v57) ? (P(v84)) : P(0);
 const float v86 = (true && !v57) ? (float(tice)) : float(0);
 const bool v87 = (true && !v57) ? (v10 <= v86) : bool(0);
 const P v88 = ((true && !v57) && v87) ? (P(v53 * v47)) : P(0);
 const P v89 = ((true && !v57) && v87) ? (P(v45 + v88)) : P(0);
 const P v90 = ((true && !v57) && v87) ? (P(v53 * v89)) : P(0);
 const P v91 = ((true && !v57) && v87) ? (P(v43 + v90)) : P(0);
 const P v92 = ((true && !v57) && v87) ? (P(v53 * v91)) : P(0);
 const P v93 = ((true && !v57) && v87) ? (P(v41 + v92)) : P(0);
 const P v94 = ((true && !v57) && v87) ? (P(v53 * v93)) : P(0);
 const P v95 = ((true && !v57) && v87) ? (P(v39 + v94)) : P(0);
 const P v96 = ((true && !v57) && v87) ? (P(v53 * v95)) : P(0);
 const P v97 = ((true && !v57) && v87) ? (P(v37 + v96)) : P(0);
 const P v98 = ((true && !v57) && v87) ? (P(v53 * v97)) : P(0);
 const P v99 = ((true && !v57) && v87) ? (P(v35 + v98)) : P(0);
 const P v100 = ((true && !v57) && v87) ? (P(v53 * v99)) : P(0);
 const P v101 = ((true && !v57) && v87) ? (P(v33 + v100)) : P(0);
 const P v102 = ((true && !v57) && v87) ? (P(v53 * v101)) : P(0);
 const P v103 = ((true && !v57) && v87) ? (P(v31 + v102)) : P(0);
 const P v104 = ((true && !v57) && v87) ? (P(v103)) : P(0);
 const float v105 = ((true && !v57) && v87) ? (float(0.15f)) : float(0);
 const P v106 = ((true && !v57) && v87) ? (P(v3 * v105)) : P(0);
 const P v107 = ((true && !v57) && v87) ? (v104 < v106 ? v104 : v106) : P(0);
 const P v108 = ((true && !v57) && v87) ? (P(v107)) : P(0);
 const float v109 = ((true && !v57) && v87) ? (float(0.622f)) : float(0);
 const P v110 = ((true && !v57) && v87) ? (P(v109 * v108)) : P(0);
 const P v111 = ((true && !v57) && v87) ? (P(v3 - v108)) : P(0);
 const float v112 = ((true && !v57) && v87) ? (float(1e-5f)) : float(0);
 const P v113 = ((true && !v57) && v87) ? (v111 > v112 ? v111 : v112) : P(0);
 const P v114 = ((true && !v57) && v87) ? (P(v110 / v113)) : P(0);
 const P v115 = ((true && !v57) && v87) ? (P(v114)) : P(0);
 const P v116 = ((true && !v57) && !v87) ? (P(v53 * v29)) : P(0);
 const P v117 = ((true && !v57) && !v87) ? (P(v26 + v116)) : P(0);
 const P v118 = ((true && !v57) && !v87) ? (P(v53 * v117)) : P(0);
 const P v119 = ((true && !v57) && !v87) ? (P(v24 + v118)) : P(0);
 const P v120 = ((true && !v57) && !v87) ? (P(v53 * v119)) : P(0);
 const P v121 = ((true && !v57) && !v87) ? (P(v22 + v120)) : P(0);
 const P v122 = ((true && !v57) && !v87) ? (P(v53 * v121)) : P(0);
 const P v123 = ((true && !v57) && !v87) ? (P(v20 + v122)) : P(0);
 const P v124 = ((true && !v57) && !v87) ? (P(v53 * v123)) : P(0);
 const P v125 = ((true && !v57) && !v87) ? (P(v18 + v124)) : P(0);
 const P v126 = ((true && !v57) && !v87) ? (P(v53 * v125)) : P(0);
 const P v127 = ((true && !v57) && !v87) ? (P(v16 + v126)) : P(0);
 const P v128 = ((true && !v57) && !v87) ? (P(v53 * v127)) : P(0);
 const P v129 = ((true && !v57) && !v87) ? (P(v14 + v128)) : P(0);
 const P v130 = ((true && !v57) && !v87) ? (P(v53 * v129)) : P(0);
 const P v131 = ((true && !v57) && !v87) ? (P(v12 + v130)) : P(0);
 const P v132 = ((true && !v57) && !v87) ? (P(v131)) : P(0);
 const float v133 = ((true && !v57) && !v87) ? (float(0.15f)) : float(0);
 const P v134 = ((true && !v57) && !v87) ? (P(v3 * v133)) : P(0);
 const P v135 = ((true && !v57) && !v87) ? (v132 < v134 ? v132 : v134) : P(0);
 const P v136 = ((true && !v57) && !v87) ? (P(v135)) : P(0);
 const P v137 = ((true && !v57) && !v87) ? (P(v53 * v47)) : P(0);
 const P v138 = ((true && !v57) && !v87) ? (P(v45 + v137)) : P(0);
 const P v139 = ((true && !v57) && !v87) ? (P(v53 * v138)) : P(0);
 const P v140 = ((true && !v57) && !v87) ? (P(v43 + v139)) : P(0);
 const P v141 = ((true && !v57) && !v87) ? (P(v53 * v140)) : P(0);
 const P v142 = ((true && !v57) && !v87) ? (P(v41 + v141)) : P(0);
 const P v143 = ((true && !v57) && !v87) ? (P(v53 * v142)) : P(0);
 const P v144 = ((true && !v57) && !v87) ? (P(v39 + v143)) : P(0);
 const P v145 = ((true && !v57) && !v87) ? (P(v53 * v144)) : P(0);
 const P v146 = ((true && !v57) && !v87) ? (P(v37 + v145)) : P(0);
 const P v147 = ((true && !v57) && !v87) ? (P(v53 * v146)) : P(0);
 const P v148 = ((true && !v57) && !v87) ? (P(v35 + v147)) : P(0);
 const P v149 = ((true && !v57) && !v87) ? (P(v53 * v148)) : P(0);
 const P v150 = ((true && !v57) && !v87) ? (P(v33 + v149)) : P(0);
 const P v151 = ((true && !v57) && !v87) ? (P(v53 * v150)) : P(0);
 const P v152 = ((true && !v57) && !v87) ? (P(v31 + v151)) : P(0);
 const P v153 = ((true && !v57) && !v87) ? (P(v152)) : P(0);
 const float v154 = ((true && !v57) && !v87) ? (float(0.15f)) : float(0);
 const P v155 = ((true && !v57) && !v87) ? (P(v3 * v154)) : P(0);
 const P v156 = ((true && !v57) && !v87) ? (v153 < v155 ? v153 : v155) : P(0);
 const P v157 = ((true && !v57) && !v87) ? (P(v156)) : P(0);
 const float v158 = ((true && !v57) && !v87) ? (float(0.622f)) : float(0);
 const P v159 = ((true && !v57) && !v87) ? (P(v158 * v136)) : P(0);
 const P v160 = ((true && !v57) && !v87) ? (P(v3 - v136)) : P(0);
 const float v161 = ((true && !v57) && !v87) ? (float(1e-5f)) : float(0);
 const P v162 = ((true && !v57) && !v87) ? (v160 > v161 ? v160 : v161) : P(0);
 const P v163 = ((true && !v57) && !v87) ? (P(v159 / v162)) : P(0);
 const P v164 = ((true && !v57) && !v87) ? (P(v163)) : P(0);
 const float v165 = ((true && !v57) && !v87) ? (float(0.622f)) : float(0);
 const P v166 = ((true && !v57) && !v87) ? (P(v165 * v157)) : P(0);
 const P v167 = ((true && !v57) && !v87) ? (P(v3 - v157)) : P(0);
 const float v168 = ((true && !v57) && !v87) ? (float(1e-5f)) : float(0);
 const P v169 = ((true && !v57) && !v87) ? (v167 > v168 ? v167 : v168) : P(0);
 const P v170 = ((true && !v57) && !v87) ? (P(v166 / v169)) : P(0);
 const P v171 = ((true && !v57) && !v87) ? (P(v170)) : P(0);
 const float v172 = ((true && !v57) && !v87) ? (float(t0c)) : float(0);
 const float v173 = ((true && !v57) && !v87) ? (float(6.f)) : float(0);
 const float v174 = ((true && !v57) && !v87) ? (float(v172 - v173)) : float(0);
 const P v175 = ((true && !v57) && !v87) ? (P(v174 - v10)) : P(0);
 const float v176 = ((true && !v57) && !v87) ? (float(t0c)) : float(0);
 const float v177 = ((true && !v57) && !v87) ? (float(6.f)) : float(0);
 const float v178 = ((true && !v57) && !v87) ? (float(v176 - v177)) : float(0);
 const float v179 = ((true && !v57) && !v87) ? (float(tice)) : float(0);
 const float v180 = ((true && !v57) && !v87) ? (float(v178 - v179)) : float(0);
 const P v181 = ((true && !v57) && !v87) ? (P(v175 / v180)) : P(0);
 const P v182 = ((true && !v57) && !v87) ? (P(v181)) : P(0);
 const float v183 = ((true && !v57) && !v87) ? (float(1.f)) : float(0);
 const P v184 = ((true && !v57) && !v87) ? (P(v183 - v182)) : P(0);
 const P v185 = ((true && !v57) && !v87) ? (P(v184 * v164)) : P(0);
 const P v186 = ((true && !v57) && !v87) ? (P(v182 * v171)) : P(0);
 const P v187 = ((true && !v57) && !v87) ? (P(v185 + v186)) : P(0);
 const P v188 = ((true && !v57) && !v87) ? (P(v187)) : P(0);
 const P v194 = (true && !v57) ? (v87 ? v115 : v188) : P(0);
 const P v204 = v57 ? v85 : v194;
 const P v209 = P(v204);
 const P v210 = P(v209);
 const float v211 = float(p5);
 const P v212 = P(v211 * v4);
 const float v213 = float(p5);
 const float v214 = float(zero);
 const P v215 = P(v2 - v210);
 const P v216 = v214 > v215 ? v214 : v215;
 const P v217 = P(v213 * v216);
 const P v218 = P(v212 + v217);
 const P v219 = P(v218);
 double b0=0;
 double b1=0;
 double b2=0;
 double b3=0;
 double b4=0;
 double b5=0;
 double b7=0;
 double b8=0;
 double b9=0;
 double b10=0;
 double b12=0;
 double b14=0;
 double b16=0;
 double b18=0;
 double b20=0;
 double b22=0;
 double b24=0;
 double b26=0;
 double b28=0;
 double b29=0;
 double b31=0;
 double b33=0;
 double b35=0;
 double b37=0;
 double b39=0;
 double b41=0;
 double b43=0;
 double b45=0;
 double b47=0;
 double b49=0;
 double b51=0;
 double b52=0;
 double b53=0;
 double b58=0;
 double b59=0;
 double b60=0;
 double b61=0;
 double b62=0;
 double b63=0;
 double b64=0;
 double b65=0;
 double b66=0;
 double b67=0;
 double b68=0;
 double b69=0;
 double b70=0;
 double b71=0;
 double b72=0;
 double b73=0;
 double b74=0;
 double b76=0;
 double b77=0;
 double b78=0;
 double b80=0;
 double b81=0;
 double b83=0;
 double b84=0;
 double b85=0;
 double b88=0;
 double b89=0;
 double b90=0;
 double b91=0;
 double b92=0;
 double b93=0;
 double b94=0;
 double b95=0;
 double b96=0;
 double b97=0;
 double b98=0;
 double b99=0;
 double b100=0;
 double b101=0;
 double b102=0;
 double b103=0;
 double b104=0;
 double b106=0;
 double b107=0;
 double b108=0;
 double b110=0;
 double b111=0;
 double b113=0;
 double b114=0;
 double b115=0;
 double b116=0;
 double b117=0;
 double b118=0;
 double b119=0;
 double b120=0;
 double b121=0;
 double b122=0;
 double b123=0;
 double b124=0;
 double b125=0;
 double b126=0;
 double b127=0;
 double b128=0;
 double b129=0;
 double b130=0;
 double b131=0;
 double b132=0;
 double b134=0;
 double b135=0;
 double b136=0;
 double b137=0;
 double b138=0;
 double b139=0;
 double b140=0;
 double b141=0;
 double b142=0;
 double b143=0;
 double b144=0;
 double b145=0;
 double b146=0;
 double b147=0;
 double b148=0;
 double b149=0;
 double b150=0;
 double b151=0;
 double b152=0;
 double b153=0;
 double b155=0;
 double b156=0;
 double b157=0;
 double b159=0;
 double b160=0;
 double b162=0;
 double b163=0;
 double b164=0;
 double b166=0;
 double b167=0;
 double b169=0;
 double b170=0;
 double b171=0;
 double b174=0;
 double b175=0;
 double b178=0;
 double b180=0;
 double b181=0;
 double b182=0;
 double b184=0;
 double b185=0;
 double b186=0;
 double b187=0;
 double b188=0;
 double b194=0;
 double b204=0;
 double b209=0;
 double b210=0;
 double b212=0;
 double b215=0;
 double b216=0;
 double b217=0;
 double b218=0;
 double b219=0;
 b219 += seed[0]; if(y)y[0]=v219;
 acc(b218, 1.0, b219);
 acc(b212, 1.0, b218);
 acc(b217, 1.0, b218);
 acc(b216, double(v213), b217);
 if(v214 > v215){}else{acc(b215, 1.0, b216);}
 acc(b2, 1.0, b215);
 acc(b210, -1.0, b215);
 acc(b4, double(v211), b212);
 acc(b209, 1.0, b210);
 acc(b204, 1.0, b209);
 if(v57){acc(b85, 1.0, b204);}else{acc(b194, 1.0, b204);}
 if((true && !v57)){if(v87){acc(b115, 1.0, b194);}else{acc(b188, 1.0, b194);}}
 if(((true && !v57) && !v87)){acc(b187, 1.0, b188);}
 if(((true && !v57) && !v87)){acc(b185, 1.0, b187);acc(b186, 1.0, b187);}
 if(((true && !v57) && !v87)){acc(b182, double(v171), b186);acc(b171, double(v182), b186);}
 if(((true && !v57) && !v87)){acc(b184, double(v164), b185);acc(b164, double(v184), b185);}
 if(((true && !v57) && !v87)){acc(b182, -1.0, b184);}
 if(((true && !v57) && !v87)){acc(b181, 1.0, b182);}
 if(((true && !v57) && !v87)){acc(b175, 1.0/double(v180), b181);acc(b180, -(double(v175)/double(v180))/double(v180), b181);}
 if(((true && !v57) && !v87)){acc(b178, 1.0, b180);}
 if(((true && !v57) && !v87)){}
 if(((true && !v57) && !v87)){acc(b174, 1.0, b175);acc(b10, -1.0, b175);}
 if(((true && !v57) && !v87)){}
 if(((true && !v57) && !v87)){acc(b170, 1.0, b171);}
 if(((true && !v57) && !v87)){acc(b166, 1.0/double(v169), b170);acc(b169, -(double(v166)/double(v169))/double(v169), b170);}
 if(((true && !v57) && !v87)){if(v167 > v168){acc(b167, 1.0, b169);}else{}}
 if(((true && !v57) && !v87)){acc(b3, 1.0, b167);acc(b157, -1.0, b167);}
 if(((true && !v57) && !v87)){acc(b157, double(v165), b166);}
 if(((true && !v57) && !v87)){acc(b163, 1.0, b164);}
 if(((true && !v57) && !v87)){acc(b159, 1.0/double(v162), b163);acc(b162, -(double(v159)/double(v162))/double(v162), b163);}
 if(((true && !v57) && !v87)){if(v160 > v161){acc(b160, 1.0, b162);}else{}}
 if(((true && !v57) && !v87)){acc(b3, 1.0, b160);acc(b136, -1.0, b160);}
 if(((true && !v57) && !v87)){acc(b136, double(v158), b159);}
 if(((true && !v57) && !v87)){acc(b156, 1.0, b157);}
 if(((true && !v57) && !v87)){if(v153 < v155){acc(b153, 1.0, b156);}else{acc(b155, 1.0, b156);}}
 if(((true && !v57) && !v87)){acc(b3, double(v154), b155);}
 if(((true && !v57) && !v87)){acc(b152, 1.0, b153);}
 if(((true && !v57) && !v87)){acc(b31, 1.0, b152);acc(b151, 1.0, b152);}
 if(((true && !v57) && !v87)){acc(b53, double(v150), b151);acc(b150, double(v53), b151);}
 if(((true && !v57) && !v87)){acc(b33, 1.0, b150);acc(b149, 1.0, b150);}
 if(((true && !v57) && !v87)){acc(b53, double(v148), b149);acc(b148, double(v53), b149);}
 if(((true && !v57) && !v87)){acc(b35, 1.0, b148);acc(b147, 1.0, b148);}
 if(((true && !v57) && !v87)){acc(b53, double(v146), b147);acc(b146, double(v53), b147);}
 if(((true && !v57) && !v87)){acc(b37, 1.0, b146);acc(b145, 1.0, b146);}
 if(((true && !v57) && !v87)){acc(b53, double(v144), b145);acc(b144, double(v53), b145);}
 if(((true && !v57) && !v87)){acc(b39, 1.0, b144);acc(b143, 1.0, b144);}
 if(((true && !v57) && !v87)){acc(b53, double(v142), b143);acc(b142, double(v53), b143);}
 if(((true && !v57) && !v87)){acc(b41, 1.0, b142);acc(b141, 1.0, b142);}
 if(((true && !v57) && !v87)){acc(b53, double(v140), b141);acc(b140, double(v53), b141);}
 if(((true && !v57) && !v87)){acc(b43, 1.0, b140);acc(b139, 1.0, b140);}
 if(((true && !v57) && !v87)){acc(b53, double(v138), b139);acc(b138, double(v53), b139);}
 if(((true && !v57) && !v87)){acc(b45, 1.0, b138);acc(b137, 1.0, b138);}
 if(((true && !v57) && !v87)){acc(b53, double(v47), b137);acc(b47, double(v53), b137);}
 if(((true && !v57) && !v87)){acc(b135, 1.0, b136);}
 if(((true && !v57) && !v87)){if(v132 < v134){acc(b132, 1.0, b135);}else{acc(b134, 1.0, b135);}}
 if(((true && !v57) && !v87)){acc(b3, double(v133), b134);}
 if(((true && !v57) && !v87)){acc(b131, 1.0, b132);}
 if(((true && !v57) && !v87)){acc(b12, 1.0, b131);acc(b130, 1.0, b131);}
 if(((true && !v57) && !v87)){acc(b53, double(v129), b130);acc(b129, double(v53), b130);}
 if(((true && !v57) && !v87)){acc(b14, 1.0, b129);acc(b128, 1.0, b129);}
 if(((true && !v57) && !v87)){acc(b53, double(v127), b128);acc(b127, double(v53), b128);}
 if(((true && !v57) && !v87)){acc(b16, 1.0, b127);acc(b126, 1.0, b127);}
 if(((true && !v57) && !v87)){acc(b53, double(v125), b126);acc(b125, double(v53), b126);}
 if(((true && !v57) && !v87)){acc(b18, 1.0, b125);acc(b124, 1.0, b125);}
 if(((true && !v57) && !v87)){acc(b53, double(v123), b124);acc(b123, double(v53), b124);}
 if(((true && !v57) && !v87)){acc(b20, 1.0, b123);acc(b122, 1.0, b123);}
 if(((true && !v57) && !v87)){acc(b53, double(v121), b122);acc(b121, double(v53), b122);}
 if(((true && !v57) && !v87)){acc(b22, 1.0, b121);acc(b120, 1.0, b121);}
 if(((true && !v57) && !v87)){acc(b53, double(v119), b120);acc(b119, double(v53), b120);}
 if(((true && !v57) && !v87)){acc(b24, 1.0, b119);acc(b118, 1.0, b119);}
 if(((true && !v57) && !v87)){acc(b53, double(v117), b118);acc(b117, double(v53), b118);}
 if(((true && !v57) && !v87)){acc(b26, 1.0, b117);acc(b116, 1.0, b117);}
 if(((true && !v57) && !v87)){acc(b53, double(v29), b116);acc(b29, double(v53), b116);}
 if(((true && !v57) && v87)){acc(b114, 1.0, b115);}
 if(((true && !v57) && v87)){acc(b110, 1.0/double(v113), b114);acc(b113, -(double(v110)/double(v113))/double(v113), b114);}
 if(((true && !v57) && v87)){if(v111 > v112){acc(b111, 1.0, b113);}else{}}
 if(((true && !v57) && v87)){acc(b3, 1.0, b111);acc(b108, -1.0, b111);}
 if(((true && !v57) && v87)){acc(b108, double(v109), b110);}
 if(((true && !v57) && v87)){acc(b107, 1.0, b108);}
 if(((true && !v57) && v87)){if(v104 < v106){acc(b104, 1.0, b107);}else{acc(b106, 1.0, b107);}}
 if(((true && !v57) && v87)){acc(b3, double(v105), b106);}
 if(((true && !v57) && v87)){acc(b103, 1.0, b104);}
 if(((true && !v57) && v87)){acc(b31, 1.0, b103);acc(b102, 1.0, b103);}
 if(((true && !v57) && v87)){acc(b53, double(v101), b102);acc(b101, double(v53), b102);}
 if(((true && !v57) && v87)){acc(b33, 1.0, b101);acc(b100, 1.0, b101);}
 if(((true && !v57) && v87)){acc(b53, double(v99), b100);acc(b99, double(v53), b100);}
 if(((true && !v57) && v87)){acc(b35, 1.0, b99);acc(b98, 1.0, b99);}
 if(((true && !v57) && v87)){acc(b53, double(v97), b98);acc(b97, double(v53), b98);}
 if(((true && !v57) && v87)){acc(b37, 1.0, b97);acc(b96, 1.0, b97);}
 if(((true && !v57) && v87)){acc(b53, double(v95), b96);acc(b95, double(v53), b96);}
 if(((true && !v57) && v87)){acc(b39, 1.0, b95);acc(b94, 1.0, b95);}
 if(((true && !v57) && v87)){acc(b53, double(v93), b94);acc(b93, double(v53), b94);}
 if(((true && !v57) && v87)){acc(b41, 1.0, b93);acc(b92, 1.0, b93);}
 if(((true && !v57) && v87)){acc(b53, double(v91), b92);acc(b91, double(v53), b92);}
 if(((true && !v57) && v87)){acc(b43, 1.0, b91);acc(b90, 1.0, b91);}
 if(((true && !v57) && v87)){acc(b53, double(v89), b90);acc(b89, double(v53), b90);}
 if(((true && !v57) && v87)){acc(b45, 1.0, b89);acc(b88, 1.0, b89);}
 if(((true && !v57) && v87)){acc(b53, double(v47), b88);acc(b47, double(v53), b88);}
 if((true && v57)){acc(b84, 1.0, b85);}
 if((true && v57)){acc(b80, 1.0/double(v83), b84);acc(b83, -(double(v80)/double(v83))/double(v83), b84);}
 if((true && v57)){if(v81 > v82){acc(b81, 1.0, b83);}else{}}
 if((true && v57)){acc(b3, 1.0, b81);acc(b78, -1.0, b81);}
 if((true && v57)){acc(b78, double(v79), b80);}
 if((true && v57)){acc(b77, 1.0, b78);}
 if((true && v57)){if(v74 < v76){acc(b74, 1.0, b77);}else{acc(b76, 1.0, b77);}}
 if((true && v57)){acc(b3, double(v75), b76);}
 if((true && v57)){acc(b73, 1.0, b74);}
 if((true && v57)){acc(b12, 1.0, b73);acc(b72, 1.0, b73);}
 if((true && v57)){acc(b53, double(v71), b72);acc(b71, double(v53), b72);}
 if((true && v57)){acc(b14, 1.0, b71);acc(b70, 1.0, b71);}
 if((true && v57)){acc(b53, double(v69), b70);acc(b69, double(v53), b70);}
 if((true && v57)){acc(b16, 1.0, b69);acc(b68, 1.0, b69);}
 if((true && v57)){acc(b53, double(v67), b68);acc(b67, double(v53), b68);}
 if((true && v57)){acc(b18, 1.0, b67);acc(b66, 1.0, b67);}
 if((true && v57)){acc(b53, double(v65), b66);acc(b65, double(v53), b66);}
 if((true && v57)){acc(b20, 1.0, b65);acc(b64, 1.0, b65);}
 if((true && v57)){acc(b53, double(v63), b64);acc(b63, double(v53), b64);}
 if((true && v57)){acc(b22, 1.0, b63);acc(b62, 1.0, b63);}
 if((true && v57)){acc(b53, double(v61), b62);acc(b61, double(v53), b62);}
 if((true && v57)){acc(b24, 1.0, b61);acc(b60, 1.0, b61);}
 if((true && v57)){acc(b53, double(v59), b60);acc(b59, double(v53), b60);}
 if((true && v57)){acc(b26, 1.0, b59);acc(b58, 1.0, b59);}
 if((true && v57)){acc(b53, double(v29), b58);acc(b29, double(v53), b58);}
 acc(b52, 1.0, b53);
 if(v49 > v51){acc(b49, 1.0, b52);}else{acc(b51, 1.0, b52);}
 acc(b10, 1.0, b51);
 acc(b28, 1.0, b29);
 acc(b9, 1.0, b10);
 acc(b5, 1.0, b9);
 acc(b8, 1.0, b9);
 acc(b7, double(v4), b8);
 acc(b4, double(v7), b8);
 acc(b0, -(double(v6)/double(v0))/double(v0), b7);
 acc(b0, double(v1), b5);
 acc(b1, double(v0), b5);
 bar[4] += b4;
 bar[3] += b3;
 bar[2] += b2;
 bar[1] += b1;
 bar[0] += b0*(planted?1.001:1.0);
}
template<class P> inline void cond_step_forward(const P*x,P*y){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = P(v0 * v1);
 const float v6 = float(xlvcp);
 const P v7 = P(v6 / v0);
 const P v8 = P(v7 * v4);
 const P v9 = P(v5 + v8);
 const P v10 = P(v9);
 const float v11 = float(.611583699e03f);
 const P v12 = P(v11);
 const float v13 = float(.444606896e02f);
 const P v14 = P(v13);
 const float v15 = float(.143177157e01f);
 const P v16 = P(v15);
 const float v17 = float(.264224321e-1f);
 const P v18 = P(v17);
 const float v19 = float(.299291081e-3f);
 const P v20 = P(v19);
 const float v21 = float(.203154182e-5f);
 const P v22 = P(v21);
 const float v23 = float(.702620698e-8f);
 const P v24 = P(v23);
 const float v25 = float(.379534310e-11f);
 const P v26 = P(v25);
 const float v27 = float(.321582393e-13f);
 const float v28 = -v27;
 const P v29 = P(v28);
 const float v30 = float(.609868993e03f);
 const P v31 = P(v30);
 const float v32 = float(.499320233e02f);
 const P v33 = P(v32);
 const float v34 = float(.184672631e01f);
 const P v35 = P(v34);
 const float v36 = float(.402737184e-1f);
 const P v37 = P(v36);
 const float v38 = float(.565392987e-3f);
 const P v39 = P(v38);
 const float v40 = float(.521693933e-5f);
 const P v41 = P(v40);
 const float v42 = float(.307839583e-7f);
 const P v43 = P(v42);
 const float v44 = float(.105785160e-9f);
 const P v45 = P(v44);
 const float v46 = float(.161444444e-12f);
 const P v47 = P(v46);
 const float v48 = float(80.f);
 const float v49 = -v48;
 const float v50 = float(t0c);
 const P v51 = P(v10 - v50);
 const P v52 = v49 > v51 ? v49 : v51;
 const P v53 = P(v52);
 const float v54 = float(t0c);
 const float v55 = float(6.f);
 const float v56 = float(v54 - v55);
 const bool v57 = v10 >= v56;
 const P v58 = (true && v57) ? (P(v53 * v29)) : P(0);
 const P v59 = (true && v57) ? (P(v26 + v58)) : P(0);
 const P v60 = (true && v57) ? (P(v53 * v59)) : P(0);
 const P v61 = (true && v57) ? (P(v24 + v60)) : P(0);
 const P v62 = (true && v57) ? (P(v53 * v61)) : P(0);
 const P v63 = (true && v57) ? (P(v22 + v62)) : P(0);
 const P v64 = (true && v57) ? (P(v53 * v63)) : P(0);
 const P v65 = (true && v57) ? (P(v20 + v64)) : P(0);
 const P v66 = (true && v57) ? (P(v53 * v65)) : P(0);
 const P v67 = (true && v57) ? (P(v18 + v66)) : P(0);
 const P v68 = (true && v57) ? (P(v53 * v67)) : P(0);
 const P v69 = (true && v57) ? (P(v16 + v68)) : P(0);
 const P v70 = (true && v57) ? (P(v53 * v69)) : P(0);
 const P v71 = (true && v57) ? (P(v14 + v70)) : P(0);
 const P v72 = (true && v57) ? (P(v53 * v71)) : P(0);
 const P v73 = (true && v57) ? (P(v12 + v72)) : P(0);
 const P v74 = (true && v57) ? (P(v73)) : P(0);
 const float v75 = (true && v57) ? (float(0.15f)) : float(0);
 const P v76 = (true && v57) ? (P(v3 * v75)) : P(0);
 const P v77 = (true && v57) ? (v74 < v76 ? v74 : v76) : P(0);
 const P v78 = (true && v57) ? (P(v77)) : P(0);
 const float v79 = (true && v57) ? (float(0.622f)) : float(0);
 const P v80 = (true && v57) ? (P(v79 * v78)) : P(0);
 const P v81 = (true && v57) ? (P(v3 - v78)) : P(0);
 const float v82 = (true && v57) ? (float(1e-5f)) : float(0);
 const P v83 = (true && v57) ? (v81 > v82 ? v81 : v82) : P(0);
 const P v84 = (true && v57) ? (P(v80 / v83)) : P(0);
 const P v85 = (true && v57) ? (P(v84)) : P(0);
 const float v86 = (true && !v57) ? (float(tice)) : float(0);
 const bool v87 = (true && !v57) ? (v10 <= v86) : bool(0);
 const P v88 = ((true && !v57) && v87) ? (P(v53 * v47)) : P(0);
 const P v89 = ((true && !v57) && v87) ? (P(v45 + v88)) : P(0);
 const P v90 = ((true && !v57) && v87) ? (P(v53 * v89)) : P(0);
 const P v91 = ((true && !v57) && v87) ? (P(v43 + v90)) : P(0);
 const P v92 = ((true && !v57) && v87) ? (P(v53 * v91)) : P(0);
 const P v93 = ((true && !v57) && v87) ? (P(v41 + v92)) : P(0);
 const P v94 = ((true && !v57) && v87) ? (P(v53 * v93)) : P(0);
 const P v95 = ((true && !v57) && v87) ? (P(v39 + v94)) : P(0);
 const P v96 = ((true && !v57) && v87) ? (P(v53 * v95)) : P(0);
 const P v97 = ((true && !v57) && v87) ? (P(v37 + v96)) : P(0);
 const P v98 = ((true && !v57) && v87) ? (P(v53 * v97)) : P(0);
 const P v99 = ((true && !v57) && v87) ? (P(v35 + v98)) : P(0);
 const P v100 = ((true && !v57) && v87) ? (P(v53 * v99)) : P(0);
 const P v101 = ((true && !v57) && v87) ? (P(v33 + v100)) : P(0);
 const P v102 = ((true && !v57) && v87) ? (P(v53 * v101)) : P(0);
 const P v103 = ((true && !v57) && v87) ? (P(v31 + v102)) : P(0);
 const P v104 = ((true && !v57) && v87) ? (P(v103)) : P(0);
 const float v105 = ((true && !v57) && v87) ? (float(0.15f)) : float(0);
 const P v106 = ((true && !v57) && v87) ? (P(v3 * v105)) : P(0);
 const P v107 = ((true && !v57) && v87) ? (v104 < v106 ? v104 : v106) : P(0);
 const P v108 = ((true && !v57) && v87) ? (P(v107)) : P(0);
 const float v109 = ((true && !v57) && v87) ? (float(0.622f)) : float(0);
 const P v110 = ((true && !v57) && v87) ? (P(v109 * v108)) : P(0);
 const P v111 = ((true && !v57) && v87) ? (P(v3 - v108)) : P(0);
 const float v112 = ((true && !v57) && v87) ? (float(1e-5f)) : float(0);
 const P v113 = ((true && !v57) && v87) ? (v111 > v112 ? v111 : v112) : P(0);
 const P v114 = ((true && !v57) && v87) ? (P(v110 / v113)) : P(0);
 const P v115 = ((true && !v57) && v87) ? (P(v114)) : P(0);
 const P v116 = ((true && !v57) && !v87) ? (P(v53 * v29)) : P(0);
 const P v117 = ((true && !v57) && !v87) ? (P(v26 + v116)) : P(0);
 const P v118 = ((true && !v57) && !v87) ? (P(v53 * v117)) : P(0);
 const P v119 = ((true && !v57) && !v87) ? (P(v24 + v118)) : P(0);
 const P v120 = ((true && !v57) && !v87) ? (P(v53 * v119)) : P(0);
 const P v121 = ((true && !v57) && !v87) ? (P(v22 + v120)) : P(0);
 const P v122 = ((true && !v57) && !v87) ? (P(v53 * v121)) : P(0);
 const P v123 = ((true && !v57) && !v87) ? (P(v20 + v122)) : P(0);
 const P v124 = ((true && !v57) && !v87) ? (P(v53 * v123)) : P(0);
 const P v125 = ((true && !v57) && !v87) ? (P(v18 + v124)) : P(0);
 const P v126 = ((true && !v57) && !v87) ? (P(v53 * v125)) : P(0);
 const P v127 = ((true && !v57) && !v87) ? (P(v16 + v126)) : P(0);
 const P v128 = ((true && !v57) && !v87) ? (P(v53 * v127)) : P(0);
 const P v129 = ((true && !v57) && !v87) ? (P(v14 + v128)) : P(0);
 const P v130 = ((true && !v57) && !v87) ? (P(v53 * v129)) : P(0);
 const P v131 = ((true && !v57) && !v87) ? (P(v12 + v130)) : P(0);
 const P v132 = ((true && !v57) && !v87) ? (P(v131)) : P(0);
 const float v133 = ((true && !v57) && !v87) ? (float(0.15f)) : float(0);
 const P v134 = ((true && !v57) && !v87) ? (P(v3 * v133)) : P(0);
 const P v135 = ((true && !v57) && !v87) ? (v132 < v134 ? v132 : v134) : P(0);
 const P v136 = ((true && !v57) && !v87) ? (P(v135)) : P(0);
 const P v137 = ((true && !v57) && !v87) ? (P(v53 * v47)) : P(0);
 const P v138 = ((true && !v57) && !v87) ? (P(v45 + v137)) : P(0);
 const P v139 = ((true && !v57) && !v87) ? (P(v53 * v138)) : P(0);
 const P v140 = ((true && !v57) && !v87) ? (P(v43 + v139)) : P(0);
 const P v141 = ((true && !v57) && !v87) ? (P(v53 * v140)) : P(0);
 const P v142 = ((true && !v57) && !v87) ? (P(v41 + v141)) : P(0);
 const P v143 = ((true && !v57) && !v87) ? (P(v53 * v142)) : P(0);
 const P v144 = ((true && !v57) && !v87) ? (P(v39 + v143)) : P(0);
 const P v145 = ((true && !v57) && !v87) ? (P(v53 * v144)) : P(0);
 const P v146 = ((true && !v57) && !v87) ? (P(v37 + v145)) : P(0);
 const P v147 = ((true && !v57) && !v87) ? (P(v53 * v146)) : P(0);
 const P v148 = ((true && !v57) && !v87) ? (P(v35 + v147)) : P(0);
 const P v149 = ((true && !v57) && !v87) ? (P(v53 * v148)) : P(0);
 const P v150 = ((true && !v57) && !v87) ? (P(v33 + v149)) : P(0);
 const P v151 = ((true && !v57) && !v87) ? (P(v53 * v150)) : P(0);
 const P v152 = ((true && !v57) && !v87) ? (P(v31 + v151)) : P(0);
 const P v153 = ((true && !v57) && !v87) ? (P(v152)) : P(0);
 const float v154 = ((true && !v57) && !v87) ? (float(0.15f)) : float(0);
 const P v155 = ((true && !v57) && !v87) ? (P(v3 * v154)) : P(0);
 const P v156 = ((true && !v57) && !v87) ? (v153 < v155 ? v153 : v155) : P(0);
 const P v157 = ((true && !v57) && !v87) ? (P(v156)) : P(0);
 const float v158 = ((true && !v57) && !v87) ? (float(0.622f)) : float(0);
 const P v159 = ((true && !v57) && !v87) ? (P(v158 * v136)) : P(0);
 const P v160 = ((true && !v57) && !v87) ? (P(v3 - v136)) : P(0);
 const float v161 = ((true && !v57) && !v87) ? (float(1e-5f)) : float(0);
 const P v162 = ((true && !v57) && !v87) ? (v160 > v161 ? v160 : v161) : P(0);
 const P v163 = ((true && !v57) && !v87) ? (P(v159 / v162)) : P(0);
 const P v164 = ((true && !v57) && !v87) ? (P(v163)) : P(0);
 const float v165 = ((true && !v57) && !v87) ? (float(0.622f)) : float(0);
 const P v166 = ((true && !v57) && !v87) ? (P(v165 * v157)) : P(0);
 const P v167 = ((true && !v57) && !v87) ? (P(v3 - v157)) : P(0);
 const float v168 = ((true && !v57) && !v87) ? (float(1e-5f)) : float(0);
 const P v169 = ((true && !v57) && !v87) ? (v167 > v168 ? v167 : v168) : P(0);
 const P v170 = ((true && !v57) && !v87) ? (P(v166 / v169)) : P(0);
 const P v171 = ((true && !v57) && !v87) ? (P(v170)) : P(0);
 const float v172 = ((true && !v57) && !v87) ? (float(t0c)) : float(0);
 const float v173 = ((true && !v57) && !v87) ? (float(6.f)) : float(0);
 const float v174 = ((true && !v57) && !v87) ? (float(v172 - v173)) : float(0);
 const P v175 = ((true && !v57) && !v87) ? (P(v174 - v10)) : P(0);
 const float v176 = ((true && !v57) && !v87) ? (float(t0c)) : float(0);
 const float v177 = ((true && !v57) && !v87) ? (float(6.f)) : float(0);
 const float v178 = ((true && !v57) && !v87) ? (float(v176 - v177)) : float(0);
 const float v179 = ((true && !v57) && !v87) ? (float(tice)) : float(0);
 const float v180 = ((true && !v57) && !v87) ? (float(v178 - v179)) : float(0);
 const P v181 = ((true && !v57) && !v87) ? (P(v175 / v180)) : P(0);
 const P v182 = ((true && !v57) && !v87) ? (P(v181)) : P(0);
 const float v183 = ((true && !v57) && !v87) ? (float(1.f)) : float(0);
 const P v184 = ((true && !v57) && !v87) ? (P(v183 - v182)) : P(0);
 const P v185 = ((true && !v57) && !v87) ? (P(v184 * v164)) : P(0);
 const P v186 = ((true && !v57) && !v87) ? (P(v182 * v171)) : P(0);
 const P v187 = ((true && !v57) && !v87) ? (P(v185 + v186)) : P(0);
 const P v188 = ((true && !v57) && !v87) ? (P(v187)) : P(0);
 const P v194 = (true && !v57) ? (v87 ? v115 : v188) : P(0);
 const P v204 = v57 ? v85 : v194;
 const P v209 = P(v204);
 const P v210 = P(v209);
 const float v211 = float(p5);
 const P v212 = P(v211 * v4);
 const float v213 = float(p5);
 const float v214 = float(zero);
 const P v215 = P(v2 - v210);
 const P v216 = v214 > v215 ? v214 : v215;
 const P v217 = P(v213 * v216);
 const P v218 = P(v212 + v217);
 const P v219 = P(v218);
y[0]=v219;
}
template<class P> inline void cond_finish_reverse(const P*x,const double*seed,double*bar,P*y=nullptr,bool planted=false){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const P v6 = P(v0 * v1);
 const float v7 = float(xlvcp);
 const P v8 = P(v7 / v0);
 const P v9 = P(v8 * v5);
 const P v10 = P(v6 + v9);
 const P v11 = P(v10);
 const float v12 = float(.611583699e03f);
 const P v13 = P(v12);
 const float v14 = float(.444606896e02f);
 const P v15 = P(v14);
 const float v16 = float(.143177157e01f);
 const P v17 = P(v16);
 const float v18 = float(.264224321e-1f);
 const P v19 = P(v18);
 const float v20 = float(.299291081e-3f);
 const P v21 = P(v20);
 const float v22 = float(.203154182e-5f);
 const P v23 = P(v22);
 const float v24 = float(.702620698e-8f);
 const P v25 = P(v24);
 const float v26 = float(.379534310e-11f);
 const P v27 = P(v26);
 const float v28 = float(.321582393e-13f);
 const float v29 = -v28;
 const P v30 = P(v29);
 const float v31 = float(.609868993e03f);
 const P v32 = P(v31);
 const float v33 = float(.499320233e02f);
 const P v34 = P(v33);
 const float v35 = float(.184672631e01f);
 const P v36 = P(v35);
 const float v37 = float(.402737184e-1f);
 const P v38 = P(v37);
 const float v39 = float(.565392987e-3f);
 const P v40 = P(v39);
 const float v41 = float(.521693933e-5f);
 const P v42 = P(v41);
 const float v43 = float(.307839583e-7f);
 const P v44 = P(v43);
 const float v45 = float(.105785160e-9f);
 const P v46 = P(v45);
 const float v47 = float(.161444444e-12f);
 const P v48 = P(v47);
 const float v49 = float(80.f);
 const float v50 = -v49;
 const float v51 = float(t0c);
 const P v52 = P(v11 - v51);
 const P v53 = v50 > v52 ? v50 : v52;
 const P v54 = P(v53);
 const float v55 = float(t0c);
 const float v56 = float(6.f);
 const float v57 = float(v55 - v56);
 const bool v58 = v11 >= v57;
 const P v59 = (true && v58) ? (P(v54 * v30)) : P(0);
 const P v60 = (true && v58) ? (P(v27 + v59)) : P(0);
 const P v61 = (true && v58) ? (P(v54 * v60)) : P(0);
 const P v62 = (true && v58) ? (P(v25 + v61)) : P(0);
 const P v63 = (true && v58) ? (P(v54 * v62)) : P(0);
 const P v64 = (true && v58) ? (P(v23 + v63)) : P(0);
 const P v65 = (true && v58) ? (P(v54 * v64)) : P(0);
 const P v66 = (true && v58) ? (P(v21 + v65)) : P(0);
 const P v67 = (true && v58) ? (P(v54 * v66)) : P(0);
 const P v68 = (true && v58) ? (P(v19 + v67)) : P(0);
 const P v69 = (true && v58) ? (P(v54 * v68)) : P(0);
 const P v70 = (true && v58) ? (P(v17 + v69)) : P(0);
 const P v71 = (true && v58) ? (P(v54 * v70)) : P(0);
 const P v72 = (true && v58) ? (P(v15 + v71)) : P(0);
 const P v73 = (true && v58) ? (P(v54 * v72)) : P(0);
 const P v74 = (true && v58) ? (P(v13 + v73)) : P(0);
 const P v75 = (true && v58) ? (P(v74)) : P(0);
 const float v76 = (true && v58) ? (float(0.15f)) : float(0);
 const P v77 = (true && v58) ? (P(v3 * v76)) : P(0);
 const P v78 = (true && v58) ? (v75 < v77 ? v75 : v77) : P(0);
 const P v79 = (true && v58) ? (P(v78)) : P(0);
 const float v80 = (true && v58) ? (float(0.622f)) : float(0);
 const P v81 = (true && v58) ? (P(v80 * v79)) : P(0);
 const P v82 = (true && v58) ? (P(v3 - v79)) : P(0);
 const float v83 = (true && v58) ? (float(1e-5f)) : float(0);
 const P v84 = (true && v58) ? (v82 > v83 ? v82 : v83) : P(0);
 const P v85 = (true && v58) ? (P(v81 / v84)) : P(0);
 const P v86 = (true && v58) ? (P(v85)) : P(0);
 const float v87 = (true && !v58) ? (float(tice)) : float(0);
 const bool v88 = (true && !v58) ? (v11 <= v87) : bool(0);
 const P v89 = ((true && !v58) && v88) ? (P(v54 * v48)) : P(0);
 const P v90 = ((true && !v58) && v88) ? (P(v46 + v89)) : P(0);
 const P v91 = ((true && !v58) && v88) ? (P(v54 * v90)) : P(0);
 const P v92 = ((true && !v58) && v88) ? (P(v44 + v91)) : P(0);
 const P v93 = ((true && !v58) && v88) ? (P(v54 * v92)) : P(0);
 const P v94 = ((true && !v58) && v88) ? (P(v42 + v93)) : P(0);
 const P v95 = ((true && !v58) && v88) ? (P(v54 * v94)) : P(0);
 const P v96 = ((true && !v58) && v88) ? (P(v40 + v95)) : P(0);
 const P v97 = ((true && !v58) && v88) ? (P(v54 * v96)) : P(0);
 const P v98 = ((true && !v58) && v88) ? (P(v38 + v97)) : P(0);
 const P v99 = ((true && !v58) && v88) ? (P(v54 * v98)) : P(0);
 const P v100 = ((true && !v58) && v88) ? (P(v36 + v99)) : P(0);
 const P v101 = ((true && !v58) && v88) ? (P(v54 * v100)) : P(0);
 const P v102 = ((true && !v58) && v88) ? (P(v34 + v101)) : P(0);
 const P v103 = ((true && !v58) && v88) ? (P(v54 * v102)) : P(0);
 const P v104 = ((true && !v58) && v88) ? (P(v32 + v103)) : P(0);
 const P v105 = ((true && !v58) && v88) ? (P(v104)) : P(0);
 const float v106 = ((true && !v58) && v88) ? (float(0.15f)) : float(0);
 const P v107 = ((true && !v58) && v88) ? (P(v3 * v106)) : P(0);
 const P v108 = ((true && !v58) && v88) ? (v105 < v107 ? v105 : v107) : P(0);
 const P v109 = ((true && !v58) && v88) ? (P(v108)) : P(0);
 const float v110 = ((true && !v58) && v88) ? (float(0.622f)) : float(0);
 const P v111 = ((true && !v58) && v88) ? (P(v110 * v109)) : P(0);
 const P v112 = ((true && !v58) && v88) ? (P(v3 - v109)) : P(0);
 const float v113 = ((true && !v58) && v88) ? (float(1e-5f)) : float(0);
 const P v114 = ((true && !v58) && v88) ? (v112 > v113 ? v112 : v113) : P(0);
 const P v115 = ((true && !v58) && v88) ? (P(v111 / v114)) : P(0);
 const P v116 = ((true && !v58) && v88) ? (P(v115)) : P(0);
 const P v117 = ((true && !v58) && !v88) ? (P(v54 * v30)) : P(0);
 const P v118 = ((true && !v58) && !v88) ? (P(v27 + v117)) : P(0);
 const P v119 = ((true && !v58) && !v88) ? (P(v54 * v118)) : P(0);
 const P v120 = ((true && !v58) && !v88) ? (P(v25 + v119)) : P(0);
 const P v121 = ((true && !v58) && !v88) ? (P(v54 * v120)) : P(0);
 const P v122 = ((true && !v58) && !v88) ? (P(v23 + v121)) : P(0);
 const P v123 = ((true && !v58) && !v88) ? (P(v54 * v122)) : P(0);
 const P v124 = ((true && !v58) && !v88) ? (P(v21 + v123)) : P(0);
 const P v125 = ((true && !v58) && !v88) ? (P(v54 * v124)) : P(0);
 const P v126 = ((true && !v58) && !v88) ? (P(v19 + v125)) : P(0);
 const P v127 = ((true && !v58) && !v88) ? (P(v54 * v126)) : P(0);
 const P v128 = ((true && !v58) && !v88) ? (P(v17 + v127)) : P(0);
 const P v129 = ((true && !v58) && !v88) ? (P(v54 * v128)) : P(0);
 const P v130 = ((true && !v58) && !v88) ? (P(v15 + v129)) : P(0);
 const P v131 = ((true && !v58) && !v88) ? (P(v54 * v130)) : P(0);
 const P v132 = ((true && !v58) && !v88) ? (P(v13 + v131)) : P(0);
 const P v133 = ((true && !v58) && !v88) ? (P(v132)) : P(0);
 const float v134 = ((true && !v58) && !v88) ? (float(0.15f)) : float(0);
 const P v135 = ((true && !v58) && !v88) ? (P(v3 * v134)) : P(0);
 const P v136 = ((true && !v58) && !v88) ? (v133 < v135 ? v133 : v135) : P(0);
 const P v137 = ((true && !v58) && !v88) ? (P(v136)) : P(0);
 const P v138 = ((true && !v58) && !v88) ? (P(v54 * v48)) : P(0);
 const P v139 = ((true && !v58) && !v88) ? (P(v46 + v138)) : P(0);
 const P v140 = ((true && !v58) && !v88) ? (P(v54 * v139)) : P(0);
 const P v141 = ((true && !v58) && !v88) ? (P(v44 + v140)) : P(0);
 const P v142 = ((true && !v58) && !v88) ? (P(v54 * v141)) : P(0);
 const P v143 = ((true && !v58) && !v88) ? (P(v42 + v142)) : P(0);
 const P v144 = ((true && !v58) && !v88) ? (P(v54 * v143)) : P(0);
 const P v145 = ((true && !v58) && !v88) ? (P(v40 + v144)) : P(0);
 const P v146 = ((true && !v58) && !v88) ? (P(v54 * v145)) : P(0);
 const P v147 = ((true && !v58) && !v88) ? (P(v38 + v146)) : P(0);
 const P v148 = ((true && !v58) && !v88) ? (P(v54 * v147)) : P(0);
 const P v149 = ((true && !v58) && !v88) ? (P(v36 + v148)) : P(0);
 const P v150 = ((true && !v58) && !v88) ? (P(v54 * v149)) : P(0);
 const P v151 = ((true && !v58) && !v88) ? (P(v34 + v150)) : P(0);
 const P v152 = ((true && !v58) && !v88) ? (P(v54 * v151)) : P(0);
 const P v153 = ((true && !v58) && !v88) ? (P(v32 + v152)) : P(0);
 const P v154 = ((true && !v58) && !v88) ? (P(v153)) : P(0);
 const float v155 = ((true && !v58) && !v88) ? (float(0.15f)) : float(0);
 const P v156 = ((true && !v58) && !v88) ? (P(v3 * v155)) : P(0);
 const P v157 = ((true && !v58) && !v88) ? (v154 < v156 ? v154 : v156) : P(0);
 const P v158 = ((true && !v58) && !v88) ? (P(v157)) : P(0);
 const float v159 = ((true && !v58) && !v88) ? (float(0.622f)) : float(0);
 const P v160 = ((true && !v58) && !v88) ? (P(v159 * v137)) : P(0);
 const P v161 = ((true && !v58) && !v88) ? (P(v3 - v137)) : P(0);
 const float v162 = ((true && !v58) && !v88) ? (float(1e-5f)) : float(0);
 const P v163 = ((true && !v58) && !v88) ? (v161 > v162 ? v161 : v162) : P(0);
 const P v164 = ((true && !v58) && !v88) ? (P(v160 / v163)) : P(0);
 const P v165 = ((true && !v58) && !v88) ? (P(v164)) : P(0);
 const float v166 = ((true && !v58) && !v88) ? (float(0.622f)) : float(0);
 const P v167 = ((true && !v58) && !v88) ? (P(v166 * v158)) : P(0);
 const P v168 = ((true && !v58) && !v88) ? (P(v3 - v158)) : P(0);
 const float v169 = ((true && !v58) && !v88) ? (float(1e-5f)) : float(0);
 const P v170 = ((true && !v58) && !v88) ? (v168 > v169 ? v168 : v169) : P(0);
 const P v171 = ((true && !v58) && !v88) ? (P(v167 / v170)) : P(0);
 const P v172 = ((true && !v58) && !v88) ? (P(v171)) : P(0);
 const float v173 = ((true && !v58) && !v88) ? (float(t0c)) : float(0);
 const float v174 = ((true && !v58) && !v88) ? (float(6.f)) : float(0);
 const float v175 = ((true && !v58) && !v88) ? (float(v173 - v174)) : float(0);
 const P v176 = ((true && !v58) && !v88) ? (P(v175 - v11)) : P(0);
 const float v177 = ((true && !v58) && !v88) ? (float(t0c)) : float(0);
 const float v178 = ((true && !v58) && !v88) ? (float(6.f)) : float(0);
 const float v179 = ((true && !v58) && !v88) ? (float(v177 - v178)) : float(0);
 const float v180 = ((true && !v58) && !v88) ? (float(tice)) : float(0);
 const float v181 = ((true && !v58) && !v88) ? (float(v179 - v180)) : float(0);
 const P v182 = ((true && !v58) && !v88) ? (P(v176 / v181)) : P(0);
 const P v183 = ((true && !v58) && !v88) ? (P(v182)) : P(0);
 const float v184 = ((true && !v58) && !v88) ? (float(1.f)) : float(0);
 const P v185 = ((true && !v58) && !v88) ? (P(v184 - v183)) : P(0);
 const P v186 = ((true && !v58) && !v88) ? (P(v185 * v165)) : P(0);
 const P v187 = ((true && !v58) && !v88) ? (P(v183 * v172)) : P(0);
 const P v188 = ((true && !v58) && !v88) ? (P(v186 + v187)) : P(0);
 const P v189 = ((true && !v58) && !v88) ? (P(v188)) : P(0);
 const P v195 = (true && !v58) ? (v88 ? v116 : v189) : P(0);
 const P v205 = v58 ? v86 : v195;
 const P v210 = P(v205);
 const P v211 = P(v210);
 const float v212 = float(zero);
 const P v213 = P(v2 - v211);
 const P v214 = v212 > v213 ? v212 : v213;
 const P v215 = P(v214);
 const float v216 = float(100.f);
 const bool v217 = v4 < v216;
 const float v218 = (true && v217) ? (float(zero)) : float(0);
 const P v219 = (true && v217) ? (P(v218)) : P(0);
 const P v220 = v217 ? v219 : v215;
 const float v221 = float(xlvcp);
 const P v222 = P(v221 / v0);
 const P v223 = P(v222 * v220);
 const P v224 = P(v1 + v223);
 const float v225 = float(one);
 const float v226 = float(rvovrd);
 const float v227 = float(one);
 const float v228 = float(v226 - v227);
 const P v229 = P(v2 * v228);
 const P v230 = P(v225 + v229);
 const float v231 = float(rvovrd);
 const P v232 = P(v231 * v220);
 const P v233 = P(v230 - v232);
 const P v234 = P(v224 * v233);
 const P v235 = P(v234);
 double b0=0;
 double b1=0;
 double b2=0;
 double b3=0;
 double b5=0;
 double b6=0;
 double b8=0;
 double b9=0;
 double b10=0;
 double b11=0;
 double b13=0;
 double b15=0;
 double b17=0;
 double b19=0;
 double b21=0;
 double b23=0;
 double b25=0;
 double b27=0;
 double b29=0;
 double b30=0;
 double b32=0;
 double b34=0;
 double b36=0;
 double b38=0;
 double b40=0;
 double b42=0;
 double b44=0;
 double b46=0;
 double b48=0;
 double b50=0;
 double b52=0;
 double b53=0;
 double b54=0;
 double b59=0;
 double b60=0;
 double b61=0;
 double b62=0;
 double b63=0;
 double b64=0;
 double b65=0;
 double b66=0;
 double b67=0;
 double b68=0;
 double b69=0;
 double b70=0;
 double b71=0;
 double b72=0;
 double b73=0;
 double b74=0;
 double b75=0;
 double b77=0;
 double b78=0;
 double b79=0;
 double b81=0;
 double b82=0;
 double b84=0;
 double b85=0;
 double b86=0;
 double b89=0;
 double b90=0;
 double b91=0;
 double b92=0;
 double b93=0;
 double b94=0;
 double b95=0;
 double b96=0;
 double b97=0;
 double b98=0;
 double b99=0;
 double b100=0;
 double b101=0;
 double b102=0;
 double b103=0;
 double b104=0;
 double b105=0;
 double b107=0;
 double b108=0;
 double b109=0;
 double b111=0;
 double b112=0;
 double b114=0;
 double b115=0;
 double b116=0;
 double b117=0;
 double b118=0;
 double b119=0;
 double b120=0;
 double b121=0;
 double b122=0;
 double b123=0;
 double b124=0;
 double b125=0;
 double b126=0;
 double b127=0;
 double b128=0;
 double b129=0;
 double b130=0;
 double b131=0;
 double b132=0;
 double b133=0;
 double b135=0;
 double b136=0;
 double b137=0;
 double b138=0;
 double b139=0;
 double b140=0;
 double b141=0;
 double b142=0;
 double b143=0;
 double b144=0;
 double b145=0;
 double b146=0;
 double b147=0;
 double b148=0;
 double b149=0;
 double b150=0;
 double b151=0;
 double b152=0;
 double b153=0;
 double b154=0;
 double b156=0;
 double b157=0;
 double b158=0;
 double b160=0;
 double b161=0;
 double b163=0;
 double b164=0;
 double b165=0;
 double b167=0;
 double b168=0;
 double b170=0;
 double b171=0;
 double b172=0;
 double b175=0;
 double b176=0;
 double b179=0;
 double b181=0;
 double b182=0;
 double b183=0;
 double b185=0;
 double b186=0;
 double b187=0;
 double b188=0;
 double b189=0;
 double b195=0;
 double b205=0;
 double b210=0;
 double b211=0;
 double b213=0;
 double b214=0;
 double b215=0;
 double b219=0;
 double b220=0;
 double b222=0;
 double b223=0;
 double b224=0;
 double b228=0;
 double b229=0;
 double b230=0;
 double b232=0;
 double b233=0;
 double b234=0;
 double b235=0;
 b235 += seed[0]; if(y)y[0]=v235;
 b220 += seed[1]; if(y)y[1]=v220;
 acc(b234, 1.0, b235);
 acc(b224, double(v233), b234);
 acc(b233, double(v224), b234);
 acc(b230, 1.0, b233);
 acc(b232, -1.0, b233);
 acc(b220, double(v231), b232);
 acc(b229, 1.0, b230);
 acc(b2, double(v228), b229);
 acc(b228, double(v2), b229);
 acc(b1, 1.0, b224);
 acc(b223, 1.0, b224);
 acc(b222, double(v220), b223);
 acc(b220, double(v222), b223);
 acc(b0, -(double(v221)/double(v0))/double(v0), b222);
 if(v217){acc(b219, 1.0, b220);}else{acc(b215, 1.0, b220);}
 if((true && v217)){}
 acc(b214, 1.0, b215);
 if(v212 > v213){}else{acc(b213, 1.0, b214);}
 acc(b2, 1.0, b213);
 acc(b211, -1.0, b213);
 acc(b210, 1.0, b211);
 acc(b205, 1.0, b210);
 if(v58){acc(b86, 1.0, b205);}else{acc(b195, 1.0, b205);}
 if((true && !v58)){if(v88){acc(b116, 1.0, b195);}else{acc(b189, 1.0, b195);}}
 if(((true && !v58) && !v88)){acc(b188, 1.0, b189);}
 if(((true && !v58) && !v88)){acc(b186, 1.0, b188);acc(b187, 1.0, b188);}
 if(((true && !v58) && !v88)){acc(b183, double(v172), b187);acc(b172, double(v183), b187);}
 if(((true && !v58) && !v88)){acc(b185, double(v165), b186);acc(b165, double(v185), b186);}
 if(((true && !v58) && !v88)){acc(b183, -1.0, b185);}
 if(((true && !v58) && !v88)){acc(b182, 1.0, b183);}
 if(((true && !v58) && !v88)){acc(b176, 1.0/double(v181), b182);acc(b181, -(double(v176)/double(v181))/double(v181), b182);}
 if(((true && !v58) && !v88)){acc(b179, 1.0, b181);}
 if(((true && !v58) && !v88)){}
 if(((true && !v58) && !v88)){acc(b175, 1.0, b176);acc(b11, -1.0, b176);}
 if(((true && !v58) && !v88)){}
 if(((true && !v58) && !v88)){acc(b171, 1.0, b172);}
 if(((true && !v58) && !v88)){acc(b167, 1.0/double(v170), b171);acc(b170, -(double(v167)/double(v170))/double(v170), b171);}
 if(((true && !v58) && !v88)){if(v168 > v169){acc(b168, 1.0, b170);}else{}}
 if(((true && !v58) && !v88)){acc(b3, 1.0, b168);acc(b158, -1.0, b168);}
 if(((true && !v58) && !v88)){acc(b158, double(v166), b167);}
 if(((true && !v58) && !v88)){acc(b164, 1.0, b165);}
 if(((true && !v58) && !v88)){acc(b160, 1.0/double(v163), b164);acc(b163, -(double(v160)/double(v163))/double(v163), b164);}
 if(((true && !v58) && !v88)){if(v161 > v162){acc(b161, 1.0, b163);}else{}}
 if(((true && !v58) && !v88)){acc(b3, 1.0, b161);acc(b137, -1.0, b161);}
 if(((true && !v58) && !v88)){acc(b137, double(v159), b160);}
 if(((true && !v58) && !v88)){acc(b157, 1.0, b158);}
 if(((true && !v58) && !v88)){if(v154 < v156){acc(b154, 1.0, b157);}else{acc(b156, 1.0, b157);}}
 if(((true && !v58) && !v88)){acc(b3, double(v155), b156);}
 if(((true && !v58) && !v88)){acc(b153, 1.0, b154);}
 if(((true && !v58) && !v88)){acc(b32, 1.0, b153);acc(b152, 1.0, b153);}
 if(((true && !v58) && !v88)){acc(b54, double(v151), b152);acc(b151, double(v54), b152);}
 if(((true && !v58) && !v88)){acc(b34, 1.0, b151);acc(b150, 1.0, b151);}
 if(((true && !v58) && !v88)){acc(b54, double(v149), b150);acc(b149, double(v54), b150);}
 if(((true && !v58) && !v88)){acc(b36, 1.0, b149);acc(b148, 1.0, b149);}
 if(((true && !v58) && !v88)){acc(b54, double(v147), b148);acc(b147, double(v54), b148);}
 if(((true && !v58) && !v88)){acc(b38, 1.0, b147);acc(b146, 1.0, b147);}
 if(((true && !v58) && !v88)){acc(b54, double(v145), b146);acc(b145, double(v54), b146);}
 if(((true && !v58) && !v88)){acc(b40, 1.0, b145);acc(b144, 1.0, b145);}
 if(((true && !v58) && !v88)){acc(b54, double(v143), b144);acc(b143, double(v54), b144);}
 if(((true && !v58) && !v88)){acc(b42, 1.0, b143);acc(b142, 1.0, b143);}
 if(((true && !v58) && !v88)){acc(b54, double(v141), b142);acc(b141, double(v54), b142);}
 if(((true && !v58) && !v88)){acc(b44, 1.0, b141);acc(b140, 1.0, b141);}
 if(((true && !v58) && !v88)){acc(b54, double(v139), b140);acc(b139, double(v54), b140);}
 if(((true && !v58) && !v88)){acc(b46, 1.0, b139);acc(b138, 1.0, b139);}
 if(((true && !v58) && !v88)){acc(b54, double(v48), b138);acc(b48, double(v54), b138);}
 if(((true && !v58) && !v88)){acc(b136, 1.0, b137);}
 if(((true && !v58) && !v88)){if(v133 < v135){acc(b133, 1.0, b136);}else{acc(b135, 1.0, b136);}}
 if(((true && !v58) && !v88)){acc(b3, double(v134), b135);}
 if(((true && !v58) && !v88)){acc(b132, 1.0, b133);}
 if(((true && !v58) && !v88)){acc(b13, 1.0, b132);acc(b131, 1.0, b132);}
 if(((true && !v58) && !v88)){acc(b54, double(v130), b131);acc(b130, double(v54), b131);}
 if(((true && !v58) && !v88)){acc(b15, 1.0, b130);acc(b129, 1.0, b130);}
 if(((true && !v58) && !v88)){acc(b54, double(v128), b129);acc(b128, double(v54), b129);}
 if(((true && !v58) && !v88)){acc(b17, 1.0, b128);acc(b127, 1.0, b128);}
 if(((true && !v58) && !v88)){acc(b54, double(v126), b127);acc(b126, double(v54), b127);}
 if(((true && !v58) && !v88)){acc(b19, 1.0, b126);acc(b125, 1.0, b126);}
 if(((true && !v58) && !v88)){acc(b54, double(v124), b125);acc(b124, double(v54), b125);}
 if(((true && !v58) && !v88)){acc(b21, 1.0, b124);acc(b123, 1.0, b124);}
 if(((true && !v58) && !v88)){acc(b54, double(v122), b123);acc(b122, double(v54), b123);}
 if(((true && !v58) && !v88)){acc(b23, 1.0, b122);acc(b121, 1.0, b122);}
 if(((true && !v58) && !v88)){acc(b54, double(v120), b121);acc(b120, double(v54), b121);}
 if(((true && !v58) && !v88)){acc(b25, 1.0, b120);acc(b119, 1.0, b120);}
 if(((true && !v58) && !v88)){acc(b54, double(v118), b119);acc(b118, double(v54), b119);}
 if(((true && !v58) && !v88)){acc(b27, 1.0, b118);acc(b117, 1.0, b118);}
 if(((true && !v58) && !v88)){acc(b54, double(v30), b117);acc(b30, double(v54), b117);}
 if(((true && !v58) && v88)){acc(b115, 1.0, b116);}
 if(((true && !v58) && v88)){acc(b111, 1.0/double(v114), b115);acc(b114, -(double(v111)/double(v114))/double(v114), b115);}
 if(((true && !v58) && v88)){if(v112 > v113){acc(b112, 1.0, b114);}else{}}
 if(((true && !v58) && v88)){acc(b3, 1.0, b112);acc(b109, -1.0, b112);}
 if(((true && !v58) && v88)){acc(b109, double(v110), b111);}
 if(((true && !v58) && v88)){acc(b108, 1.0, b109);}
 if(((true && !v58) && v88)){if(v105 < v107){acc(b105, 1.0, b108);}else{acc(b107, 1.0, b108);}}
 if(((true && !v58) && v88)){acc(b3, double(v106), b107);}
 if(((true && !v58) && v88)){acc(b104, 1.0, b105);}
 if(((true && !v58) && v88)){acc(b32, 1.0, b104);acc(b103, 1.0, b104);}
 if(((true && !v58) && v88)){acc(b54, double(v102), b103);acc(b102, double(v54), b103);}
 if(((true && !v58) && v88)){acc(b34, 1.0, b102);acc(b101, 1.0, b102);}
 if(((true && !v58) && v88)){acc(b54, double(v100), b101);acc(b100, double(v54), b101);}
 if(((true && !v58) && v88)){acc(b36, 1.0, b100);acc(b99, 1.0, b100);}
 if(((true && !v58) && v88)){acc(b54, double(v98), b99);acc(b98, double(v54), b99);}
 if(((true && !v58) && v88)){acc(b38, 1.0, b98);acc(b97, 1.0, b98);}
 if(((true && !v58) && v88)){acc(b54, double(v96), b97);acc(b96, double(v54), b97);}
 if(((true && !v58) && v88)){acc(b40, 1.0, b96);acc(b95, 1.0, b96);}
 if(((true && !v58) && v88)){acc(b54, double(v94), b95);acc(b94, double(v54), b95);}
 if(((true && !v58) && v88)){acc(b42, 1.0, b94);acc(b93, 1.0, b94);}
 if(((true && !v58) && v88)){acc(b54, double(v92), b93);acc(b92, double(v54), b93);}
 if(((true && !v58) && v88)){acc(b44, 1.0, b92);acc(b91, 1.0, b92);}
 if(((true && !v58) && v88)){acc(b54, double(v90), b91);acc(b90, double(v54), b91);}
 if(((true && !v58) && v88)){acc(b46, 1.0, b90);acc(b89, 1.0, b90);}
 if(((true && !v58) && v88)){acc(b54, double(v48), b89);acc(b48, double(v54), b89);}
 if((true && v58)){acc(b85, 1.0, b86);}
 if((true && v58)){acc(b81, 1.0/double(v84), b85);acc(b84, -(double(v81)/double(v84))/double(v84), b85);}
 if((true && v58)){if(v82 > v83){acc(b82, 1.0, b84);}else{}}
 if((true && v58)){acc(b3, 1.0, b82);acc(b79, -1.0, b82);}
 if((true && v58)){acc(b79, double(v80), b81);}
 if((true && v58)){acc(b78, 1.0, b79);}
 if((true && v58)){if(v75 < v77){acc(b75, 1.0, b78);}else{acc(b77, 1.0, b78);}}
 if((true && v58)){acc(b3, double(v76), b77);}
 if((true && v58)){acc(b74, 1.0, b75);}
 if((true && v58)){acc(b13, 1.0, b74);acc(b73, 1.0, b74);}
 if((true && v58)){acc(b54, double(v72), b73);acc(b72, double(v54), b73);}
 if((true && v58)){acc(b15, 1.0, b72);acc(b71, 1.0, b72);}
 if((true && v58)){acc(b54, double(v70), b71);acc(b70, double(v54), b71);}
 if((true && v58)){acc(b17, 1.0, b70);acc(b69, 1.0, b70);}
 if((true && v58)){acc(b54, double(v68), b69);acc(b68, double(v54), b69);}
 if((true && v58)){acc(b19, 1.0, b68);acc(b67, 1.0, b68);}
 if((true && v58)){acc(b54, double(v66), b67);acc(b66, double(v54), b67);}
 if((true && v58)){acc(b21, 1.0, b66);acc(b65, 1.0, b66);}
 if((true && v58)){acc(b54, double(v64), b65);acc(b64, double(v54), b65);}
 if((true && v58)){acc(b23, 1.0, b64);acc(b63, 1.0, b64);}
 if((true && v58)){acc(b54, double(v62), b63);acc(b62, double(v54), b63);}
 if((true && v58)){acc(b25, 1.0, b62);acc(b61, 1.0, b62);}
 if((true && v58)){acc(b54, double(v60), b61);acc(b60, double(v54), b61);}
 if((true && v58)){acc(b27, 1.0, b60);acc(b59, 1.0, b60);}
 if((true && v58)){acc(b54, double(v30), b59);acc(b30, double(v54), b59);}
 acc(b53, 1.0, b54);
 if(v50 > v52){acc(b50, 1.0, b53);}else{acc(b52, 1.0, b53);}
 acc(b11, 1.0, b52);
 acc(b29, 1.0, b30);
 acc(b10, 1.0, b11);
 acc(b6, 1.0, b10);
 acc(b9, 1.0, b10);
 acc(b8, double(v5), b9);
 acc(b5, double(v8), b9);
 acc(b0, -(double(v7)/double(v0))/double(v0), b8);
 acc(b0, double(v1), b6);
 acc(b1, double(v0), b6);
 bar[5] += b5;
 bar[3] += b3;
 bar[2] += b2;
 bar[1] += b1;
 bar[0] += b0*(planted?1.001:1.0);
}
template<class P> inline void cond_finish_forward(const P*x,P*y){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const P v6 = P(v0 * v1);
 const float v7 = float(xlvcp);
 const P v8 = P(v7 / v0);
 const P v9 = P(v8 * v5);
 const P v10 = P(v6 + v9);
 const P v11 = P(v10);
 const float v12 = float(.611583699e03f);
 const P v13 = P(v12);
 const float v14 = float(.444606896e02f);
 const P v15 = P(v14);
 const float v16 = float(.143177157e01f);
 const P v17 = P(v16);
 const float v18 = float(.264224321e-1f);
 const P v19 = P(v18);
 const float v20 = float(.299291081e-3f);
 const P v21 = P(v20);
 const float v22 = float(.203154182e-5f);
 const P v23 = P(v22);
 const float v24 = float(.702620698e-8f);
 const P v25 = P(v24);
 const float v26 = float(.379534310e-11f);
 const P v27 = P(v26);
 const float v28 = float(.321582393e-13f);
 const float v29 = -v28;
 const P v30 = P(v29);
 const float v31 = float(.609868993e03f);
 const P v32 = P(v31);
 const float v33 = float(.499320233e02f);
 const P v34 = P(v33);
 const float v35 = float(.184672631e01f);
 const P v36 = P(v35);
 const float v37 = float(.402737184e-1f);
 const P v38 = P(v37);
 const float v39 = float(.565392987e-3f);
 const P v40 = P(v39);
 const float v41 = float(.521693933e-5f);
 const P v42 = P(v41);
 const float v43 = float(.307839583e-7f);
 const P v44 = P(v43);
 const float v45 = float(.105785160e-9f);
 const P v46 = P(v45);
 const float v47 = float(.161444444e-12f);
 const P v48 = P(v47);
 const float v49 = float(80.f);
 const float v50 = -v49;
 const float v51 = float(t0c);
 const P v52 = P(v11 - v51);
 const P v53 = v50 > v52 ? v50 : v52;
 const P v54 = P(v53);
 const float v55 = float(t0c);
 const float v56 = float(6.f);
 const float v57 = float(v55 - v56);
 const bool v58 = v11 >= v57;
 const P v59 = (true && v58) ? (P(v54 * v30)) : P(0);
 const P v60 = (true && v58) ? (P(v27 + v59)) : P(0);
 const P v61 = (true && v58) ? (P(v54 * v60)) : P(0);
 const P v62 = (true && v58) ? (P(v25 + v61)) : P(0);
 const P v63 = (true && v58) ? (P(v54 * v62)) : P(0);
 const P v64 = (true && v58) ? (P(v23 + v63)) : P(0);
 const P v65 = (true && v58) ? (P(v54 * v64)) : P(0);
 const P v66 = (true && v58) ? (P(v21 + v65)) : P(0);
 const P v67 = (true && v58) ? (P(v54 * v66)) : P(0);
 const P v68 = (true && v58) ? (P(v19 + v67)) : P(0);
 const P v69 = (true && v58) ? (P(v54 * v68)) : P(0);
 const P v70 = (true && v58) ? (P(v17 + v69)) : P(0);
 const P v71 = (true && v58) ? (P(v54 * v70)) : P(0);
 const P v72 = (true && v58) ? (P(v15 + v71)) : P(0);
 const P v73 = (true && v58) ? (P(v54 * v72)) : P(0);
 const P v74 = (true && v58) ? (P(v13 + v73)) : P(0);
 const P v75 = (true && v58) ? (P(v74)) : P(0);
 const float v76 = (true && v58) ? (float(0.15f)) : float(0);
 const P v77 = (true && v58) ? (P(v3 * v76)) : P(0);
 const P v78 = (true && v58) ? (v75 < v77 ? v75 : v77) : P(0);
 const P v79 = (true && v58) ? (P(v78)) : P(0);
 const float v80 = (true && v58) ? (float(0.622f)) : float(0);
 const P v81 = (true && v58) ? (P(v80 * v79)) : P(0);
 const P v82 = (true && v58) ? (P(v3 - v79)) : P(0);
 const float v83 = (true && v58) ? (float(1e-5f)) : float(0);
 const P v84 = (true && v58) ? (v82 > v83 ? v82 : v83) : P(0);
 const P v85 = (true && v58) ? (P(v81 / v84)) : P(0);
 const P v86 = (true && v58) ? (P(v85)) : P(0);
 const float v87 = (true && !v58) ? (float(tice)) : float(0);
 const bool v88 = (true && !v58) ? (v11 <= v87) : bool(0);
 const P v89 = ((true && !v58) && v88) ? (P(v54 * v48)) : P(0);
 const P v90 = ((true && !v58) && v88) ? (P(v46 + v89)) : P(0);
 const P v91 = ((true && !v58) && v88) ? (P(v54 * v90)) : P(0);
 const P v92 = ((true && !v58) && v88) ? (P(v44 + v91)) : P(0);
 const P v93 = ((true && !v58) && v88) ? (P(v54 * v92)) : P(0);
 const P v94 = ((true && !v58) && v88) ? (P(v42 + v93)) : P(0);
 const P v95 = ((true && !v58) && v88) ? (P(v54 * v94)) : P(0);
 const P v96 = ((true && !v58) && v88) ? (P(v40 + v95)) : P(0);
 const P v97 = ((true && !v58) && v88) ? (P(v54 * v96)) : P(0);
 const P v98 = ((true && !v58) && v88) ? (P(v38 + v97)) : P(0);
 const P v99 = ((true && !v58) && v88) ? (P(v54 * v98)) : P(0);
 const P v100 = ((true && !v58) && v88) ? (P(v36 + v99)) : P(0);
 const P v101 = ((true && !v58) && v88) ? (P(v54 * v100)) : P(0);
 const P v102 = ((true && !v58) && v88) ? (P(v34 + v101)) : P(0);
 const P v103 = ((true && !v58) && v88) ? (P(v54 * v102)) : P(0);
 const P v104 = ((true && !v58) && v88) ? (P(v32 + v103)) : P(0);
 const P v105 = ((true && !v58) && v88) ? (P(v104)) : P(0);
 const float v106 = ((true && !v58) && v88) ? (float(0.15f)) : float(0);
 const P v107 = ((true && !v58) && v88) ? (P(v3 * v106)) : P(0);
 const P v108 = ((true && !v58) && v88) ? (v105 < v107 ? v105 : v107) : P(0);
 const P v109 = ((true && !v58) && v88) ? (P(v108)) : P(0);
 const float v110 = ((true && !v58) && v88) ? (float(0.622f)) : float(0);
 const P v111 = ((true && !v58) && v88) ? (P(v110 * v109)) : P(0);
 const P v112 = ((true && !v58) && v88) ? (P(v3 - v109)) : P(0);
 const float v113 = ((true && !v58) && v88) ? (float(1e-5f)) : float(0);
 const P v114 = ((true && !v58) && v88) ? (v112 > v113 ? v112 : v113) : P(0);
 const P v115 = ((true && !v58) && v88) ? (P(v111 / v114)) : P(0);
 const P v116 = ((true && !v58) && v88) ? (P(v115)) : P(0);
 const P v117 = ((true && !v58) && !v88) ? (P(v54 * v30)) : P(0);
 const P v118 = ((true && !v58) && !v88) ? (P(v27 + v117)) : P(0);
 const P v119 = ((true && !v58) && !v88) ? (P(v54 * v118)) : P(0);
 const P v120 = ((true && !v58) && !v88) ? (P(v25 + v119)) : P(0);
 const P v121 = ((true && !v58) && !v88) ? (P(v54 * v120)) : P(0);
 const P v122 = ((true && !v58) && !v88) ? (P(v23 + v121)) : P(0);
 const P v123 = ((true && !v58) && !v88) ? (P(v54 * v122)) : P(0);
 const P v124 = ((true && !v58) && !v88) ? (P(v21 + v123)) : P(0);
 const P v125 = ((true && !v58) && !v88) ? (P(v54 * v124)) : P(0);
 const P v126 = ((true && !v58) && !v88) ? (P(v19 + v125)) : P(0);
 const P v127 = ((true && !v58) && !v88) ? (P(v54 * v126)) : P(0);
 const P v128 = ((true && !v58) && !v88) ? (P(v17 + v127)) : P(0);
 const P v129 = ((true && !v58) && !v88) ? (P(v54 * v128)) : P(0);
 const P v130 = ((true && !v58) && !v88) ? (P(v15 + v129)) : P(0);
 const P v131 = ((true && !v58) && !v88) ? (P(v54 * v130)) : P(0);
 const P v132 = ((true && !v58) && !v88) ? (P(v13 + v131)) : P(0);
 const P v133 = ((true && !v58) && !v88) ? (P(v132)) : P(0);
 const float v134 = ((true && !v58) && !v88) ? (float(0.15f)) : float(0);
 const P v135 = ((true && !v58) && !v88) ? (P(v3 * v134)) : P(0);
 const P v136 = ((true && !v58) && !v88) ? (v133 < v135 ? v133 : v135) : P(0);
 const P v137 = ((true && !v58) && !v88) ? (P(v136)) : P(0);
 const P v138 = ((true && !v58) && !v88) ? (P(v54 * v48)) : P(0);
 const P v139 = ((true && !v58) && !v88) ? (P(v46 + v138)) : P(0);
 const P v140 = ((true && !v58) && !v88) ? (P(v54 * v139)) : P(0);
 const P v141 = ((true && !v58) && !v88) ? (P(v44 + v140)) : P(0);
 const P v142 = ((true && !v58) && !v88) ? (P(v54 * v141)) : P(0);
 const P v143 = ((true && !v58) && !v88) ? (P(v42 + v142)) : P(0);
 const P v144 = ((true && !v58) && !v88) ? (P(v54 * v143)) : P(0);
 const P v145 = ((true && !v58) && !v88) ? (P(v40 + v144)) : P(0);
 const P v146 = ((true && !v58) && !v88) ? (P(v54 * v145)) : P(0);
 const P v147 = ((true && !v58) && !v88) ? (P(v38 + v146)) : P(0);
 const P v148 = ((true && !v58) && !v88) ? (P(v54 * v147)) : P(0);
 const P v149 = ((true && !v58) && !v88) ? (P(v36 + v148)) : P(0);
 const P v150 = ((true && !v58) && !v88) ? (P(v54 * v149)) : P(0);
 const P v151 = ((true && !v58) && !v88) ? (P(v34 + v150)) : P(0);
 const P v152 = ((true && !v58) && !v88) ? (P(v54 * v151)) : P(0);
 const P v153 = ((true && !v58) && !v88) ? (P(v32 + v152)) : P(0);
 const P v154 = ((true && !v58) && !v88) ? (P(v153)) : P(0);
 const float v155 = ((true && !v58) && !v88) ? (float(0.15f)) : float(0);
 const P v156 = ((true && !v58) && !v88) ? (P(v3 * v155)) : P(0);
 const P v157 = ((true && !v58) && !v88) ? (v154 < v156 ? v154 : v156) : P(0);
 const P v158 = ((true && !v58) && !v88) ? (P(v157)) : P(0);
 const float v159 = ((true && !v58) && !v88) ? (float(0.622f)) : float(0);
 const P v160 = ((true && !v58) && !v88) ? (P(v159 * v137)) : P(0);
 const P v161 = ((true && !v58) && !v88) ? (P(v3 - v137)) : P(0);
 const float v162 = ((true && !v58) && !v88) ? (float(1e-5f)) : float(0);
 const P v163 = ((true && !v58) && !v88) ? (v161 > v162 ? v161 : v162) : P(0);
 const P v164 = ((true && !v58) && !v88) ? (P(v160 / v163)) : P(0);
 const P v165 = ((true && !v58) && !v88) ? (P(v164)) : P(0);
 const float v166 = ((true && !v58) && !v88) ? (float(0.622f)) : float(0);
 const P v167 = ((true && !v58) && !v88) ? (P(v166 * v158)) : P(0);
 const P v168 = ((true && !v58) && !v88) ? (P(v3 - v158)) : P(0);
 const float v169 = ((true && !v58) && !v88) ? (float(1e-5f)) : float(0);
 const P v170 = ((true && !v58) && !v88) ? (v168 > v169 ? v168 : v169) : P(0);
 const P v171 = ((true && !v58) && !v88) ? (P(v167 / v170)) : P(0);
 const P v172 = ((true && !v58) && !v88) ? (P(v171)) : P(0);
 const float v173 = ((true && !v58) && !v88) ? (float(t0c)) : float(0);
 const float v174 = ((true && !v58) && !v88) ? (float(6.f)) : float(0);
 const float v175 = ((true && !v58) && !v88) ? (float(v173 - v174)) : float(0);
 const P v176 = ((true && !v58) && !v88) ? (P(v175 - v11)) : P(0);
 const float v177 = ((true && !v58) && !v88) ? (float(t0c)) : float(0);
 const float v178 = ((true && !v58) && !v88) ? (float(6.f)) : float(0);
 const float v179 = ((true && !v58) && !v88) ? (float(v177 - v178)) : float(0);
 const float v180 = ((true && !v58) && !v88) ? (float(tice)) : float(0);
 const float v181 = ((true && !v58) && !v88) ? (float(v179 - v180)) : float(0);
 const P v182 = ((true && !v58) && !v88) ? (P(v176 / v181)) : P(0);
 const P v183 = ((true && !v58) && !v88) ? (P(v182)) : P(0);
 const float v184 = ((true && !v58) && !v88) ? (float(1.f)) : float(0);
 const P v185 = ((true && !v58) && !v88) ? (P(v184 - v183)) : P(0);
 const P v186 = ((true && !v58) && !v88) ? (P(v185 * v165)) : P(0);
 const P v187 = ((true && !v58) && !v88) ? (P(v183 * v172)) : P(0);
 const P v188 = ((true && !v58) && !v88) ? (P(v186 + v187)) : P(0);
 const P v189 = ((true && !v58) && !v88) ? (P(v188)) : P(0);
 const P v195 = (true && !v58) ? (v88 ? v116 : v189) : P(0);
 const P v205 = v58 ? v86 : v195;
 const P v210 = P(v205);
 const P v211 = P(v210);
 const float v212 = float(zero);
 const P v213 = P(v2 - v211);
 const P v214 = v212 > v213 ? v212 : v213;
 const P v215 = P(v214);
 const float v216 = float(100.f);
 const bool v217 = v4 < v216;
 const float v218 = (true && v217) ? (float(zero)) : float(0);
 const P v219 = (true && v217) ? (P(v218)) : P(0);
 const P v220 = v217 ? v219 : v215;
 const float v221 = float(xlvcp);
 const P v222 = P(v221 / v0);
 const P v223 = P(v222 * v220);
 const P v224 = P(v1 + v223);
 const float v225 = float(one);
 const float v226 = float(rvovrd);
 const float v227 = float(one);
 const float v228 = float(v226 - v227);
 const P v229 = P(v2 * v228);
 const P v230 = P(v225 + v229);
 const float v231 = float(rvovrd);
 const P v232 = P(v231 * v220);
 const P v233 = P(v230 - v232);
 const P v234 = P(v224 * v233);
 const P v235 = P(v234);
y[0]=v235;
y[1]=v220;
}
template<class P> inline void adjustment_reverse(const P*x,const double*seed,double*bar,P*y=nullptr,bool planted=false){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const P v6 = P(std::fabs(v0));
 const float v7 = float(1e-8f);
 const bool v8 = v6 > v7;
 const P v9 = (true && v8) ? (P(v1 - v2)) : P(0);
 const P v10 = (true && v8) ? (P(v0 * v9)) : P(0);
 const P v11 = (true && v8) ? (P(v10 / v3)) : P(0);
 const float v12 = (true && v8) ? (float(1.0e-6f)) : float(0);
 const P v13 = (true && v8) ? (v11 > v12 ? v11 : v12) : P(0);
 const P v14 = (true && v8) ? (P(v13)) : P(0);
 const float v15 = (true && !v8) ? (float(zero)) : float(0);
 const P v16 = (true && !v8) ? (P(v15)) : P(0);
 const P v17 = v8 ? v14 : v16;
 const float v18 = float(one);
 const P v19 = P(v18);
 const float v20 = float(zero);
 const P v21 = v20 > v5 ? v20 : v5;
 const P v22 = P(v21);
 const float v23 = float(0.75f);
 const P v24 = P(v23 * v22);
 const P v25 = P(v24 / v4);
 const bool v26 = v17 > v25;
 const float v27 = float(zero);
 const bool v28 = v17 > v27;
 const bool v29 = v26 && v28;
 const float v30 = (true && v29) ? (float(0.75f)) : float(0);
 const P v31 = (true && v29) ? (P(v30 * v22)) : P(0);
 const P v32 = (true && v29) ? (P(v31 / v4)) : P(0);
 const P v33 = (true && v29) ? (P(v32 / v17)) : P(0);
 const float v34 = (true && v29) ? (float(0.01f)) : float(0);
 const P v35 = (true && v29) ? (v33 > v34 ? v33 : v34) : P(0);
 const P v36 = (true && v29) ? (P(v35)) : P(0);
 const P v37 = v29 ? v36 : v19;
 double b0=0;
 double b1=0;
 double b2=0;
 double b3=0;
 double b4=0;
 double b5=0;
 double b9=0;
 double b10=0;
 double b11=0;
 double b13=0;
 double b14=0;
 double b16=0;
 double b17=0;
 double b19=0;
 double b21=0;
 double b22=0;
 double b31=0;
 double b32=0;
 double b33=0;
 double b35=0;
 double b36=0;
 double b37=0;
 b37 += seed[0]; if(y)y[0]=v37;
 if(v29){acc(b36, 1.0, b37);}else{acc(b19, 1.0, b37);}
 if((true && v29)){acc(b35, 1.0, b36);}
 if((true && v29)){if(v33 > v34){acc(b33, 1.0, b35);}else{}}
 if((true && v29)){acc(b32, 1.0/double(v17), b33);acc(b17, -(double(v32)/double(v17))/double(v17), b33);}
 if((true && v29)){acc(b31, 1.0/double(v4), b32);acc(b4, -(double(v31)/double(v4))/double(v4), b32);}
 if((true && v29)){acc(b22, double(v30), b31);}
 acc(b21, 1.0, b22);
 if(v20 > v5){}else{acc(b5, 1.0, b21);}
 if(v8){acc(b14, 1.0, b17);}else{acc(b16, 1.0, b17);}
 if((true && !v8)){}
 if((true && v8)){acc(b13, 1.0, b14);}
 if((true && v8)){if(v11 > v12){acc(b11, 1.0, b13);}else{}}
 if((true && v8)){acc(b10, 1.0/double(v3), b11);acc(b3, -(double(v10)/double(v3))/double(v3), b11);}
 if((true && v8)){acc(b0, double(v9), b10);acc(b9, double(v0), b10);}
 if((true && v8)){acc(b1, 1.0, b9);acc(b2, -1.0, b9);}
 bar[5] += b5;
 bar[4] += b4;
 bar[3] += b3;
 bar[2] += b2;
 bar[1] += b1;
 bar[0] += b0*(planted?1.001:1.0);
}
template<class P> inline void adjustment_forward(const P*x,P*y){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const P v6 = P(std::fabs(v0));
 const float v7 = float(1e-8f);
 const bool v8 = v6 > v7;
 const P v9 = (true && v8) ? (P(v1 - v2)) : P(0);
 const P v10 = (true && v8) ? (P(v0 * v9)) : P(0);
 const P v11 = (true && v8) ? (P(v10 / v3)) : P(0);
 const float v12 = (true && v8) ? (float(1.0e-6f)) : float(0);
 const P v13 = (true && v8) ? (v11 > v12 ? v11 : v12) : P(0);
 const P v14 = (true && v8) ? (P(v13)) : P(0);
 const float v15 = (true && !v8) ? (float(zero)) : float(0);
 const P v16 = (true && !v8) ? (P(v15)) : P(0);
 const P v17 = v8 ? v14 : v16;
 const float v18 = float(one);
 const P v19 = P(v18);
 const float v20 = float(zero);
 const P v21 = v20 > v5 ? v20 : v5;
 const P v22 = P(v21);
 const float v23 = float(0.75f);
 const P v24 = P(v23 * v22);
 const P v25 = P(v24 / v4);
 const bool v26 = v17 > v25;
 const float v27 = float(zero);
 const bool v28 = v17 > v27;
 const bool v29 = v26 && v28;
 const float v30 = (true && v29) ? (float(0.75f)) : float(0);
 const P v31 = (true && v29) ? (P(v30 * v22)) : P(0);
 const P v32 = (true && v29) ? (P(v31 / v4)) : P(0);
 const P v33 = (true && v29) ? (P(v32 / v17)) : P(0);
 const float v34 = (true && v29) ? (float(0.01f)) : float(0);
 const P v35 = (true && v29) ? (v33 > v34 ? v33 : v34) : P(0);
 const P v36 = (true && v29) ? (P(v35)) : P(0);
 const P v37 = v29 ? v36 : v19;
y[0]=v37;
}
template<class P> inline void normalize_reverse(const P*x,const double*seed,double*bar,P*y=nullptr,bool planted=false){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const P v6 = x[6];
 const float v7 = float(0.f);
 const bool v8 = v0 > v7;
 const P v9 = (true && v8) ? (P(v1 / v0)) : P(0);
 const P v10 = (true && v8) ? (P(v9)) : P(0);
 const P v11 = (true && v8) ? (P(v2 / v0)) : P(0);
 const P v12 = (true && v8) ? (P(v11)) : P(0);
 const P v13 = (true && v8) ? (P(v3 / v0)) : P(0);
 const P v14 = (true && v8) ? (P(v13)) : P(0);
 const P v15 = (true && v8) ? (P(v4 / v0)) : P(0);
 const P v16 = (true && v8) ? (P(v15)) : P(0);
 const P v17 = (true && v8) ? (P(v5 / v0)) : P(0);
 const P v18 = (true && v8) ? (P(v17)) : P(0);
 const P v19 = (true && v8) ? (P(v0 * v6)) : P(0);
 const P v20 = (true && v8) ? (P(v19)) : P(0);
 const P v21 = v8 ? v20 : v0;
 const P v22 = v8 ? v16 : v4;
 const P v23 = v8 ? v18 : v5;
 const P v24 = v8 ? v12 : v2;
 const P v25 = v8 ? v14 : v3;
 const P v26 = v8 ? v10 : v1;
 double b0=0;
 double b1=0;
 double b2=0;
 double b3=0;
 double b4=0;
 double b5=0;
 double b6=0;
 double b9=0;
 double b10=0;
 double b11=0;
 double b12=0;
 double b13=0;
 double b14=0;
 double b15=0;
 double b16=0;
 double b17=0;
 double b18=0;
 double b19=0;
 double b20=0;
 double b21=0;
 double b22=0;
 double b23=0;
 double b24=0;
 double b25=0;
 double b26=0;
 b21 += seed[0]; if(y)y[0]=v21;
 b26 += seed[1]; if(y)y[1]=v26;
 b24 += seed[2]; if(y)y[2]=v24;
 b25 += seed[3]; if(y)y[3]=v25;
 b22 += seed[4]; if(y)y[4]=v22;
 b23 += seed[5]; if(y)y[5]=v23;
 if(v8){acc(b10, 1.0, b26);}else{acc(b1, 1.0, b26);}
 if(v8){acc(b14, 1.0, b25);}else{acc(b3, 1.0, b25);}
 if(v8){acc(b12, 1.0, b24);}else{acc(b2, 1.0, b24);}
 if(v8){acc(b18, 1.0, b23);}else{acc(b5, 1.0, b23);}
 if(v8){acc(b16, 1.0, b22);}else{acc(b4, 1.0, b22);}
 if(v8){acc(b20, 1.0, b21);}else{acc(b0, 1.0, b21);}
 if((true && v8)){acc(b19, 1.0, b20);}
 if((true && v8)){acc(b0, double(v6), b19);acc(b6, double(v0), b19);}
 if((true && v8)){acc(b17, 1.0, b18);}
 if((true && v8)){acc(b5, 1.0/double(v0), b17);acc(b0, -(double(v5)/double(v0))/double(v0), b17);}
 if((true && v8)){acc(b15, 1.0, b16);}
 if((true && v8)){acc(b4, 1.0/double(v0), b15);acc(b0, -(double(v4)/double(v0))/double(v0), b15);}
 if((true && v8)){acc(b13, 1.0, b14);}
 if((true && v8)){acc(b3, 1.0/double(v0), b13);acc(b0, -(double(v3)/double(v0))/double(v0), b13);}
 if((true && v8)){acc(b11, 1.0, b12);}
 if((true && v8)){acc(b2, 1.0/double(v0), b11);acc(b0, -(double(v2)/double(v0))/double(v0), b11);}
 if((true && v8)){acc(b9, 1.0, b10);}
 if((true && v8)){acc(b1, 1.0/double(v0), b9);acc(b0, -(double(v1)/double(v0))/double(v0), b9);}
 bar[6] += b6;
 bar[5] += b5;
 bar[4] += b4;
 bar[3] += b3;
 bar[2] += b2;
 bar[1] += b1;
 bar[0] += b0*(planted?1.001:1.0);
}
template<class P> inline void normalize_forward(const P*x,P*y){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const P v6 = x[6];
 const float v7 = float(0.f);
 const bool v8 = v0 > v7;
 const P v9 = (true && v8) ? (P(v1 / v0)) : P(0);
 const P v10 = (true && v8) ? (P(v9)) : P(0);
 const P v11 = (true && v8) ? (P(v2 / v0)) : P(0);
 const P v12 = (true && v8) ? (P(v11)) : P(0);
 const P v13 = (true && v8) ? (P(v3 / v0)) : P(0);
 const P v14 = (true && v8) ? (P(v13)) : P(0);
 const P v15 = (true && v8) ? (P(v4 / v0)) : P(0);
 const P v16 = (true && v8) ? (P(v15)) : P(0);
 const P v17 = (true && v8) ? (P(v5 / v0)) : P(0);
 const P v18 = (true && v8) ? (P(v17)) : P(0);
 const P v19 = (true && v8) ? (P(v0 * v6)) : P(0);
 const P v20 = (true && v8) ? (P(v19)) : P(0);
 const P v21 = v8 ? v20 : v0;
 const P v22 = v8 ? v16 : v4;
 const P v23 = v8 ? v18 : v5;
 const P v24 = v8 ? v12 : v2;
 const P v25 = v8 ? v14 : v3;
 const P v26 = v8 ? v10 : v1;
y[0]=v21;
y[1]=v26;
y[2]=v24;
y[3]=v25;
y[4]=v22;
y[5]=v23;
}
template<class P> inline void aggregate_reverse(const P*x,const double*seed,double*bar,P*y=nullptr,bool planted=false){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const P v6 = x[6];
 const P v7 = P(v2 * v5);
 const P v8 = P(v1 * v6);
 const P v9 = P(v7 + v8);
 const P v10 = P(v6 + v5);
 const P v11 = P(v9 / v10);
 const P v12 = P(v11);
 const P v13 = P(v4 * v5);
 const P v14 = P(v3 * v6);
 const P v15 = P(v13 + v14);
 const P v16 = P(v12 * v15);
 const P v17 = P(v6 + v5);
 const P v18 = P(v16 / v17);
 const P v19 = P(v0 + v18);
 const P v20 = P(v19);
 double b0=0;
 double b1=0;
 double b2=0;
 double b3=0;
 double b4=0;
 double b5=0;
 double b6=0;
 double b7=0;
 double b8=0;
 double b9=0;
 double b10=0;
 double b11=0;
 double b12=0;
 double b13=0;
 double b14=0;
 double b15=0;
 double b16=0;
 double b17=0;
 double b18=0;
 double b19=0;
 double b20=0;
 b20 += seed[0]; if(y)y[0]=v20;
 acc(b19, 1.0, b20);
 acc(b0, 1.0, b19);
 acc(b18, 1.0, b19);
 acc(b16, 1.0/double(v17), b18);
 acc(b17, -(double(v16)/double(v17))/double(v17), b18);
 acc(b6, 1.0, b17);
 acc(b5, 1.0, b17);
 acc(b12, double(v15), b16);
 acc(b15, double(v12), b16);
 acc(b13, 1.0, b15);
 acc(b14, 1.0, b15);
 acc(b3, double(v6), b14);
 acc(b6, double(v3), b14);
 acc(b4, double(v5), b13);
 acc(b5, double(v4), b13);
 acc(b11, 1.0, b12);
 acc(b9, 1.0/double(v10), b11);
 acc(b10, -(double(v9)/double(v10))/double(v10), b11);
 acc(b6, 1.0, b10);
 acc(b5, 1.0, b10);
 acc(b7, 1.0, b9);
 acc(b8, 1.0, b9);
 acc(b1, double(v6), b8);
 acc(b6, double(v1), b8);
 acc(b2, double(v5), b7);
 acc(b5, double(v2), b7);
 bar[6] += b6;
 bar[5] += b5;
 bar[4] += b4;
 bar[3] += b3;
 bar[2] += b2;
 bar[1] += b1;
 bar[0] += b0*(planted?1.001:1.0);
}
template<class P> inline void aggregate_forward(const P*x,P*y){
 const P v0 = x[0];
 const P v1 = x[1];
 const P v2 = x[2];
 const P v3 = x[3];
 const P v4 = x[4];
 const P v5 = x[5];
 const P v6 = x[6];
 const P v7 = P(v2 * v5);
 const P v8 = P(v1 * v6);
 const P v9 = P(v7 + v8);
 const P v10 = P(v6 + v5);
 const P v11 = P(v9 / v10);
 const P v12 = P(v11);
 const P v13 = P(v4 * v5);
 const P v14 = P(v3 * v6);
 const P v15 = P(v13 + v14);
 const P v16 = P(v12 * v15);
 const P v17 = P(v6 + v5);
 const P v18 = P(v16 / v17);
 const P v19 = P(v0 + v18);
 const P v20 = P(v19);
y[0]=v20;
}
}}
