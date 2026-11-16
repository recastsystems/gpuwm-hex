#pragma once
#include <cmath>
#include <cstddef>
// DIRECT_FROZEN_BAR(x): the bar of a frozen input of a cut (a transport coefficient, a plume flux, a held tendency).
// A block's reverse accumulates it into x. The whole-column join reads none of them back: join.hpp defines
// DIRECT_SINK_FROZEN before its includes, and the accumulation then goes to the reverse's local sink Z, which the compiler
// drops with the arithmetic that fed only it. The live bars' operations and their order are the same in both forms.
#ifdef DIRECT_SINK_FROZEN
#define DIRECT_FROZEN_BAR(x) Z
#define DIRECT_FROZEN_BAR_IS_SINK 1
#else
#define DIRECT_FROZEN_BAR(x) x
#endif
// DIRECT_MIX_BAR(x): the bar of a diffusivity (dfh in the heat solves, dfm/dfh in the six other solves). Frozen as above
// unless JOIN_MIX (the mixing live, join.hpp mix_reverse reads it back).
#if defined(DIRECT_SINK_FROZEN) && !defined(JOIN_MIX)
#define DIRECT_MIX_BAR(x) Z
#else
#define DIRECT_MIX_BAR(x) x
#endif
// DIRECT_PMOM_BAR(x): the bar of a plume momentum term of the wind solves (s_awu/s_awv, sub_u/sub_v, det_u/det_v). Frozen
// as above unless JOIN_PMOM (the plume's momentum live, join.hpp pmom_reverse reads it back). bl-fast-fix, 2026-11-14.
#if defined(DIRECT_SINK_FROZEN) && !defined(JOIN_PMOM)
#define DIRECT_PMOM_BAR(x) Z
#else
#define DIRECT_PMOM_BAR(x) x
#endif
namespace direct {
constexpr int MAX=61;
#ifdef DIRECT_COUNT
inline unsigned long long counts[7]{};
#define DCOUNT(I) (++counts[I])
#else
#define DCOUNT(I) ((void)0)
#endif
template<class P> inline P r6_add(P a,P b){DCOUNT(0);return P(a+b);}
template<class P> inline P r6_sub(P a,P b){DCOUNT(1);return P(a-b);}
template<class P> inline P r6_mul(P a,P b){DCOUNT(2);return P(a*b);}
template<class P> inline P r6_div(P a,P b){DCOUNT(3);return P(a/b);}
template<class P> inline P r6_neg(P a){DCOUNT(4);return -a;}
template<class P> inline P mynn_sqrt(P a){DCOUNT(5);return std::sqrt(a);}
template<int N,class P> inline P mynn_ipow(P a){P t=r6_mul(a,a);if constexpr(N==3)t=r6_mul(t,a);return t;}
inline void acc(double& a,double w,double b){if(w!=0){DCOUNT(6);a+=w*b;}}
template<class P> inline P mx(P a,P b){return a>b?a:b;}
template<class P> inline P mn(P a,P b){return a<b?a:b;}
}
#include <type_traits>
namespace direct {
#define DIRECT_MIX(F,OP,I) template<class A,class B,std::enable_if_t<!std::is_same_v<A,B>,int> =0> inline auto F(A a,B b){DCOUNT(I);using R=std::common_type_t<A,B>;return R(R(a) OP R(b));}
DIRECT_MIX(r6_add,+,0) DIRECT_MIX(r6_sub,-,1) DIRECT_MIX(r6_mul,*,2) DIRECT_MIX(r6_div,/,3)
#undef DIRECT_MIX
}
