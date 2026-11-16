#pragma once
#include "numeric.hpp"
#include "ess_constants.hpp"
namespace direct {
template<class P, bool REVERSE> inline void ess_thetae(const P*x,P*y,const double*s,double*b,int planted=0){
using namespace ess_const;
const P v0 = x[0];
const P v1 = x[1];
const P v2 = x[2];
const P v3 = r6_mul(v0,v1);
const P v7 = P(r6_sub(P(P(cpv)),P(P(cliq))));
const P v9 = r6_sub(v3,P(t0c));
const P v10 = r6_mul(v7,v9);
const P v11 = r6_add(P(xlv),v10);
const P v15 = P(r6_sub(P(P(cpv)),P(P(cice))));
const P v17 = r6_sub(v3,P(t0c));
const P v18 = r6_mul(v15,v17);
const P v19 = r6_add(P(xls),v18);
const P v21 = r6_sub(P(t0c),v3);
const P v24 = P(r6_sub(P(P(t0c)),P(P(tice))));
const P v25 = r6_div(v21,v24);
const P v27 = r6_sub(P(1.0f),v25);
const P v28 = r6_mul(v27,v11);
const P v29 = r6_mul(v25,v19);
const P v30 = r6_add(v28,v29);
const P v31 = (v3 <= P(tice)) ? v19 : v30;
const P v32 = (v3 >= P(t0c)) ? v11 : v31;
const P v34 = r6_sub(v3,P(t0c));
const P v36 = mx(P(-80.0f),v34);
const P v39 = r6_mul(v36,P(-3.21582393e-14f));
const P v40 = r6_add(P(3.7953431e-12f),v39);
const P v42 = r6_mul(v36,v40);
const P v43 = r6_add(P(7.02620698e-09f),v42);
const P v45 = r6_mul(v36,v43);
const P v46 = r6_add(P(2.03154182e-06f),v45);
const P v48 = r6_mul(v36,v46);
const P v49 = r6_add(P(0.000299291081f),v48);
const P v51 = r6_mul(v36,v49);
const P v52 = r6_add(P(0.0264224321f),v51);
const P v54 = r6_mul(v36,v52);
const P v55 = r6_add(P(1.43177157f),v54);
const P v57 = r6_mul(v36,v55);
const P v58 = r6_add(P(44.4606896f),v57);
const P v60 = r6_mul(v36,v58);
const P v61 = r6_add(P(611.583699f),v60);
const P v63 = r6_mul(v2,P(0.15f));
const P v64 = mn(v61,v63);
const P v67 = r6_mul(v36,P(1.61444444e-13f));
const P v68 = r6_add(P(1.0578516e-10f),v67);
const P v70 = r6_mul(v36,v68);
const P v71 = r6_add(P(3.07839583e-08f),v70);
const P v73 = r6_mul(v36,v71);
const P v74 = r6_add(P(5.21693933e-06f),v73);
const P v76 = r6_mul(v36,v74);
const P v77 = r6_add(P(0.000565392987f),v76);
const P v79 = r6_mul(v36,v77);
const P v80 = r6_add(P(0.0402737184f),v79);
const P v82 = r6_mul(v36,v80);
const P v83 = r6_add(P(1.84672631f),v82);
const P v85 = r6_mul(v36,v83);
const P v86 = r6_add(P(49.9320233f),v85);
const P v88 = r6_mul(v36,v86);
const P v89 = r6_add(P(609.868993f),v88);
const P v91 = r6_mul(v2,P(0.15f));
const P v92 = mn(v89,v91);
const P v94 = r6_mul(P(0.622f),v64);
const P v95 = r6_sub(v2,v64);
const P v97 = mx(v95,P(1e-05f));
const P v98 = r6_div(v94,v97);
const P v100 = r6_mul(P(0.622f),v92);
const P v101 = r6_sub(v2,v92);
const P v103 = mx(v101,P(1e-05f));
const P v104 = r6_div(v100,v103);
const P v107 = P(r6_sub(P(P(t0c)),P(P(6.0f))));
const P v109 = P(r6_sub(P(v107),P(P(tice))));
const P v110 = r6_sub(v107,v3);
const P v111 = r6_div(v110,v109);
const P v113 = r6_sub(P(1.0f),v111);
const P v114 = r6_mul(v113,v98);
const P v115 = r6_mul(v111,v104);
const P v116 = r6_add(v114,v115);
const P v117 = (v3 <= P(tice)) ? v104 : v116;
const P v118 = (v3 >= v107) ? v98 : v117;
const P v119 = r6_mul(v32,v118);
const P v121 = r6_mul(P(cp),v3);
const P v122 = r6_div(v119,v121);
const P v123 = std::exp(v122);
const P v124 = r6_mul(v0,v123);
if constexpr(!REVERSE){
y[0] = v124;
} else {
double &b0=b[0];
double &b1=b[1];
double &b2=b[2];
double b3=0;
double b9=0;
double b10=0;
double b11=0;
double b17=0;
double b18=0;
double b19=0;
double b21=0;
double b25=0;
double b27=0;
double b28=0;
double b29=0;
double b30=0;
double b31=0;
double b32=0;
double b34=0;
double b36=0;
double b39=0;
double b40=0;
double b42=0;
double b43=0;
double b45=0;
double b46=0;
double b48=0;
double b49=0;
double b51=0;
double b52=0;
double b54=0;
double b55=0;
double b57=0;
double b58=0;
double b60=0;
double b61=0;
double b63=0;
double b64=0;
double b67=0;
double b68=0;
double b70=0;
double b71=0;
double b73=0;
double b74=0;
double b76=0;
double b77=0;
double b79=0;
double b80=0;
double b82=0;
double b83=0;
double b85=0;
double b86=0;
double b88=0;
double b89=0;
double b91=0;
double b92=0;
double b94=0;
double b95=0;
double b97=0;
double b98=0;
double b100=0;
double b101=0;
double b103=0;
double b104=0;
double b110=0;
double b111=0;
double b113=0;
double b114=0;
double b115=0;
double b116=0;
double b117=0;
double b118=0;
double b119=0;
double b121=0;
double b122=0;
double b123=0;
double b124=0;
b124 += s[0];
acc(b0, double(v123), b124);
acc(b123, double(v0), b124);
acc(b122, double(v123), b123);
const double tq122=(1.0/double(v121))*b122;
acc(b119, 1, tq122);
acc(b121, double(v119)/double(v121), -tq122);
acc(b3, double(P(cp)), b121);
acc(b32, double(v118), b119);
acc(b118, double(v32), b119);
if(v3 >= v107){
acc(b98, 1, b118);
} else {
acc(b117, 1, b118);
}
if(v3 <= P(tice)){
acc(b104, 1, b117);
} else {
acc(b116, 1, b117);
}
acc(b114, 1, b116);
acc(b115, 1, b116);
acc(b111, double(v104), b115);
acc(b104, double(v111), b115);
acc(b113, double(v98), b114);
acc(b98, double(v113), b114);
acc(b111, -1, b113);
const double tq111=(1.0/double(v109))*b111;
acc(b110, 1, tq111);
acc(b3, -1, b110);
const double tq104=(1.0/double(v103))*b104;
acc(b100, 1, tq104);
acc(b103, double(v100)/double(v103), -tq104);
if(v101 > P(1e-05f)){
acc(b101, 1, b103);
} else {
}
acc(b2, 1, b101);
acc(b92, -1, b101);
acc(b92, double(P(0.622f)), b100);
const double tq98=(1.0/double(v97))*b98;
acc(b94, 1, tq98);
acc(b97, double(v94)/double(v97), -tq98);
if(v95 > P(1e-05f)){
acc(b95, 1, b97);
} else {
}
acc(b2, 1, b95);
acc(b64, -1, b95);
acc(b64, double(P(0.622f)), b94);
if(v89 < v91){
acc(b89, 1, b92);
} else {
acc(b91, 1, b92);
}
acc(b2, double(P(0.15f)), b91);
acc(b88, 1, b89);
acc(b36, double(v86), b88);
acc(b86, double(v36), b88);
acc(b85, 1, b86);
acc(b36, double(v83), b85);
acc(b83, double(v36), b85);
acc(b82, 1, b83);
acc(b36, double(v80), b82);
acc(b80, double(v36), b82);
acc(b79, 1, b80);
acc(b36, double(v77), b79);
acc(b77, double(v36), b79);
acc(b76, 1, b77);
acc(b36, double(v74), b76);
acc(b74, double(v36), b76);
acc(b73, 1, b74);
acc(b36, double(v71), b73);
acc(b71, double(v36), b73);
acc(b70, 1, b71);
acc(b36, double(v68), b70);
acc(b68, double(v36), b70);
acc(b67, 1, b68);
acc(b36, double(P(1.61444444e-13f)), b67);
if(v61 < v63){
acc(b61, 1, b64);
} else {
acc(b63, 1, b64);
}
acc(b2, double(P(0.15f)), b63);
acc(b60, 1, b61);
acc(b36, double(v58), b60);
acc(b58, double(v36), b60);
acc(b57, 1, b58);
acc(b36, double(v55), b57);
acc(b55, double(v36), b57);
acc(b54, 1, b55);
acc(b36, double(v52), b54);
acc(b52, double(v36), b54);
acc(b51, 1, b52);
acc(b36, double(v49), b51);
acc(b49, double(v36), b51);
acc(b48, 1, b49);
acc(b36, double(v46), b48);
acc(b46, double(v36), b48);
acc(b45, 1, b46);
acc(b36, double(v43), b45);
acc(b43, double(v36), b45);
acc(b42, 1, b43);
acc(b36, double(v40), b42);
acc(b40, double(v36), b42);
acc(b39, 1, b40);
acc(b36, double(P(-3.21582393e-14f)), b39);
if(P(-80.0f) > v34){
} else {
acc(b34, 1, b36);
}
acc(b3, 1, b34);
if(v3 >= P(t0c)){
acc(b11, 1, b32);
} else {
acc(b31, 1, b32);
}
if(v3 <= P(tice)){
acc(b19, 1, b31);
} else {
acc(b30, 1, b31);
}
acc(b28, 1, b30);
acc(b29, 1, b30);
acc(b25, double(v19), b29);
acc(b19, double(v25), b29);
acc(b27, double(v11), b28);
acc(b11, double(v27), b28);
acc(b25, -1, b27);
const double tq25=(1.0/double(v24))*b25;
acc(b21, 1, tq25);
acc(b3, -1, b21);
acc(b18, 1, b19);
acc(b17, double(v15), b18);
acc(b3, 1, b17);
acc(b10, 1, b11);
acc(b9, double(v7), b10);
acc(b3, 1, b9);
acc(b0, double(v1), b3);
acc(b1, double(v0), b3);
}
}
template<class P, bool REVERSE> inline void ess_edge(const P*x,P*y,const double*s,double*b,int planted=0){
using namespace ess_const;
const P v0 = x[0];
const P v1 = x[1];
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
const P v19 = x[19];
const P v20 = x[20];
const P v21 = x[21];
const P v22 = x[22];
const P v23 = x[23];
const P v24 = x[24];
const P v25 = r6_add(v0,v1);
const P v27 = r6_mul(P(0.5f),v25);
const P v28 = r6_sub(v2,v3);
const P v29 = mynn_ipow<2>(v28);
const P v30 = r6_sub(v4,v5);
const P v31 = mynn_ipow<2>(v30);
const P v32 = r6_add(v29,v31);
const P v33 = mynn_ipow<2>(v27);
const P v34 = r6_div(v32,v33);
const P v35 = r6_sub(v6,v7);
const P v36 = r6_div(v35,v27);
const P v37 = r6_sub(v8,v9);
const P v38 = r6_div(v37,v27);
const P v40 = r6_add(v20,P(100.0f));
const P v41 = r6_sub(v18,v40);
const P v43 = mx(v41,P(0.0f));
const P v45 = r6_mul(P(0.3f),v20);
const P v47 = mx(v45,P(400.0f));
const P v48 = r6_div(v43,v47);
const P v50 = mn(v48,P(1.0f));
const P v52 = r6_sub(P(1.0f),v50);
const P v53 = r6_mul(v19,v52);
const P v55 = r6_mul(P(0.2f),v50);
const P v56 = r6_add(v53,v55);
const P v58 = r6_sub(v18,P(20.0f));
const P v60 = mx(v58,P(0.0f));
const P v62 = r6_div(v60,P(200.0f));
const P v64 = mn(v62,P(1.0f));
const P v65 = r6_mul(v56,v64);
const P v66 = r6_add(v21,v22);
const P v68 = r6_mul(P(0.5f),v66);
const P v69 = r6_add(v23,v24);
const P v71 = r6_mul(P(0.5f),v69);
const P v72 = r6_mul(v68,v71);
const P v74 = mn(v72,P(0.2f));
const P v76 = mx(P(0.0f),v74);
const P v77 = r6_add(v14,v15);
const P v79 = r6_mul(P(0.5f),v77);
const P v81 = mx(v79,P(qkemin));
const P v83 = r6_mul(P(0.25f),v81);
const P v84 = std::sqrt(v83);
const P v86 = mn(v84,P(0.5f));
const P v88 = r6_mul(v86,P(0.2f));
const P v89 = r6_add(v16,v17);
const P v91 = r6_mul(P(0.5f),v89);
const P v93 = r6_sub(v91,P(1.0f));
const P v94 = mynn_ipow<2>(v93);
const P v96 = r6_mul(P(2.7f),v94);
const P v98 = mn(v96,P(1.0f));
const P v100 = mx(v98,P(0.01f));
const P v101 = r6_mul(v91,v100);
const P v103 = mn(v101,P(0.1f));
const P v105 = r6_mul(v103,P(10.0f));
const P v106 = mx(v76,v88);
const P v107 = r6_mul(v65,v106);
const P v108 = r6_mul(v107,v105);
const P v109 = r6_sub(v10,v11);
const P v110 = r6_div(v109,v27);
const P v111 = r6_sub(v12,v13);
const P v112 = r6_div(v111,v27);
const P v114 = mn(v112,P(0.0f));
const P v115 = r6_mul(v108,v114);
const P v116 = r6_add(v110,v115);
const P v118 = r6_mul(v116,P(gtr));
const P v119 = r6_neg(v118);
if constexpr(!REVERSE){
y[0] = v36;
y[1] = v38;
y[2] = v116;
y[3] = v34;
y[4] = v119;
} else {
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
double b25=0;
double b27=0;
double b28=0;
double b29=0;
double b30=0;
double b31=0;
double b32=0;
double b33=0;
double b34=0;
double b35=0;
double b36=0;
double b37=0;
double b38=0;
double b41=0;
double b43=0;
double b48=0;
double b50=0;
double b52=0;
double b53=0;
double b55=0;
double b56=0;
double b58=0;
double b60=0;
double b62=0;
double b64=0;
double b65=0;
double b77=0;
double b79=0;
double b81=0;
double b83=0;
double b84=0;
double b86=0;
double b88=0;
double b89=0;
double b91=0;
double b93=0;
double b94=0;
double b96=0;
double b98=0;
double b100=0;
double b101=0;
double b103=0;
double b105=0;
double b106=0;
double b107=0;
double b108=0;
double b109=0;
double b110=0;
double b111=0;
double b112=0;
double b114=0;
double b115=0;
double b116=0;
double b118=0;
double b119=0;
b36 += s[0];
b38 += s[1];
b116 += s[2];
b34 += s[3];
b119 += s[4];
acc(b118, -1, b119);
acc(b116, double(P(gtr)), b118);
acc(b110, 1, b116);
acc(b115, 1, b116);
acc(b108, double(v114), b115);
acc(b114, double(v108), b115);
if(v112 < P(0.0f)){
acc(b112, 1, b114);
} else {
}
const double tq112=(1.0/double(v27))*b112;
acc(b111, 1, tq112);
acc(b27, double(v111)/double(v27), -tq112);
acc(b12, 1, b111);
acc(b13, -1, b111);
const double tq110=(1.0/double(v27))*b110;
acc(b109, 1, tq110);
acc(b27, double(v109)/double(v27), -tq110);
acc(b10, 1, b109);
acc(b11, -1, b109);
acc(b107, double(v105), b108);
acc(b105, double(v107), b108);
acc(b65, double(v106), b107);
acc(b106, double(v65), b107);
if(v76 > v88){
} else {
acc(b88, 1, b106);
}
acc(b103, double(P(10.0f)), b105);
if(v101 < P(0.1f)){
acc(b101, 1, b103);
} else {
}
acc(b91, double(v100), b101);
acc(b100, double(v91), b101);
if(v98 > P(0.01f)){
acc(b98, 1, b100);
} else {
}
if(v96 < P(1.0f)){
acc(b96, 1, b98);
} else {
}
acc(b94, double(P(2.7f)), b96);
acc(b93, 2.0*double(v93), b94);
acc(b91, 1, b93);
acc(b89, double(P(0.5f)), b91);
acc(b16, 1, b89);
acc(b17, 1, b89);
acc(b86, double(P(0.2f)), b88);
if(v84 < P(0.5f)){
acc(b84, 1, b86);
} else {
}
acc(b83, 1.0/(2.0*double(v84)), b84);
acc(b81, double(P(0.25f)), b83);
if(v79 > P(qkemin)){
acc(b79, 1, b81);
} else {
}
acc(b77, double(P(0.5f)), b79);
acc(b14, 1, b77);
acc(b15, 1, b77);
acc(b56, double(v64), b65);
acc(b64, double(v56), b65);
if(v62 < P(1.0f)){
acc(b62, 1, b64);
} else {
}
const double tq62=(1.0/double(P(200.0f)))*b62;
acc(b60, 1, tq62);
if(v58 > P(0.0f)){
acc(b58, 1, b60);
} else {
}
acc(b18, 1, b58);
acc(b53, 1, b56);
acc(b55, 1, b56);
acc(b50, double(P(0.2f)), b55);
acc(b19, double(v52), b53);
acc(b52, double(v19), b53);
acc(b50, -1, b52);
if(v48 < P(1.0f)){
acc(b48, 1, b50);
} else {
}
const double tq48=(1.0/double(v47))*b48;
acc(b43, 1, tq48);
if(v41 > P(0.0f)){
acc(b41, 1, b43);
} else {
}
acc(b18, 1, b41);
const double tq38=(1.0/double(v27))*b38;
acc(b37, 1, tq38);
acc(b27, double(v37)/double(v27), -tq38);
acc(b8, 1, b37);
acc(b9, -1, b37);
const double tq36=(1.0/double(v27))*b36;
acc(b35, 1, tq36);
acc(b27, double(v35)/double(v27), -tq36);
acc(b6, 1, b35);
acc(b7, -1, b35);
const double tq34=(1.0/double(v33))*b34;
acc(b32, 1, tq34);
acc(b33, double(v32)/double(v33), -tq34);
acc(b27, 2.0*double(v27), b33);
acc(b29, 1, b32);
acc(b31, 1, b32);
acc(b30, 2.0*double(v30), b31);
acc(b4, 1, b30);
acc(b5, -1, b30);
acc(b28, 2.0*double(v28), b29);
acc(b2, 1, b28);
acc(b3, -1, b28);
acc(b25, double(P(0.5f)), b27);
acc(b0, 1, b25);
acc(b1, 1, b25);
b[0] += b0*(planted?1.001:1.0);
b[1] += b1;
b[2] += b2;
b[3] += b3;
b[4] += b4;
b[5] += b5;
b[6] += b6;
b[7] += b7;
b[8] += b8;
b[9] += b9;
b[10] += b10;
b[11] += b11;
b[12] += b12;
b[13] += b13;
b[14] += b14;
b[15] += b15;
b[16] += b16;
b[17] += b17;
b[18] += b18;
b[19] += b19;
}
}
template<class P, bool REVERSE> inline void ess_clam(const P*x,P*y,const double*s,double*b,int planted=0){
using namespace ess_const;
const P v0 = x[0];
const P v1 = x[1];
const P v2 = x[2];
const P v3 = mynn_ipow<2>(v0);
const P v4 = mynn_ipow<2>(v1);
const P v5 = r6_add(v3,v4);
const P v6 = std::sqrt(v5);
const P v8 = r6_sub(v6,P(15.0f));
const P v10 = mx(v8,P(0.0f));
const P v12 = r6_div(v10,P(20.0f));
const P v14 = mn(v12,P(1.0f));
const P v16 = r6_mul(P(0.95f),v14);
const P v18 = r6_sub(P(1.0f),v16);
const P v20 = r6_mul(P(1.8f),v18);
const P v22 = (v2 >= P(1.5f)) ? v20 : P(0.08f);
if constexpr(!REVERSE){
y[0] = v22;
} else {
double b0=0;
double b1=0;
double b3=0;
double b4=0;
double b5=0;
double b6=0;
double b8=0;
double b10=0;
double b12=0;
double b14=0;
double b16=0;
double b18=0;
double b20=0;
double b22=0;
b22 += s[0];
if(v2 >= P(1.5f)){
acc(b20, 1, b22);
} else {
}
acc(b18, double(P(1.8f)), b20);
acc(b16, -1, b18);
acc(b14, double(P(0.95f)), b16);
if(v12 < P(1.0f)){
acc(b12, 1, b14);
} else {
}
const double tq12=(1.0/double(P(20.0f)))*b12;
acc(b10, 1, tq12);
if(v8 > P(0.0f)){
acc(b8, 1, b10);
} else {
}
acc(b6, 1, b8);
acc(b5, 1.0/(2.0*double(v6)), b6);
acc(b3, 1, b5);
acc(b4, 1, b5);
acc(b1, 2.0*double(v1), b4);
acc(b0, 2.0*double(v0), b3);
b[0] += b0*(planted?1.001:1.0);
b[1] += b1;
}
}
template<class P, bool REVERSE> inline void driver_thermo(const P*x,P*y,const double*s,double*b,int planted=0){
using namespace ess_const;
const P v0 = x[0];
const P v1 = x[1];
const P v2 = x[2];
const P v3 = x[3];
const P v4 = x[4];
const P v5 = x[5];
const P v7 = r6_sub(P(1.0f),v0);
const P v8 = r6_div(v0,v7);
const P v9 = r6_div(v1,v7);
const P v10 = r6_div(v2,v7);
const P v11 = r6_div(v3,v7);
const P v12 = r6_add(v0,v1);
const P v13 = r6_add(v12,v2);
const P v15 = r6_div(P(xlvcp),v5);
const P v16 = r6_mul(v15,v1);
const P v17 = r6_sub(v4,v16);
const P v19 = r6_div(P(xlscp),v5);
const P v20 = r6_mul(v19,v2);
const P v21 = r6_sub(v17,v20);
const P v23 = r6_mul(P(p608),v0);
const P v25 = r6_add(P(1.0f),v23);
const P v26 = r6_add(v1,v2);
const P v27 = r6_sub(v25,v26);
const P v28 = r6_mul(v4,v27);
if constexpr(!REVERSE){
y[0] = v8;
y[1] = v9;
y[2] = v10;
y[3] = v11;
y[4] = v13;
y[5] = v21;
y[6] = v28;
} else {
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
double b12=0;
double b13=0;
double b15=0;
double b16=0;
double b17=0;
double b19=0;
double b20=0;
double b21=0;
double b23=0;
double b25=0;
double b26=0;
double b27=0;
double b28=0;
b8 += s[0];
b9 += s[1];
b10 += s[2];
b11 += s[3];
b13 += s[4];
b21 += s[5];
b28 += s[6];
acc(b4, double(v27), b28);
acc(b27, double(v4), b28);
acc(b25, 1, b27);
acc(b26, -1, b27);
acc(b1, 1, b26);
acc(b2, 1, b26);
acc(b23, 1, b25);
acc(b0, double(P(p608)), b23);
acc(b17, 1, b21);
acc(b20, -1, b21);
acc(b19, double(v2), b20);
acc(b2, double(v19), b20);
const double tq19=(1.0/double(v5))*b19;
acc(b5, double(P(xlscp))/double(v5), -tq19);
acc(b4, 1, b17);
acc(b16, -1, b17);
acc(b15, double(v1), b16);
acc(b1, double(v15), b16);
const double tq15=(1.0/double(v5))*b15;
acc(b5, double(P(xlvcp))/double(v5), -tq15);
acc(b12, 1, b13);
acc(b2, 1, b13);
acc(b0, 1, b12);
acc(b1, 1, b12);
const double tq11=(1.0/double(v7))*b11;
acc(b3, 1, tq11);
acc(b7, double(v3)/double(v7), -tq11);
const double tq10=(1.0/double(v7))*b10;
acc(b2, 1, tq10);
acc(b7, double(v2)/double(v7), -tq10);
const double tq9=(1.0/double(v7))*b9;
acc(b1, 1, tq9);
acc(b7, double(v1)/double(v7), -tq9);
const double tq8=(1.0/double(v7))*b8;
acc(b0, 1, tq8);
acc(b7, double(v0)/double(v7), -tq8);
acc(b0, -1, b7);
b[0] += b0*(planted?1.001:1.0);
b[1] += b1;
b[2] += b2;
b[3] += b3;
b[4] += b4;
b[5] += b5;
}
}
template<class P, bool REVERSE> inline void driver_surface(const P*x,P*y,const double*s,double*b,int planted=0){
using namespace ess_const;
const P v0 = x[0];
const P v1 = x[1];
const P v2 = x[2];
const P v3 = x[3];
const P v4 = x[4];
const P v5 = x[5];
const P v6 = x[6];
const P v7 = x[7];
const P v8 = x[8];
const P v11 = mx(v0,P(1e-08f));
const P v13 = r6_mul(P(0.84f),v11);
const P v15 = r6_add(P(1.0f),v13);
const P v16 = r6_mul(P(cp),v15);
const P v17 = r6_div(v4,v1);
const P v18 = r6_div(v3,v2);
const P v19 = r6_mul(v1,v16);
const P v20 = r6_div(v5,v19);
const P v22 = r6_mul(v17,P(p608));
const P v23 = r6_mul(v22,v18);
const P v24 = r6_add(v20,v23);
const P v27 = P(r6_mul(P(P(karman)),P(P(gtr))));
const P v28 = P(r6_neg(P(v27)));
const P v29 = mynn_ipow<3>(v6);
const P v31 = mx(v29,P(1e-06f));
const P v32 = r6_mul(v28,v24);
const P v33 = r6_div(v32,v31);
const P v34 = r6_mul(v28,v20);
const P v35 = r6_div(v34,v31);
const P v37 = r6_mul(P(0.5f),v7);
const P v38 = r6_mul(v37,v33);
const P v40 = mx(v38,P(-10.0f));
const P v42 = mn(v40,P(10.0f));
const P v44 = (v42 >= P(0)) ? P(0.0f) : v42;
const P v46 = r6_mul(P(cphm_unst),v44);
const P v48 = r6_sub(P(1.0f),v46);
const P v50 = r6_mul(P(3.0f),v42);
const P v52 = r6_add(P(1.0f),v50);
const P v54 = r6_mul(P(3.5f),v42);
const P v56 = r6_add(P(0.95f),v54);
const P v57 = (v8 >= P(1.5f)) ? v52 : v56;
const P v60 = (v8 >= P(1.5f)) ? P(1.0f) : P(0.95f);
const P v62 = std::pow(v48,P(0.25f));
const P v63 = r6_div(v60,v62);
const P v64 = r6_sub(v63,v42);
const P v65 = (v42 >= P(0)) ? v57 : v64;
const P v67 = r6_mul(P(cphh_st),v42);
const P v69 = r6_add(P(1.0f),v67);
const P v71 = r6_mul(P(cphh_unst),v44);
const P v73 = r6_sub(P(1.0f),v71);
const P v74 = std::sqrt(v73);
const P v76 = r6_div(P(1.0f),v74);
const P v77 = (v42 >= P(0)) ? v69 : v76;
if constexpr(!REVERSE){
y[0] = v17;
y[1] = v17;
y[2] = v18;
y[3] = v20;
y[4] = v24;
y[5] = v33;
y[6] = v35;
y[7] = v65;
y[8] = v77;
} else {
double b0=0;
double b1=0;
double b2=0;
double b3=0;
double b4=0;
double b5=0;
double b6=0;
double b7=0;
double b11=0;
double b13=0;
double b15=0;
double b16=0;
double b17=0;
double b18=0;
double b19=0;
double b20=0;
double b22=0;
double b23=0;
double b24=0;
double b29=0;
double b31=0;
double b32=0;
double b33=0;
double b34=0;
double b35=0;
double b37=0;
double b38=0;
double b40=0;
double b42=0;
double b44=0;
double b46=0;
double b48=0;
double b50=0;
double b52=0;
double b54=0;
double b56=0;
double b57=0;
double b62=0;
double b63=0;
double b64=0;
double b65=0;
double b67=0;
double b69=0;
double b71=0;
double b73=0;
double b74=0;
double b76=0;
double b77=0;
b17 += s[0];
b17 += s[1];
b18 += s[2];
b20 += s[3];
b24 += s[4];
b33 += s[5];
b35 += s[6];
b65 += s[7];
b77 += s[8];
if(v42 >= P(0)){
acc(b69, 1, b77);
} else {
acc(b76, 1, b77);
}
const double tq76=(1.0/double(v74))*b76;
acc(b74, double(P(1.0f))/double(v74), -tq76);
acc(b73, 1.0/(2.0*double(v74)), b74);
acc(b71, -1, b73);
acc(b44, double(P(cphh_unst)), b71);
acc(b67, 1, b69);
acc(b42, double(P(cphh_st)), b67);
if(v42 >= P(0)){
acc(b57, 1, b65);
} else {
acc(b64, 1, b65);
}
acc(b63, 1, b64);
acc(b42, -1, b64);
const double tq63=(1.0/double(v62))*b63;
acc(b62, double(v60)/double(v62), -tq63);
acc(b48, double(P(0.25f))*(double(v62)/double(v48)), b62);
if(v8 >= P(1.5f)){
acc(b52, 1, b57);
} else {
acc(b56, 1, b57);
}
acc(b54, 1, b56);
acc(b42, double(P(3.5f)), b54);
acc(b50, 1, b52);
acc(b42, double(P(3.0f)), b50);
acc(b46, -1, b48);
acc(b44, double(P(cphm_unst)), b46);
if(v42 >= P(0)){
} else {
acc(b42, 1, b44);
}
if(v40 < P(10.0f)){
acc(b40, 1, b42);
} else {
}
if(v38 > P(-10.0f)){
acc(b38, 1, b40);
} else {
}
acc(b37, double(v33), b38);
acc(b33, double(v37), b38);
acc(b7, double(P(0.5f)), b37);
const double tq35=(1.0/double(v31))*b35;
acc(b34, 1, tq35);
acc(b31, double(v34)/double(v31), -tq35);
acc(b20, double(v28), b34);
const double tq33=(1.0/double(v31))*b33;
acc(b32, 1, tq33);
acc(b31, double(v32)/double(v31), -tq33);
acc(b24, double(v28), b32);
if(v29 > P(1e-06f)){
acc(b29, 1, b31);
} else {
}
acc(b6, 2.0*double(v6)*double(v6)+double(r6_mul(v6,v6)), b29);
acc(b20, 1, b24);
acc(b23, 1, b24);
acc(b22, double(v18), b23);
acc(b18, double(v22), b23);
acc(b17, double(P(p608)), b22);
const double tq20=(1.0/double(v19))*b20;
acc(b5, 1, tq20);
acc(b19, double(v5)/double(v19), -tq20);
acc(b1, double(v16), b19);
acc(b16, double(v1), b19);
const double tq18=(1.0/double(v2))*b18;
acc(b3, 1, tq18);
acc(b2, double(v3)/double(v2), -tq18);
const double tq17=(1.0/double(v1))*b17;
acc(b4, 1, tq17);
acc(b1, double(v4)/double(v1), -tq17);
acc(b15, double(P(cp)), b16);
acc(b13, 1, b15);
acc(b11, double(P(0.84f)), b13);
if(v0 > P(1e-08f)){
acc(b0, 1, b11);
} else {
}
b[0] += b0*(planted?1.001:1.0);
b[1] += b1;
b[2] += b2;
b[3] += b3;
b[4] += b4;
b[5] += b5;
b[6] += b6;
b[7] += b7;
}
}
template<class P, bool REVERSE> inline void dissipation_level(const P*x,P*y,const double*s,double*b,int planted=0){
using namespace ess_const;
const P v0 = x[0];
const P v1 = x[1];
const P v2 = x[2];
const P v3 = x[3];
const P v5 = std::pow(v0,P(1.5f));
const P v7 = r6_add(v1,v2);
const P v9 = r6_mul(P(0.5f),v7);
const P v11 = mx(v9,P(1.0f));
const P v12 = r6_mul(P(b1),v11);
const P v13 = r6_div(v5,v12);
const P v15 = r6_div(v13,P(cp));
const P v17 = mx(P(0.0f),v15);
const P v19 = mn(v17,P(0.002f));
const P v21 = mx(v3,P(1.0f));
const P v23 = r6_div(P(10000.0f),v21);
const P v24 = r6_neg(v23);
const P v25 = std::exp(v24);
const P v26 = r6_mul(v19,v25);
if constexpr(!REVERSE){
y[0] = v26;
} else {
double b0=0;
double b3=0;
double b5=0;
double b13=0;
double b15=0;
double b17=0;
double b19=0;
double b21=0;
double b23=0;
double b24=0;
double b25=0;
double b26=0;
b26 += s[0];
acc(b19, double(v25), b26);
acc(b25, double(v19), b26);
acc(b24, double(v25), b25);
acc(b23, -1, b24);
const double tq23=(1.0/double(v21))*b23;
acc(b21, double(P(10000.0f))/double(v21), -tq23);
if(v3 > P(1.0f)){
acc(b3, 1, b21);
} else {
}
if(v17 < P(0.002f)){
acc(b17, 1, b19);
} else {
}
if(P(0.0f) > v15){
} else {
acc(b15, 1, b17);
}
const double tq15=(1.0/double(P(cp)))*b15;
acc(b13, 1, tq15);
const double tq13=(1.0/double(v12))*b13;
acc(b5, 1, tq13);
acc(b0, double(P(1.5f))*(double(v5)/double(v0)), b5);
b[0] += b0*(planted?1.001:1.0);
b[3] += b3;
}
}
}
