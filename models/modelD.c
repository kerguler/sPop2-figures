#include <math.h>
#include "spop2/spop2.h"

#define p_p1_1     0
#define p_p1_2     1
#define p_p1_3     2
#define p_p1_4     3
#define p_p1_5     4
#define p_p2_1     5
#define p_p2_2     6
#define p_p2_3     7
#define p_p2_4     8
#define p_p2_5     9
#define p_p3_1     10
#define p_p3_2     11
#define p_p3_3     12
#define p_p3_4     13
#define p_p3_5     14
#define p_p4_1     15
#define p_p4_2     16
#define p_p4_3     17
#define p_p4_4     18
#define p_p4_5     19
#define p_d1m_1    20
#define p_d1m_2    21
#define p_d1m_3    22
#define p_d1s_1    23
#define p_d2m_1    24
#define p_d2m_2    25
#define p_d2m_3    26
#define p_d2s_1    27
#define p_d3m_1    28
#define p_d3m_2    29
#define p_d3m_3    30
#define p_d3s_1    31
#define p_ph_thr   32
#define p_ph_scale 33
#define p_ph_steep 34

double min(double x, double y) { return(x<y ? x : y); }
double max(double x, double y) { return(x>y ? x : y); }

#define briere1C(T,T0,T1,a) ((T)<=(T0) ? 1e13 : ((T)>=((T0)+(T1)) ? 1e13 : min(1e13, max(1.0, 1.0/(exp(a)*(T)*((T)-(T0))*sqrt((T0)+(T1)-(T)))))))
#define briere1(T,T0,T1,a) (briere1C(273.15+(T),273.15+(T0),(T1),(a)))

#define fundev2(T,T0,T1,M0,M1,Ts) ((M0)+(M1)/(1.0+exp((Ts)*((T0)+(T1)-(T))*((T)-(T0)))))
#define funphoto(P,PT,PS,S) (1.0 + ((PS)/(1.0 + exp((S)*((P)-(PT))))))

void f_ph(double ph, double *p, double *m) {
    *m = funphoto(ph,
                  p[p_ph_thr],
                  p[p_ph_scale],
                  p[p_ph_steep]);
}

void f_p1(double x, double ph, double *p, double *m) {
    *m = fundev2(x,
                 p[p_p1_1],
                 p[p_p1_2],
                 p[p_p1_3],
                 p[p_p1_4],
                 p[p_p1_5]);
}
void f_p2(double x, double ph, double *p, double *m) {
    *m = fundev2(x,
                 p[p_p2_1],
                 p[p_p2_2],
                 p[p_p2_3],
                 p[p_p2_4],
                 p[p_p2_5]);
}
void f_p3(double x, double ph, double *p, double *m) {
    *m = fundev2(x,
                 p[p_p3_1],
                 p[p_p3_2],
                 p[p_p3_3],
                 p[p_p3_4],
                 p[p_p3_5]);
}
void f_p4(double x, double ph, double *p, double *m) {
    *m = fundev2(x,
                 p[p_p4_1],
                 p[p_p4_2],
                 p[p_p4_3],
                 p[p_p4_4],
                 p[p_p4_5]);
}
void f_d1ms(double x, double ph, double *p, double *m, double *s) {
    *m = briere1(x,
                 p[p_d1m_1],
                 p[p_d1m_2],
                 p[p_d1m_3]);
    *s = p[p_d1s_1] * (*m);
}
void f_d2ms(double x, double ph, double *p, double *m, double *s) {
    *m = briere1(x,
                 p[p_d2m_1],
                 p[p_d2m_2],
                 p[p_d2m_3]);
    if ((p[p_ph_thr] > 0) & (p[p_ph_scale] > 0) & (p[p_ph_steep] > 0)) {
        double scl;
        f_ph(ph,p,&scl);
        *m *= scl;
    }
    *s = p[p_d2s_1] * (*m);
}
void f_d3ms(double x, double ph, double *p, double *m, double *s) {
    *m = briere1(x,
                 p[p_d3m_1],
                 p[p_d3m_2],
                 p[p_d3m_3]);
    *s = p[p_d3s_1] * (*m);
}

void getPD(double x, double ph, double *p, double *m) {
    f_p1(  x, ph, p, &m[0]);
    f_p2(  x, ph, p, &m[1]);
    f_p3(  x, ph, p, &m[2]);
    f_p4(  x, ph, p, &m[3]);
    f_d1ms(x, ph, p, &m[4], &m[5]);
    f_d2ms(x, ph, p, &m[6], &m[7]);
    f_d3ms(x, ph, p, &m[8], &m[9]);
    f_ph(     ph, p, &m[10]);
}

void print_out(int tm, spop2 *egg, spop2 *larva, spop2 *pupa, spop *adult, double *ret) {
    ret[tm*8+0] = (*egg)->size.d;
    ret[tm*8+1] = (*larva)->size.d;
    ret[tm*8+2] = (*pupa)->size.d;
    ret[tm*8+3] = (*adult)->size.d;
    ret[tm*8+4] = 0.0;
    ret[tm*8+5] = (*egg)->developed.d;
    ret[tm*8+6] = (*larva)->developed.d;
    ret[tm*8+7] = (*pupa)->developed.d;
}

void print_zero(int tm, double *ret) {
    ret[tm*8+0] = 0.0;
    ret[tm*8+1] = 0.0;
    ret[tm*8+2] = 0.0;
    ret[tm*8+3] = 0.0;
    ret[tm*8+4] = 0.0;
    ret[tm*8+5] = 0.0;
    ret[tm*8+6] = 0.0;
    ret[tm*8+7] = 0.0;
}

void sim(int tf, double *temp, double *photo, double *pr, double *y0, double thr, double *ret) {
    set_APPROX(1e-2);
    //
    double p1  = 0.0,
           p2  = 0.0,
           p3  = 0.0,
           p4  = 0.0,
           d1m = 0.0,
           d1s = 0.0,
           d2m = 0.0,
           d2s = 0.0,
           d3m = 0.0,
           d3s = 0.0;
    //
    spop2 egg   = spop2_init(0, MODE_ACCP_ERLANG);
    spop2 larva = spop2_init(0, MODE_ACCP_ERLANG);
    spop2 pupa  = spop2_init(0, MODE_ACCP_ERLANG);
    spop adult  = spop_init(0, MODE_GAMMA_HASH);
    //
    int tm = 0;
    if (y0[0]) spop2_add(egg,   0, y0[0]);
    if (y0[1]) spop2_add(larva, 0, y0[1]);
    if (y0[2]) spop2_add(pupa,  0, y0[2]);
    if (y0[3]) spop_add(adult,  0, 0, 0, y0[3]);
    print_out(tm, &egg, &larva, &pupa, &adult, ret);
    //
    for (tm=1; tm<tf; tm++) {
        //
        f_p1(temp[tm-1],photo[tm-1],pr,&p1);
        f_p2(temp[tm-1],photo[tm-1],pr,&p2);
        f_p3(temp[tm-1],photo[tm-1],pr,&p3);
        f_p4(temp[tm-1],photo[tm-1],pr,&p4);
        //
        f_d1ms(temp[tm-1],photo[tm-1],pr,&d1m,&d1s);
        f_d2ms(temp[tm-1],photo[tm-1],pr,&d2m,&d2s);
        f_d3ms(temp[tm-1],photo[tm-1],pr,&d3m,&d3s);
        //
        spop2_iterate(egg,   d1m, d1s, p1, 0);
        spop2_iterate(larva, d2m, d2s, p2, 0);
        spop2_iterate(pupa,  d3m, d3s, p3, 0);
        spop_iterate(adult,    0, 0, 0, 0,  p4, 0, 0, 0,  0);
        //
        spop_add(adult,  0, 0, 0, pupa->developed.d);
        spop2_add(pupa,  0, larva->developed.d);
        spop2_add(larva, 0, egg->developed.d);
        //
        print_out(tm, &egg, &larva, &pupa, &adult, ret);
        //
        if (thr > 0 && egg->size.d < thr && larva->size.d < thr && pupa->size.d < thr) {
            for (tm++; tm<tf; tm++)
                print_zero(tm, ret);
            break;
        }
    }
    //
    spop2_destroy(&egg);
    spop2_destroy(&larva);
    spop2_destroy(&pupa);
    spop_destroy(&adult);
}
