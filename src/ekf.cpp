#include "ekf.h"

using namespace af;

void init_filter(state_t *state, process_params_t *process_params, measurement_params_t *measurement_params)
{
    float p[] = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0};
    float x[] = {1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0};
    
    array P(MAT_DIM,MAT_DIM,p);
    array X(MAT_DIM,1,x);

    state->x = X;
    state->P = P;

    float f[] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    float q[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    array F(MAT_DIM,MAT_DIM,f);
    array Q(MAT_DIM,MAT_DIM,q);

    process_params->F = F;
    process_params->Q = Q;

    float h[] = {
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.1
                };
    float r[] = {0.0};

    array H(MES_DIM,MAT_DIM,h);
    array R(1,1,r);

    measurement_params->H = H;
    measurement_params->R = R;
}

void predict(state_t *state, process_params_t *process_params)
{
    state->x = matmul(process_params->F,state->x);
    state->P = matmul(process_params->F,state->x,transpose(process_params->F)) + process_params->Q;
}

void update(state_t *state, measurement_params_t *measurement_params, array z)
{
    array S = matmul(measurement_params->H,state->P,transpose(measurement_params->H));
    array K = matmul(state->P,transpose(measurement_params->H),inverse(S));

    array y = z - matmul(measurement_params->H,state->x);

    state->x = state->x + matmul(K,y);
    state->P = (identity(MAT_DIM,MAT_DIM) - matmul(K,measurement_params->H)) + state->P;
}

EKF::EKF()
{
    float p[] = {100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 100.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 100.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 100.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 100.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 100.0,
                };
    float x[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            
    array P(MAT_DIM,MAT_DIM,p);
    array X(MAT_DIM,1,x);

    state.x = X;
    state.P = P;

    float f[] = {
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                };
    float q[] = {0.0000001, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0000001, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0000001, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0000001, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0000001, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0000001,
                };

    array F(MAT_DIM,MAT_DIM,f);
    array Q(MAT_DIM,MAT_DIM,q);

    process_params.F = F;
    process_params.Q = Q;

    float h[] = {1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 1.0,
                };

    float r[] = {
                0.0000001, 0.0, 0.0, 0.0,
                0.0, 0.0000001, 0.0, 0.0,
                0.0, 0.0, 0.0000001, 0.0,
                0.0, 0.0, 0.0, 0.0000001,
                };

    array H(MES_DIM,MAT_DIM,h);
    array R(MES_DIM,MES_DIM,r);

    measurement_params.H = H;
    measurement_params.R = R;
}

void EKF::predict(float dt)
{

    subs_F(dt);

    state.x = matmul(process_params.F,state.x);
    state.P = matmul(matmul(process_params.F,state.P),transpose(process_params.F)) + process_params.Q;
}
        

void EKF::update(array Z)
{
    array S = matmul(measurement_params.H,state.P,transpose(measurement_params.H)) + measurement_params.R;

    array K = matmul(state.P,transpose(measurement_params.H),inverse(S));

    array y = Z - matmul(measurement_params.H,state.x);

    state.x = state.x + matmul(K,y);

    state.P = matmul((identity(MAT_DIM,MAT_DIM) - matmul(K,measurement_params.H)),state.P,transpose((identity(MAT_DIM,MAT_DIM) - matmul(K,measurement_params.H)))) + matmul(K,measurement_params.R,transpose(K));
}

void EKF::subs_F(float dt)
{
    float f_sub[] = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    dt, 1.0, 0.0, 0.0, 0.0, 0.0,
                    dt*dt/2, dt, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, dt, 1.0, 0.0,
                    0.0, 0.0, 0.0, dt*dt/2, dt, 1.0,
                    };
    
    array F_sub(MAT_DIM,MAT_DIM,f_sub);

    process_params.F = F_sub;
}

void EKF::subs_H(float dt)
{
    // This method is empty for this particular state space
}
