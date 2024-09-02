#ifndef _EKF_GPU
#define _EKF_GPU

/*
    EKF Algorithm
    
    Predict
    x = F * x
    P = F * P * F_t + Q

    Update
    
    S = H * P * H_t + R
    K = P * H_t * S_inv
    y = z - H * x
    x = x + K * y
    P = (I - K * H) * P (can build up floating point error over time)
    P = (I - K * H) * P * (I - K * H)_T + K * R * K_T Joesphs form

    */

#include <arrayfire.h>

#define MAT_DIM 6

#define MES_DIM 4

struct state_t
{
    af::array x;
    af::array P;
};

typedef struct state_t state_t;

struct process_params_t
{
    af::array F;
    af::array Q;
};

typedef struct process_params_t process_params_t;

struct measurement_params_t
{
    af::array H;
    af::array R;
};

typedef struct measurement_params_t measurement_params_t;

void init_filter(state_t *, process_params_t *,measurement_params_t *);

void predict(state_t *, process_params_t *);

void update(state_t *, measurement_params_t *, af::array);

class EKF
{
    public:
        state_t state;
        process_params_t process_params;
        measurement_params_t measurement_params;

        EKF();

        void predict(float);
        
        void update(af::array);

        void subs_F(float);

        void subs_H(float);
          
};
#endif