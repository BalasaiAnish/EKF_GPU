#include "ekf.h"

#define N_POINTS 100

using namespace af;

EKF ekf;

int main()
{
    float dt[N_POINTS];
    float imu_readings[N_POINTS];
    float gps_readings[N_POINTS];

    for(int i=0;i<N_POINTS;i++)
    {
        imu_readings[i] = 0;
        gps_readings[i] = i;
        dt[i] = i;
    }

    for(int i=50;i<100;i++)
    {
        gps_readings[i] = gps_readings[i-1] + 2;
    }

    array DT(1,N_POINTS,dt);

    array IMU(1,N_POINTS,imu_readings);
    array GPS(1,N_POINTS,gps_readings);

    array Z(MES_DIM,1);

    array X(1,N_POINTS);
    array X_D(1,N_POINTS);
    array X_DD(1,N_POINTS);
    array Y(1,N_POINTS);
    array Y_D(1,N_POINTS);
    array Y_DD(1,N_POINTS);

    for(int i=0;i<N_POINTS;i++)
    {
        Z(0,0) = GPS(0,i) + randn(1,1)/10;
        Z(1,0) = IMU(0,i) + randn(1,1)/10;
        Z(2,0) = GPS(0,i)*GPS(0,i) + randn(1,1)/10;
        Z(3,0) = 1 + randn(1,1)/10;

        ekf.predict(1.0);
        ekf.update(Z);

        X(0,i) = ekf.state.x(0,0);
        X_D(0,i) = ekf.state.x(1,0);
        X_DD(0,i) = ekf.state.x(2,0);
        Y(0,i) = ekf.state.x(3,0);
        Y_D(0,i) = ekf.state.x(4,0);
        Y_DD(0,i) = ekf.state.x(5,0);

        if(i%10==0)
        {
            af_print(ekf.state.x);
            af_print(ekf.state.P);
        }
    }    

    const int width = 512, height = 512;

    Window gps(width, height, "Position readings");
    Window imu(width, height, "Acceleration readings");
    Window traj(width, height, "Trajectory estimate");
    Window x_d(width, height, "X velocity estimate");
    Window x_dd(width, height, "X acceleration estimate");    

    
    do
    {
        imu.plot(transpose(DT),transpose(IMU));
        gps.plot(transpose(DT),transpose(GPS));
        traj.plot(transpose(X),transpose(Y));
        x_d.plot(transpose(DT),transpose(X_D));
        x_dd.plot(transpose(DT),transpose(X_DD));
    }while(!gps.close() && !imu.close() && !traj.close() && !x_d.close() && !x_dd.close());
    
}
