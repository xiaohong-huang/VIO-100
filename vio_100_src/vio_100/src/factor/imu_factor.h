
#pragma once

#include <eigen3/Eigen/Dense>
#include "integration_base.h"
#include "../solver/cost_function.h"



class IMUFactor  {
  public:
    IMUFactor() = delete;
    IMUFactor(IntegrationBase* _pre_integration): pre_integration(_pre_integration) {
    }
    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;


    IntegrationBase* pre_integration;
    double dt_i = 0, dt_j = 0;

};