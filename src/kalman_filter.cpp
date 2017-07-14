#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {
}

KalmanFilter::~KalmanFilter() {
}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
    MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * predict the state
   */
  x_ = F_ * x_; // + u;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * update the state by using Kalman Filter equations
   */
  VectorXd y = z - H_ * x_;
  UpdateGeneral(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   TODO:
   * update the state by using Extended Kalman Filter equations
   */
  double rho = sqrt(x_(0) * x_(0) + x_(1) * x_(1));
  double phi = atan2(x_(1), x_(0));
  double rho_dot = (x_(0) * x_(2) + x_(1) * x_(3)) / rho;
  VectorXd h = VectorXd(3); // h(x_)
  h << rho, phi, rho_dot;

  VectorXd y = z - h;

  y[1] = atan2(sin(y[1]), cos(y[1]));

  UpdateGeneral(y);
}

void KalmanFilter::UpdateGeneral(const Eigen::VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);

  //new state
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}
