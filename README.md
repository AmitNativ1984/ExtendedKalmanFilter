# Extended Kalman Filter
extended kalman filter with sensor fusion of lidar and radar

## lidar data is linear
uses normal kalman filter
    
## radar linear is none linear 
   used with extended kalman filter,
    
   Jacobian matrix
   
## sensor fusion
time difference is not necessarily constant

prediction state rely on last sensor reading
(either same sensor or the other sensor)
