import  torch;

"""
The functions in this file implement various finite difference approximations for first and second
time derivatives of tensor-valued time sequences.
"""


def Derivative1_Order2(X : torch.Tensor, h : float) -> torch.Tensor:
    """
    This function finds an O(h^2) approximation of the time derivative to the time series stored 
    in the rows of X. Specifically, we assume the i'th row of X represents a sample of a function, 
    x, at time t_0 + i*h. We return a new tensor whose i'th row holds an O(h^2) approximation of 
    (d/dt)x(t_0 + i*h)
    
    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    X: A torch.Tensor object representing a time sequence. We assume that X has shape [Nt, ...] 
    and that X[i, ...] represents the value of some function at the i'th time step. Specifically, 
    we assume that X[i, ...] is the value of some function, X, at time t_0 + i*h.

    h: The time step size.


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    A torch.Tensor object with the same shape as X whose i'th row holds a O(h^2) approximation of 
    the time derivative of X at the i'th time step. 
    """

    # For this scheme to work, X must contain at least 3 rows.
    assert(X.shape[0] >= 3);

    # Initialize a tensor to hold the time derivative.
    dX_dt   : torch.Tensor  = torch.empty_like(X);
    Nt      : int           = X.shape[0];

    # Now... cycle through the time steps. Note that we use a different method for the first and
    # last time steps.
    dX_dt[0, ...] = (1./h)*((-3./2.)*X[0, ...] + 2*X[1., ...] - (1/2)*X[2., ...]);
    for i in range(1, Nt - 1):
        dX_dt[i, ...] = (1./(2.*h))*(X[i + 1, ...] - X[i - 1, ...]);

    dX_dt[-1, ...] = (1./h)*((3./2.)*X[-1, ...] - 2*X[-2, ...] + (1./2.)*X[-3, ...]);

    # All done!
    return dX_dt;



def Derivative2_Order2(X : torch.Tensor, h : float) -> torch.Tensor:
    """
    This function finds an O(h^2) approximation of the second time derivative to the time series 
    stored in the rows of X. Specifically, we assume the i'th row of X represents a sample of a 
    function, x, at time t_0 + i*h. We return a new tensor whose i'th row holds an O(h^2) 
    approximation of (d^2/dt^2)x(t_0 + i*h)
    
    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    X: A torch.Tensor object representing a time sequence. We assume that X has shape [Nt, ...] 
    and that X[i, ...] represents the value of some function at the i'th time step. Specifically, 
    we assume that X[i, ...] is the value of some function, x, at time t_0 + i*h.

    h: The time step size.


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    A torch.Tensor object with the same shape as X whose i'th row holds a O(h^2) approximation of 
    the time derivative of x at the i'th time step (that is, it approximates x''(t_0 + i h)).
    """

    # For this scheme to work, X must contain at least 4 rows.
    assert(X.shape[0] >= 4);

    # Initialize a tensor to hold the time derivative.
    d2X_dt2  : torch.Tensor  = torch.empty_like(X);
    Nt      : int           = X.shape[0];

    # Now... cycle through the time steps. Note that we use a different method for the first and
    # last time steps.
    d2X_dt2[0, ...] = (1./(h*h))*(2*X[0, ...] - 5*X[1, ...] + 4*X[2, ...] - X[3, ...]);
    for i in range(1, Nt - 1):
        d2X_dt2[i, ...] = (1./(h*h))*(X[i - 1, ...] - 2*X[i, ...] + X[i + 1, ...]);

    d2X_dt2[-1, ...] = (1./(h*h))*(2*X[-1, ...] - 5*X[-2, ...] + 4*X[-3, ...] - X[-4, ...]);

    # All done!
    return d2X_dt2;