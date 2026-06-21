import numpy as np
# Create time-space grids for 1D spatial domain (t,x) and 2D spatial domain (t,x,y)

# Create time-space grids for 1D spatial domain (t,x)
def create_time_space_grid_1d(t_min, t_max, x_min, x_max,
                              num_inner, num_boundary, num_initial_displacement, num_initial_velocity):
    interior_points = create_interior_points_1d(t_min, t_max, x_min, x_max, num_inner)
    boundary_points = create_boundary_points_1d(t_min, t_max, x_min, x_max, num_boundary)
    initial_displacement_points = create_initial_displacement_points_1d(t_min, x_min, x_max, num_initial_displacement)
    initial_velocity_points = create_initial_velocity_points_1d(t_min, x_min, x_max, num_initial_velocity)

    # Total number of points
    N_total = (interior_points.shape[0] + boundary_points.shape[0] +
               initial_displacement_points.shape[0] + initial_velocity_points.shape[0])
    print(f"Total number of collocation points created: {N_total}")
    print(f"Interior points: {interior_points.shape[0]}, Boundary points: {boundary_points.shape[0]}, "
          f"Initial displacement points: {initial_displacement_points.shape[0]}, Initial velocity points: {initial_velocity_points.shape[0]}")
    return interior_points, boundary_points, initial_displacement_points, initial_velocity_points


def create_interior_points_1d(t_min, t_max, x_min, x_max, num_inner):
    # exclude the starting time t_min from interior points so no interior point lies exactly on the initial slice
    # TODO: Currently as many points in time as in space, generalize later if needed
    t_values = np.linspace(t_min, t_max, num_inner + 1, endpoint=True)[1:]
    x_values = np.linspace(x_min, x_max, num_inner + 1, endpoint=True)[1:]
    T, X = np.meshgrid(t_values, x_values, indexing='ij')
    interior_points = np.column_stack((T.ravel(), X.ravel()))
    return interior_points


# Boundary points for 1d domain (t, x)
def create_boundary_points_1d(t_min, t_max, x_min, x_max, num_boundary):
    # Num_boundary points at  (t, x_min) and num_boundary points at (t, x_max)
    t_values = np.linspace(t_min, t_max, int(num_boundary))
    x_left = np.full(t_values.size, x_min)
    left_boundary = np.column_stack((t_values, x_left))
    x_right = np.full(t_values.size, x_max)
    right_boundary = np.column_stack((t_values, x_right))
    boundary_points = np.vstack((left_boundary, right_boundary))
    return boundary_points
    


def create_initial_displacement_points_1d(t_min, x_min, x_max, num_initial_displacement):
    # Points with t = t_min and varying x
    x_values = np.linspace(x_min, x_max, num_initial_displacement + 1, endpoint=False)[1:]
    t_values = np.full(x_values.size, t_min)
    initial_displacement_points = np.column_stack((t_values, x_values))
    return initial_displacement_points


def create_initial_velocity_points_1d(t_min, x_min, x_max, num_initial_velocity):
    # Points with t = t_min and varying x
    x_values = np.linspace(x_min, x_max, num_initial_velocity + 1, endpoint=False)[1:]
    t_values = np.full(x_values.size, t_min)
    initial_velocity_points = np.column_stack((t_values, x_values))
    return initial_velocity_points

# Create time-space grids for 2D spatial domain (t,x,y)
def create_time_space_grid_2d(t_min, t_max, x_min, x_max, y_min, y_max,
                              num_inner, num_boundary, num_initial_displacement, num_initial_velocity):
    interior_points = create_interior_points_2d(t_min, t_max, x_min, x_max, y_min, y_max, num_inner)
    boundary_points = create_boundary_points_2d(t_min, t_max, x_min, x_max, y_min, y_max, num_boundary)
    initial_displacement_points = create_initial_displacement_points_2d(t_min, x_min, x_max, y_min, y_max, num_initial_displacement)
    initial_velocity_points = create_initial_velocity_points_2d(t_min, x_min, x_max, y_min, y_max, num_initial_velocity)

    # Total number of points
    N_total = (interior_points.shape[0] + boundary_points.shape[0] +
               initial_displacement_points.shape[0] + initial_velocity_points.shape[0])
    print(f"Total number of collocation points created: {N_total}")
    print(f"Interior points: {interior_points.shape[0]}, Boundary points: {boundary_points.shape[0]}, "
          f"Initial displacement points: {initial_displacement_points.shape[0]}, Initial velocity points: {initial_velocity_points.shape[0]}")
    return interior_points, boundary_points, initial_displacement_points, initial_velocity_points


def create_interior_points_2d(t_min, t_max, x_min, x_max, y_min, y_max, num_inner):
    # exclude the starting time t_min from interior points so no interior point lies exactly on the initial slice
    t_values = np.linspace(t_min, t_max, int((num_inner + 1)/2), endpoint=True)[1:]
    x_values = np.linspace(x_min, x_max, num_inner + 1, endpoint=True)[1:]
    y_values = np.linspace(y_min, y_max, num_inner + 1, endpoint=True)[1:]
    T, X, Y = np.meshgrid(t_values, x_values, y_values, indexing='ij')
    interior_points = np.column_stack((T.ravel(), X.ravel(), Y.ravel()))
    return interior_points


def create_boundary_points_2d(t_min, t_max, x_min, x_max, y_min, y_max, num_boundary):
    # Boundary points on the four edges of the spatial domain (x,y) for all t
    t_values = np.linspace(t_min, t_max, int(num_boundary))
    
    # x boundaries (y varies)
    y_values = np.linspace(y_min, y_max, num_boundary)
    X_left = np.full(t_values.size * y_values.size, x_min)
    X_right = np.full(t_values.size * y_values.size, x_max)
    T_mesh, Y_mesh = np.meshgrid(t_values, y_values, indexing='ij')
    T_mesh = T_mesh.ravel()
    Y_mesh = Y_mesh.ravel()
    left_boundary = np.column_stack((T_mesh, X_left, Y_mesh))
    right_boundary = np.column_stack((T_mesh, X_right, Y_mesh))

    # y boundaries (x varies)
    x_values = np.linspace(x_min, x_max, num_boundary)
    Y_bottom = np.full(t_values.size * x_values.size, y_min)
    Y_top = np.full(t_values.size * x_values.size, y_max)
    T_mesh2, X_mesh2 = np.meshgrid(t_values, x_values, indexing='ij')
    T_mesh2 = T_mesh2.ravel()
    X_mesh2 = X_mesh2.ravel()
    bottom_boundary = np.column_stack((T_mesh2, X_mesh2, Y_bottom))
    top_boundary = np.column_stack((T_mesh2, X_mesh2, Y_top))

    boundary_points = np.vstack((left_boundary, right_boundary, bottom_boundary, top_boundary))
    return boundary_points


def create_initial_displacement_points_2d(t_min, x_min, x_max, y_min, y_max, num_initial_displacement):
    # Points at initial time t = t_min
    x_values = np.linspace(x_min, x_max, num_initial_displacement + 1, endpoint=False)[1:]
    y_values = np.linspace(y_min, y_max, num_initial_displacement + 1, endpoint=False)[1:]
    X, Y = np.meshgrid(x_values, y_values, indexing='ij')
    T = np.full(X.size, t_min)
    initial_displacement_points = np.column_stack((T, X.ravel(), Y.ravel()))
    return initial_displacement_points


def create_initial_velocity_points_2d(t_min, x_min, x_max, y_min, y_max, num_initial_velocity):
    # Points at initial time t = t_min
    x_values = np.linspace(x_min, x_max, num_initial_velocity + 1, endpoint=False)[1:]
    y_values = np.linspace(y_min, y_max, num_initial_velocity + 1, endpoint=False)[1:]
    X, Y = np.meshgrid(x_values, y_values, indexing='ij')
    T = np.full(X.size, t_min)
    initial_velocity_points = np.column_stack((T, X.ravel(), Y.ravel()))
    return initial_velocity_points