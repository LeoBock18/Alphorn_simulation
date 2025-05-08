import numpy as np
import warnings

def conical_piece(inlet_radius: float, outlet_radius: float, length: float, x_initial: float, tau: float):
    """
    Gives back np.array with coordinates and radii for a conical piece.

    :param inlet_radius: The radius of the inlet part of the cone (smaller extremity).
    :type inlet_radius: float
    :param outlet_radius: The radius of the outlet part of the cone (bigger extremity).
    :type outlet_radius: float
    :param length: Length of the cone.
    :type length: float
    :param x_initial: x-axis coordinate of the first point of the cone.
    :type x_initial: float
    :param tau: Set discretization step along x-axis.
    :type tau: float
    :return: Coordinates and radii of the conical piece
    :rtype: ndarray
    """

    # Compute cone slope
    m = (outlet_radius - inlet_radius) / length
    # Define x range
    xx = np.arange(0, length, tau)
    # compute radii
    radii = m*xx + inlet_radius
    # translate x-axis properly
    xx += x_initial
    # put together data and add y,z coordinates
    xx_and_radii = np.column_stack((xx, radii))
    y_z = np.zeros([xx_and_radii.shape[0],2])
    coord_and_radii = np.insert(xx_and_radii, 1, y_z.T, axis=1)

    return coord_and_radii


def assemble_mesh_data(intersection_radius: float, len_cone_1: float, len_cone_2: float, file_index: int, filename: str = "mesh_data/mesh_data", mouthpiece_flag: bool = True, len_cone_3: float = 825.0, len_cone_4: float = 465.0, mouthpiece_file: str = "mesh_data/mouthpiece.csv", bell_file: str = "mesh_data/final_bell.csv",  tau: float = 2.0, verbose: bool = False):
    """
    Creates a CSV file of a mesh with the given parameters.

    :param intersection_radius: The radius between first and second cones (measured at the end of first cone).
    :type intersection_radius: float
    :param len_cone_1: Length of the first conical part.
    :type len_cone_1: float
    :param len_cone_2: Length of the second conical part.
    :type len_cone_2: float
    :param len_cone_3: Length of the third conical part (keep it default in our experiments).
    :type len_cone_3: float
    :param len_cone_4: Length of the last conical part (keep it default in our experiments).
    :type len_cone_4: float
    :param file_index: Index used for creating output filename (useful when function is called inside a loop, avoids overwriting of data).
    :type file_index: int
    :param filename: File name where to save mesh data.
    :type filename: str
    :param mouthpiece_flag: Indicates whether mesh includes a mouthpiece or starts directly with the conical part.
    :type mouthpiece_flag: bool
    :param mouthpiece_file: File from which mouthpiece data are loaded.
    :type mouthpiece_file: str
    :param bell_file: File from which bell data are loaded.
    :type bell_file: str
    :param tau: Set discretization step along x-axis.
    :type tau: float
    :param verbose: Optional warning raiser for correct mesh construction.
    :type verbose: bool
    """


    # Raise warning if intersection_radius is not coherent with the overall geometry
    if verbose and not (6.5 <= intersection_radius <= 28):
        warnings.warn(f"Warning: 'intersection_radius' ({intersection_radius}) should be between 6.5 and 28 for geometry coherence.", stacklevel=2)

    # Set initial x-axis coordinate
    initial_point = 0

    # Load moutpiece
    if mouthpiece_flag:
        mouthpiece = np.loadtxt(mouthpiece_file, delimiter=",", skiprows=1)
        # If mouthpiece added, we need to translate the initial point of the conical parts
        initial_point = 110
    
    # Load final_bell
    final_bell = np.loadtxt(bell_file, delimiter=",", skiprows=1)

    # Construct the conical parts. The intersection between the first two conical parts varies depending on input parameters
    cone_1 = conical_piece(6.5, intersection_radius, len_cone_1, initial_point, tau)
    # update initial point of next conical part
    initial_point += len_cone_1
    # Repeat for all the cones
    cone_2 = conical_piece(intersection_radius + 0.5, 28.5, len_cone_2, initial_point, tau)
    initial_point += len_cone_2
    cone_3 = conical_piece(29, 44, len_cone_3, initial_point, tau)
    initial_point += len_cone_3
    cone_4 = conical_piece(44.5, 51, len_cone_4, initial_point, tau)
    initial_point += len_cone_4

    # Set right x-value for final part
    final_bell[:,0] += initial_point

    # Concatenate data in a single arrat
    if mouthpiece_flag:
        final_mesh = np.concatenate((mouthpiece, cone_1, cone_2, cone_3, cone_4, final_bell), axis=0)
    else:
        final_mesh = np.concatenate((cone_1, cone_2, cone_3, cone_4, final_bell), axis=0)
    
    # Create the file name by concatenating the string and integer
    final_filename = f"{filename}{file_index}.csv"

    # Save mesh in a file
    np.savetxt(final_filename, final_mesh, delimiter=",", header="x-spline, y-spline, z-spline, radius", comments='', fmt="%.3f")



def main():
    
    # Change script here
    intersection_radii = [13.5, 16]
    len_first_cones = [560, 570]
    len_second_cones = [925, 950]

    for i in range(len(intersection_radii)):
        assemble_mesh_data(intersection_radii[i], len_first_cones[i], len_second_cones[i], i)
    


# Ensure the main function is called when the script is executed
if __name__ == "__main__":
    main()