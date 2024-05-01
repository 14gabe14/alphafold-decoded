import torch
from torch import nn
# from tests.structure_module.residue_constants import rigid_group_atom_position_map, chi_angles_mask
from geometry.residue_constants import rigid_group_atom_position_map, chi_angles_mask,  chi_angles_chain
from geometry import residue_constants

def create_3x3_rotation(ex, ey):
    """
    Creates a rotation matrix by orthonormalizing ex and ey via Gram-Schmidt.
    Supports batched operation.

    Args:
        ex (torch.tensor): X-axes of the new frames, of shape (*, 3).
        ey (torch.tensor): Y-axes of the new frames, of shape (*, 3).

    Returns:
        torch.tensor: Rotation matrices of shape (*, 3, 3).
    """
    
    R = None
    
    ##########################################################################
    # TODO: Orthonormalize ex and ey, then compute ez as their crossproduct. # 
    #  Use torch.linalg.vector_norm to compute the norms for normalization.  #
    #  Orthogonalize ey against ex by subtracting the non-orthogonal part,   #
    #  ex * <ex, ey> from ey, after normalizing ex.                          #
    #  The keepdim parameter can be helpful for both operations.             #
    #  Stack the vectors as columns to build the rotation matrix.            #
    #  Make your to broadcast correctly, to allow for any number of          #
    #  leading dimensions.                                                   #
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return R

def quat_from_axis(phi, n):
    """
    Creates a quaternion with scalar cos(phi/2)
    and vector sin(phi/2)*n.

    Args:
        phi (torch.tensor): Angle of shape (*,)
        n (torch.tensor): Unit vector of shape (*, 3).

    Returns:
        torch.tensor: Quaternion of shape (*, 4).
    """

    q = None

    ##########################################################################
    # TODO: Implement the method as described above. You might need to       # 
    #   reshape phi to allow for broadcasting and concatenation.             # 
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return q

def quat_mul(q1, q2):
    """
    Batched multiplication of two quaternions.

    Args:
        q1 (torch.tensor): Quaternion of shape (*, 4).
        q2 (torch.tensor): Quaternion of shape (*, 4).

    Returns:
        torch.tensor: Quaternion of shape (*, 4).
    """
    
    a1 = q1[...,0:1] # a1 has shape (*, 1)
    v1 = q1[..., 1:] # v1 has shape (*, 3)
    a2 = q2[...,0:1] # a2 has shape (*, 1)
    v2 = q2[..., 1:] # v2 has shape (*, 3)

    q_out = None

    ##########################################################################
    # TODO: Implement batched quaternion multiplication.                     # 
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return q_out

def conjugate_quat(q):
    """
    Calculates the conjugate of a quaternion, i.e. 
    (a, -v) for q=(a, v).

    Args:
        q (torch.tensor): Quaternion of shape (*, 4).

    Returns:
        torch.tensor: Conjugate quaternion of shape (*, 4).
    """

    q_out = None

    ##########################################################################
    # TODO: Implement quaternion conjugation.                                # 
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return q_out

def quat_vector_mul(q, v):
    """
    Rotates a vector by a quaternion according to q*v*q', where q' 
    denotes the conjugate. The vector v is promoted to a quaternion 
    by padding a 0 for the scalar aprt.

    Args:
        q (torch.tensor): Quaternion of shape (*, 4).
        v (torch.tensor): Vector of shape (*, 3).

    Returns:
        torch.tensor: Rotated vector of shape (*, 3).
    """
    batch_shape = v.shape[:-1]
    v_out = None

    ##########################################################################
    # TODO: Implement batched quaternion vector multiplication.              # 
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return v_out

def quat_to_3x3_rotation(q):
    """
    Converts a quaternion to a rotation matrix.

    Args:
        q (torch.tensor): Quaternion of shape (*, 4).
    """

    R = None
    
    ##########################################################################
    # TODO: Follow these steps to convert a quaternion to a rotation matrix: # 
    #   - Create the vectors [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0].    #
    #       broadcast them to shape (*, 3) for batched use.                  #
    #   - Rotate these vectors by q and assemble the result into a matrix.   #
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return R

def assemble_4x4_transform(R, t):
    """
    Assembles a rotation matrix R and a translation t into a 4x4 homogenous transform.

    Args:
        R (torch.tensor): Rotation matrix of shape (*, 3, 3).
        t (torch.tensor): Translation of shape (*, 3).

    Returns:
        torch.tensor: Transform of shape (*, 4, 4).
    """

    T = None

    ##########################################################################
    # TODO: Implement the method in the following steps:                     # 
    #   - Concatenate R and t along the column axis.                         #
    #   - Build the pad [0,0,0,1] and broadcast it to shape (*, 1, 4)        #
    #   - Concatenate Rt and the pad along the row axis.                     #
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return T
    

def create_4x4_transform(ex, ey, translation):
    """
    Creates a 4x4 transform, where the rotation matrix is constructed from ex and ey
    (after orthogonalizing ey against ex) and with the given translation.

    Args:
        ex (torch.tensor): Vector of shape (*, 3).
        ey (torch.tensor): Vector of shape (*, 3). Orthogonalized against ex before
            used for the creation of the rotation matrix.  
        translation (torch.tensor): Vector of shape (*, 3).

    Returns:
        torch.tensor: Transform of shape (*, 4, 4).
    """

    T = None

    ##########################################################################
    # TODO: Implement create_4x4_transform. This can be done in two lines,   # 
    #   using the methods you constructed earlier.                           # 
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return T
    
def invert_4x4_transform(T):
    """
    Inverts a 4x4 transform (R, t) according to 
    (R.T, -R.T @ t).

    Args:
        T (torch.tensor): Transform of shape (*, 4, 4).

    Returns:
        torch.tensor: Inverted transform of shape (*, 4, 4).
    """

    inv_T = None 

    ##########################################################################
    # TODO: Implement the 4x4 transform inversion.                           # 
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return inv_T

def makeRotX(phi):
    """
    Creates a 4x4 transform for rotation of phi around the X axis.
    phi is given as (cos(phi), sin(phi)).
    The matrix is constructed according to
    [1  0         0        ]
    [0  cos(phi)  -sin(phi)]
    [0  sin(phi)  cos(phi) ]

    Args:
        phi (torch.tensor): Angle of shape (*, 2). The angle is given as
            (cos(phi), sin(phi)).

    Returns:
        torch.tensor: Rotation transform of shape (*, 4, 4).
    """

    batch_shape = phi.shape[:-1]
    phi1, phi2 = torch.unbind(phi, dim=-1)
    T = None

    ##########################################################################
    # TODO: Build the rotation matrix described above. Assemble it together  # 
    #   with a translation of 0 to a 4x4 transformation.                     #
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return T

    
### End of general geometry
### Start of AF specific geometry



def calculate_non_chi_transforms():
    """
    Calculates transforms for the following local backbone frames:
    
    backbone_group: Identity
    pre_omega_group: Identity
    phi_group: 
        ex: CA -> N
        ey: (1, 0, 0)
        t:  N
    psi_group:
        ex: CA -> C
        ey: N  -> CA
        t:  C

    Returns:
        torch.tensor: Stacked transforms of shape (20, 4, 4, 4).
            The second dim corresponds to the different frames.
            The last two dims are the shape of the individual transforms.
    """

    non_chi_transforms = None
    
    ##########################################################################
    # TODO: Build the four non-chi transforms as described above. Stack them # 
    #   to build non_chi_transforms.                                         #
    #   The transforms are built for every amino acid individually. You can  # 
    #   iterate over rigid_group_atom_position_map.values() to get the       #
    #   individual atom -> position maps for each amino acid. You can use    #
    #   enumerate(rigid_group_atom_position_map.values()) to iterate over    #
    #   the amino acid indices and the values jointly.                       #
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return non_chi_transforms

def calculate_chi_transforms():
    """
    Calculates transforms for the following local side-chain frames:
    chi1: 
        ex: CA   -> #SC0
        ey: CA   -> N
        t:  #SC0
    chi2:
        ex: #SC0 -> #SC1
        ey: #SC0 -> CA
        t:  #SC1
    chi3:
        ex: #SC1 -> #SC2
        ey: #SC1 -> #SC0
        t: #SC2
    chi4:
        ex: #SC2 -> #SC3
        ey: #SC2 -> #SC1
        t: #SC3

    #SC0 - #SC3 denote the names of the side-chain atoms.
    If the chi angles are not present for the amino acid according to
    chi_angles_mask, they are substituted by the Identity transform.
    

    Returns:
        torch.tensor: Stacked transforms of shape (20, 4, 4, 4).
            The second dim corresponds to the different frames.
            The last two dims are the shape of the individual transforms.
    """

    chi_transforms = None

    # Note: For chi2, chi3 and chi4, ey is the inverse of the previous ex.
    # This means, that ey is (-1, 0, 0) in local coordinates for the frame.
    # Also note: For chi2, chi3, and chi4, ex starts at t of the previous transform.
    # This means, that the starting point is 0 in local coordinates.

    ##########################################################################
    # TODO: Construct the chi transforms. You can follow these steps:        # 
    #   - Construct an empty tensor of shape (20, 4, 4, 4).                  #
    #   - Iterate over rigid_group_atom_position_map.items() to get the      #
    #       amino acids names and atom->position maps for each amino acid.   #
    #   - Iterate over range(4) for the different side-chain angles.         #
    #   - If chi_angles_mask is False, set the transform to the Identity.    #
    #   - Select the next side-chain atom from chi_angles_chain.             #
    #   - Build the transforms as described above. You'll need an if-clause  #
    #       to differentiate between chi1 and the other transforms.          #
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return chi_transforms

def precalculate_rigid_transforms():
    """
    Calculates the non-chi transforms backbone_group, pre_omega_group, phi_group and psi_group,
    together with the chi transforms chi1, chi2, chi3, and chi4.

    Returns:
        torch.tensor: Transforms of shape (20, 8, 4, 4).
    """

    rigid_transforms = None
    
    ##########################################################################
    # TODO: Concatenate the non-chi transforms and chi transforms.           # 
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return rigid_transforms

def compute_global_transforms(T, alpha, F):
    """
    Calculates global frames for each frame group of each amino acid
    by applying the global transform T and injecting rotation transforms
    in between the side chain frames. 
    Implements Line 1 - Line 10 of Algorithm 24.

    Args:
        T (torch.tensor): Global backbone transform for each amino acid. Shape (N_res, 4, 4).
        alpha (torch.tensor): Torsion angles for each amino acid. Shape (N_res, 7, 2).
            The angles are in the order (omega, phi, psi, chi1, chi2, chi3, chi4).
            Angles are given as (cos(a), sin(a)).
        F (torch.tensor): Label for each amino acid of shape (N_res,).
            Labels are encoded as 0: Ala, 1: Arg, ..., 19: Val.

    Returns:
        torch.tensor: Global frames for each amino acid of shape (N_res, 8, 4, 4).
    """

    global_transforms = None

    ##########################################################################
    # TODO: Construct the global transforms, according to line 1 - line 10   #
    #   from Algorithm 24. You don't need to support batched use.            #
    #   You can follow these steps:                                          #  
    #   - Normalize alpha, so that its values represent (cos(phi), sin(phi)) #
    #   - Use `torch.unbind` to unbind alpha into omega, phi, psi, chi1,     #
    #       chi2, chi3, and chi4.                                            #
    #   - Compute all_rigid_transforms with precalculate_rigid_transforms.   #
    #   - Send the transforms to the same device that T/alpha/F are on.      #
    #   - Select the correct local_transforms by indexing with F.            #
    #   - The global backbone transform is T, since the local backbone       #
    #       transform is the identity.                                       #
    #   - Iterate over omega, phi, psi, and chi1. Build the global           #
    #       transforms by concatenating T, the local transform, and a        #
    #       rotation matrix. Concatenating 4x4 transforms means multiplying  #
    #       them. You can use the matrix multiplication operator `@`.        #
    #   - Iterate over chi2, chi3, and chi4. Build the global transforms by  #
    #       concatenating the upstream global transform (chi1, chi2, chi3)   #
    #       with the local transform and the rotation.                       #
    #   - Stack the global transforms.                                       #
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return global_transforms

def compute_all_atom_coordinates(T, alpha, F):
    """
    Implements Algorithm 24. 

    Args:
        T (torch.tensor): Global backbone transform for each amino acid. Shape (N_res, 4, 4).
        alpha (torch.tensor): Torsion angles for each amino acid. Shape (N_res, 7, 2).
            The angles are in the order (omega, phi, psi, chi1, chi2, chi3, chi4).
            Angles are given as (cos(a), sin(a)).
        F (torch.tensor): Label for each amino acid of shape (N_res,).
            Labels are encoded as 0: Ala, 1: Arg, ..., 19: Val.

    Returns:
        tuple: A tuple consisting of the following values:
            global_positions: Tensor of shape (N_res, 37, 3), containing the global positions
                for each atom for each amino acid.
            atom_mask: Boolean tensor of shape (N_res, 37), containing whether or not the atoms
                are present in the amino acids.
    """

    global_positions, atom_mask = None, None

    ##########################################################################
    # TODO: Implement Algorithm 24. You can follow these steps:              # 
    #   - build the global frames using compute_global_transforms.           #
    #   - retrieve atom_local_positions, atom_frame_inds, and atom_mask      #
    #       from residue_constants. Map them to the same device used by T,   #
    #       alpha and F using `tensor.to(device=device)`.                    #
    #   - Select the local positions, frame inds and mask using F.           #
    #   - Select the global frames using your selected frame indices.        #
    #       You can use integer indexing or `torch.gather` for this.         #
    #       If using integer indexing, you'll need to index with             #
    #       [0,...,N_res], broadcasted to (N_res, N_atoms) into the first    #
    #       dimension, so that the shape matches the selected frame inds.    #
    #   - Pad the local positions with 1 to promote them to homogenous       #
    #       coordinates. Warp them through the selected global frames by     #  
    #       batched matrix vector multiplication. You can use `torch.einsum` #
    #       to handle the dimensions.                                        #
    #   - Drop the 1 of the resulting homogenous coordinates to select the   #
    #       atom positions.                                                  #
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return global_positions, atom_mask

