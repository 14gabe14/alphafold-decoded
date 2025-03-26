import torch
from torch import nn
# from tests.structure_module.residue_constants import rigid_group_atom_position_map, chi_angles_mask
from geometry.residue_constants import rigid_group_atom_position_map, chi_angles_mask,  chi_angles_chain
from geometry import residue_constants
from torch import linalg as LA

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
    
    ex = ex / LA.vector_norm(ex, dim=-1, keepdim = True)

    projection = torch.einsum("...i,...i->...", ex, ey).unsqueeze(-1) 
    projection = projection * ex

    ey = ey - projection

    ey = ey / LA.vector_norm(ey, dim=-1, keepdim = True)

    ez = LA.cross(ex, ey, dim=-1)

    R = torch.stack((ex, ey, ez), dim=-1)
   

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

    phi = phi.unsqueeze(-1)
    cos_phi = torch.cos(phi/2)
    sin_phi = torch.sin(phi/2)
    q = torch.cat((cos_phi, sin_phi * n), dim=-1)

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

    temp_1 = a1 * a2 - torch.sum(v1 * v2, dim=-1, keepdim=True)
    temp_2 = a1 * v2 + a2 * v1 + LA.cross(v1, v2, dim=-1)

    q_out = torch.cat((temp_1, temp_2), dim=-1)


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
    q_out = q.clone()
    q_out[..., 1:4] = -q[..., 1:4]
    
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

    v_quat = torch.cat((torch.zeros(*batch_shape, 1).to(q.device).to(q.dtype), v), dim=-1)

    q_conjugate = conjugate_quat(q)

    v_out = quat_mul(quat_mul(q, v_quat), q_conjugate)
    
    v_out = v_out[..., 1:4]
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

    x = torch.tensor([1.0, 0.0, 0.0], dtype=q.dtype, device=q.device)
    y = torch.tensor([0.0, 1.0, 0.0], dtype=q.dtype, device=q.device)
    z = torch.tensor([0.0, 0.0, 1.0], dtype=q.dtype, device=q.device)

    batch_size = q.shape[:-1]
    x = x.broadcast_to((*batch_size, 3))
    y = y.broadcast_to((*batch_size, 3))
    z = z.broadcast_to((*batch_size, 3))

    x = quat_vector_mul(q, x)
    y = quat_vector_mul(q, y)
    z = quat_vector_mul(q, z)

    R = torch.stack((x, y, z), dim=-1)


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

    t = t.unsqueeze(-1)

    T = torch.cat((R, t), dim=-1)
    # print(T.shape)
    pad = torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device)
    pad = pad.unsqueeze(0)

    pad_dim = (*T.shape[:-2],) + (1, 4)
    # print(pad_dim)
    pad = torch.broadcast_to(pad, pad_dim)


    T = torch.cat((T, pad), dim=-2)

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return T

def warp_3d_point(T, x):
    """
    Warps a 3D point through a homogenous 4x4 transform. This means promoting the
    point to homogenous coordinates by padding with 1, multiplying it against the
    4x4 matrix T, then cropping the first three coordinates.

    Args:
        T (torch.tensor): Homogenous 4x4 transform of shape (*, 4, 4).
        x (torch.tensor): 3D points of shape (*, 3).

    Returns:
        torch.tensor: Warped points of shape (*, 3).
    """

    x_warped = None
    device = x.device
    dtype = x.dtype

    ##########################################################################
    # TODO: Implement the method according to the method description.        #
    ##########################################################################

    # print(x.shape)
    # print(T.shape)

    pad = torch.ones(x.shape[:-1]+(1,), dtype=dtype, device=device)
    x = torch.cat((x, pad), dim=-1)
    x = x.unsqueeze(-1)

    # print(x.shape)
    
    x_warped = T @ x
    x_warped = x_warped.squeeze(-1)
    x_warped = x_warped[..., 0:3]

    

    # print(x_warped.shape)
    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return x_warped
    
    

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

    R = create_3x3_rotation(ex, ey)
    T = assemble_4x4_transform(R, translation)

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

    R = T[..., 0:3, 0:3] # (*, 3, 3)
    t = T[..., 0:3, 3] # (*, 3)
    t = t.unsqueeze(-1)

    R_t = R.transpose(-1, -2)

    new_t = - R_t @ t # (*, 3, 1)
    new_t = new_t.squeeze(-1) # (*, 3)

    inv_T = assemble_4x4_transform(R_t, new_t)


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
    device = phi.device
    dtype = phi.dtype
    phi1, phi2 = torch.unbind(phi, dim=-1)
    T = None

    ##########################################################################
    # TODO: Build the rotation matrix described above. Assemble it together  # 
    #   with a translation of 0 to a 4x4 transformation.                     #
    ##########################################################################

    T = torch.eye(4, dtype=dtype, device=device)
    if len(batch_shape) > 0:
        T = torch.broadcast_to(T, batch_shape+(4,4)).clone()
    T[..., 1, 1] = phi1
    T[..., 1, 2] = -phi2
    T[..., 2, 1] = phi2
    T[..., 2, 2] = phi1

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

    non_chi_transforms = torch.zeros((20, 4, 4, 4))

    identity = torch.eye(4).unsqueeze(0).unsqueeze(0) # (1, 1, 4, 4)
    identity = identity.broadcast_to((20, 2, 4, 4)).clone()

    non_chi_transforms[:, 0:2, :, :]  = identity

    for i, entries in enumerate(rigid_group_atom_position_map.values()):

        phi_group = create_4x4_transform(entries['N'] - entries['CA'], torch.tensor([1.0, 0.0, 0.0]), entries['N'])
        # phi_group = phi_group.unsqueeze(0).unsqueeze(0)

        non_chi_transforms[i, 2, : :] = phi_group

        psi_group = create_4x4_transform(entries['C'] - entries['CA'], entries['CA'] - entries['N'], entries['C'])
        # psi_group = psi_group.unsqueeze(0).unsqueeze(0)

        non_chi_transforms[i, 3, : :] = psi_group
        

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

    chi_transforms = torch.zeros((20, 4, 4, 4))

    for i, (aa, entries) in enumerate(rigid_group_atom_position_map.items()):
        for j in range(4):
            if not chi_angles_mask[i][j]:
                chi_transforms[i, j, :, :] = torch.eye(4)
                continue
            
            # if j == 0:
            #     prev_atom = entries['N']
            #     current_atom = entries['CA']
            # else:
            #     prev_atom = current_atom
            #     current_atom = next_atom
            
            # next_atom = entries[chi_angles_chain[aa][j]]
            #chi_transforms[i, j, :, :] = create_4x4_transform(next_atom-current_atom, prev_atom-current_atom, next_atom)

            next_atom = chi_angles_chain[aa][j]

            if j==0:
                ex = entries[next_atom] - entries['CA']
                ey = entries['N'] - entries['CA']
                
            else:
                ex = entries[next_atom]
                ey = torch.tensor([-1.0,0.0,0.0])

            chi_transforms[i, j, :, :] = create_4x4_transform(ex, ey, entries[next_atom])

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

    rigid_transforms = torch.cat((calculate_non_chi_transforms(), calculate_chi_transforms()), dim=1)

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
    device = T.device
    dtype = T.dtype

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

    alpha = alpha / LA.vector_norm(alpha, dim=-1, keepdim=True)
    (omega, phi, psi, chi1, chi2, chi3, chi4) = torch.unbind(alpha, dim=-2)

    all_rigid_transforms = precalculate_rigid_transforms().to(device=device, dtype=dtype)

    local_transforms = all_rigid_transforms[F] # (N_res, 8, 4, 4)

    global_transforms = torch.zeros_like(local_transforms)

    global_transforms[..., 0, :, :] = T

    for i, angle in enumerate([omega, phi, psi, chi1], start=1):
        global_transforms[..., i, :, :] = T @ local_transforms[..., i, :, :] @ makeRotX(angle)
    
    for i, angle in enumerate([chi2, chi3, chi4], start=5):
        global_transforms[..., i, :, :] = global_transforms[..., i-1, :, :] @ local_transforms[..., i, :, :] @ makeRotX(angle)


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
    device = T.device
    dtype = T.dtype
    N_res = T.shape[-3]

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

    global_frames = compute_global_transforms(T, alpha, F) # (N_res, 8, 4, 4)
    
    atom_local_positions = residue_constants.atom_local_positions.to(device) # (20, 37, 3)
    atom_frame_inds = residue_constants.atom_frame_inds.to(device) # (20, 37)
    atom_mask = residue_constants.atom_mask.to(device) # (20, 37)

    atom_local_positions = atom_local_positions[F, ...] # (..., N_res, 37, 3)
    atom_frame_inds = atom_frame_inds[F, ...] # (..., N_res, 37)
    atom_mask = atom_mask[F, ...] # (..., N_res, 37)

    gather_index = atom_frame_inds.unsqueeze(-1).unsqueeze(-1) # (..., N_res, 37, 1, 1)
    gather_index = gather_index.expand(*atom_frame_inds.shape, 4, 4) # (..., N_res, 37, 4, 4)

    # gather => shape (..., N_res, 37, 4, 4)
    selected_frames = torch.gather(global_frames, dim=-3, index=gather_index)

    ones = torch.ones(*atom_local_positions.shape[:-1], 1, dtype=dtype, device=device)

    atom_local_positions = torch.cat((atom_local_positions, ones), dim=-1) # (N_res, 37, 4)

    global_positions = torch.einsum('...ij,...j->...i', selected_frames, atom_local_positions) # (N_res, 37, 4)
    
    global_positions = global_positions[..., :3]

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return global_positions, atom_mask

