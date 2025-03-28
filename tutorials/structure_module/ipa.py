import torch
import math
from torch import nn

from geometry.geometry import invert_4x4_transform, warp_3d_point

class InvariantPointAttention(nn.Module):
    """
    Implements invariant point attention, according to Algorithm 22.
    """
    
    def __init__(self, c_s, c_z, n_query_points=4, n_point_values=8, N_head=12, c=16):
        """
        Initializes the invariant point attention module. 

        Args:
            c_s (int): Number of channels for the single representation.
            c_z (int): Number of channels for the pair representation.
            n_query_points (int, optional): Number of query points for point attention. 
                Used for the embedding of q_points and k_points. Defaults to 4.
            n_point_values (int, optional): Number of value points for point attention. 
                Used for the embedding of v_points. Defaults to 8.
            n_head (int, optional): Number of heads for multi-head attention. Defaults to 12.
            c (int, optional): Embedding dimension for each individual head. Defaults to 16.
        """
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.n_query_points = n_query_points
        self.n_point_values = n_point_values
        self.N_head = N_head
        self.c = c


        ##########################################################################
        # TODO: Initialize the layers linear_q, linear_k, linear_v,              #
        #   linear_q_points, linear_k_points, linear_v_points, linear_b, and     # 
        #   linear_out. The embeddings for q, k and v are similar to             #
        #   MultiHeadAttention, except that they use bias (this clashes with the #
        #   supplement, but follows the official implementation).                #
        #   The point embeddings need to create three values per head and point. #
        #   They also use bias.                                                  #
        #   The embedding for the bias computes one bias value per head.         #
        #   For the input dimension of linear_out, count the channels of the     #
        #   various outputs in line 11 from the algorithm. If you have trouble   #
        #   with this, you can look below at the output description of           #
        #   `compute_outputs`. The output dimension of linear_out is c_s.        #
        #                                                                        #
        #   For the weight per head, gamma, initialize head_weights to a         #
        #   zero-tensor wrapped in nn.Parameter. Also, initialize nn.Softplus    #
        #   for the computation of gamma.                                        #
        ##########################################################################

        self.linear_q = nn.Linear(c_s, c*N_head) # (*, N_res, c_s) -> (*, N_res, N_head*c)
        self.linear_k = nn.Linear(c_s, c*N_head)
        self.linear_v = nn.Linear(c_s, c*N_head)

        self.linear_q_points = nn.Linear(c_s, 3*N_head*n_query_points) # -> (*, N_head*3*n_query_points) -> (*, 3, N_head, n_query_points)
        self.linear_k_points = nn.Linear(c_s, 3*N_head*n_query_points)
        self.linear_v_points = nn.Linear(c_s, 3*N_head*n_point_values) # -> (*, N_head*3*n_point_values) -> (*, 3, N_head, n_point_values)

        self.linear_b = nn.Linear(c_z, N_head) # (*, N_res, N_res, c_z) -> (*, N_res, N_res, N_head)

        self.linear_out = nn.Linear(N_head*(c_z+c+4*n_point_values), c_s)

        self.head_weights = nn.Parameter(torch.zeros(N_head))
        self.soft_plus = nn.Softplus()

        # a: (*, N_head, N_res, N_res)

        # o_dash_hi: (*, N_res, N_head, c_z)
        # o_hi: (*, N_res, N_head, c)
        # o_hpi: (*, N_res, N_head, n_point_values, 3)
        # o_hpi_norm: (*, N_res, N_head, n_point_values, 1)

        # out (*, N_res, N_head*(c_z+c+4*n_point_values)) -> (*, N_res, c_s) 


        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def prepare_qkv(self, s):
        """
        Creates the standard attention embeddings q, k, and v, as well as the point 
        embeddings qp, kp, and vp, for invariant point attention.

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).

        Returns:
            tuple: A tuple consisting of the following embeddings:
                q: Tensor of shape (*, N_head, N_res, c)  
                k: Tensor of shape (*, N_head, N_res, c)
                v: Tensor of shape (*, N_head, N_res, c)
                qp: Tensor of shape (*, N_head, N_query_poitns, N_res, 3)
                kp: Tensor of shape (*, N_head, N_query_points, N_res, 3)
                vp: Tensor of shape (*, N_head, N_point_values, N_res, 3)
        """
        c = self.c
        n_head = self.N_head
        n_qp = self.n_query_points
        n_pv = self.n_point_values
        batch_dim = s.shape[:-1]

        embeddings = None

        ##########################################################################
        # TODO: Implement the embedding preparation in the following steps:      #
        #   - Pass s through all of the embedding layers.                        # 
        #   - Reshape the feature dimension of the embeddings so that q, k and v #
        #     have shape (*, N_head, c), qp and kp have shape                    #
        #     (*, 3, N_head, n_qp) and vp has shape (*, 3, N_head, n_pv).        #
        #   - Move the dimensions to match the shapes in the method description. # 
        ##########################################################################

        # (*, N_res, c_z) -> (*, N_res, N_head*c)
        q = self.linear_q(s) 
        k = self.linear_k(s)
        v = self.linear_v(s)

        # -> (*, N_res, N_head, c)
        q = q.view(*batch_dim,n_head, c)
        k = k.view(*batch_dim,n_head, c)
        v = v.view(*batch_dim,n_head, c)

        # -> (*, N_head, N_res, c)
        q = torch.transpose(q, -2, -3)
        k = torch.transpose(k, -2, -3)
        v = torch.transpose(v, -2, -3)

        # (*, N_res, c_z) -> (*, N_res, N_head*3*N_points)
        qp = self.linear_q_points(s) 
        kp = self.linear_k_points(s)
        vp = self.linear_v_points(s)

        # (*, N_res, 3, N_head, N_points)
        qp = qp.view(*batch_dim, 3, n_head, n_qp)
        kp = kp.view(*batch_dim, 3, n_head, n_qp)
        vp = vp.view(*batch_dim, 3, n_head, n_pv)

        # (*, N_head, N_res, 3, N_points)
        qp = torch.movedim(qp, -2, -4)
        kp = torch.movedim(kp, -2, -4)
        vp = torch.movedim(vp, -2, -4)

        # -> (*, N_head, N_points, N_res, 3)
        qp = torch.movedim(qp, -1, -3)
        kp = torch.movedim(kp, -1, -3)
        vp = torch.movedim(vp, -1, -3)

        embeddings = (q, k, v, qp, kp, vp)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return embeddings

    def compute_attention_scores(self, q, k, qp, kp, z, T):
        """
        Computes the attention scores for invariant point attention, 
        according to line 7 from Algorithm 22.

        Args:
            q (torch.tensor): Query embeddings of shape (*, N_head, N_res, c).
            k (torch.tensor): Key embeddings of shape (*, N_head, N_res, c).
            qp (torch.tensor): Query point embeddings of shape (*, N_head, N_query_points, N_res, 3).
            kp (torch.tensor): Key point embeddings of shape (*, N_head, N_query_points, N_res, 3).
            T (torch.tensor): Backbone transforms of shape (*, N_res, 4, 4).

        Returns:
            torch.tensor: Attention scores of shape (*, N_head, N_res, N_res).
        """

        att_scores = None

        ##########################################################################
        # TODO: Implement the method in the following steps:                     #
        #   - Compute wc, wl and gamma.                                          # 
        #   - Reshape gamma (formerly shape (N_head,) so that it's broadcastable #
        #     against the attention scores.                                      #
        #   - Scale q and compute the bias. Move the dimension of the bias so    # 
        #     that it matches the attention scores.                              #
        #   - Compute the qk term. You can use torch.einsum for this.            #  
        #   - Reshape the transforms so that they can be used for batched        # 
        #     matrix multiplication against the query and key points.            #
        #   - Use warp_3d_point to warp the query and key points through T.      # 
        #   - Compute the query points / key points term.                        # 
        #   - Compute the full attention scores.                                 # 
        ##########################################################################

        # compute scalars
        w_C = math.sqrt(2/(9*self.n_query_points))
        w_L = math.sqrt(1/3)
        gamma = self.soft_plus(self.head_weights) # (N_head,)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) # (N_head,1,1)

        # scale q
        q = q /math.sqrt(self.c)

        # compute bias
        b = self.linear_b(z) # (*, N_res, N_res, c_z) -> (*, N_res, N_res, N_head)
        b = torch.movedim(b, -1, -3) # -> (*, N_head, N_res, N_res)

        # compute qk (*, N_head, N_res, N_res).
        qk = q @ k.mT 

        # broadcast T
        T = T.unsqueeze(-4).unsqueeze(-4)
        T = torch.broadcast_to(T, qp.shape[:-1]+(4,4)) # -> (*, N_head, N_query_points, N_res, 4, 4)

        warped_qp = warp_3d_point(T, qp) # -> (*, N_head, N_query_points, N_res, 3)
        warped_kp = warp_3d_point(T, kp) # -> (*, N_head, N_query_points, N_res, 3)

        warped_qp = warped_qp.transpose(-1, -2) # -> (*, N_head, N_query_points, 3, N_res)
        warped_qp = warped_qp.unsqueeze(-1) # -> (*, N_head, N_query_points, 3, N_res, 1)

        warped_kp = warped_kp.transpose(-1, -2) # -> (*, N_head, N_query_points, 3, N_res)
        warped_kp = warped_kp.unsqueeze(-1) # -> (*, N_head, N_query_points, 3, N_res, 1)
        
        qk_point = warped_qp - warped_kp.mT # -> (*, N_head, N_query_points, 3, N_res, N_res)

        qk_point = torch.linalg.vector_norm(qk_point, dim=-3) # -> (*, N_head, N_query_points, N_res, N_res)

        qk_point = qk_point ** 2
        qk_point = torch.sum(qk_point, dim=-3, keepdim=False) # -> (*, N_head, N_res, N_res)

        qk_point = gamma * w_C * qk_point / 2

        att_scores = torch.softmax(w_L * (qk + b - qk_point), dim=-1)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return att_scores

    def compute_outputs(self, att_scores, z, v, vp, T):
        """
        Computes the different output vectors for the IPA attention mechanism:
        The pair output, the standard attention output, and the point attention output,
        as well as the norm of the point attention output.

        Args:
            att_scores (torch.tensor): Attention scores of shape (*, N_head, N_res, N_res).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).
            v (torch.tensor): Value vectors of shape (*, N_head, N_res, c).
            vp (torch.tensor): Value points of shape (*, N_head, N_point_values, N_res, 3).
            T (torch.tensor): Backbone transforms of shape (*, N_res, 4, 4).

        Returns:
            tuple: A tuple consisting of the following outputs:
                - output from the value vectors of shape (*, N_res, N_head*c).
                - output from the value points of shape (*, N_res, N_head*3*N_point_values).
                - norm of the output vectors from the value points of shape (*, N_res, N_head*N_point_values)
                - output from the pair representation of shape (*, N_res, N_head*c_z).
        """

        v_out, vp_out, vp_out_norm, pairwise_out = None, None, None, None

        ##########################################################################
        # TODO: Compute the different attention outputs in the following steps:  #
        #   - Compute the pairwise output, move the dimension so that they       # 
        #     are (**, N_head, c), then flatten the heads and channels.          #
        #   - Compute the value vector output, move the dimensions so that they  #
        #     are (**, N_head, c), then flatten the heads and channels.          #
        #   - Reshape the transforms so that they can be used for batched        # 
        #     matrix multiplication against the value points.                    #
        #   - Warp the value points, compute the point attention values, compute # 
        #     the inverse of the transforms with invert_4x4_transform            #
        #     and warp the value points back through them.                       #
        #   - Transpose the axes of the value points from ...hpic to ...ichp     # 
        #     (the letters mean N_head, point_values, N_res, c). You can use     #
        #     torch.einsum for this.                                             #
        #   - Compute the vector norms of the point values.                      # 
        #   - Flatten the trailing channel, N_head and N_point_value dims for    # 
        #     the value points and their norm.                                   #
        ##########################################################################

        # (*, N_head, N_res, N_res) *  (*, N_res, N_res, c_z)
        pairwise_out = torch.einsum('...hij,...ijc->...hic', att_scores, z) # -> (*, N_res, N_head, c_z)
        pairwise_out = pairwise_out.movedim(-3, -2).flatten(start_dim=-2) # -> (*, N_res, N_head*c_z)

        # (*, N_head, N_res, N_res) * (*, N_head, N_res, c)
        v_out = torch.einsum('...hij,...hjc->...hic', att_scores, v) # -> (*, N_head, N_res, c)
        v_out = v_out.movedim(-3, -2)
        v_out = v_out.flatten(start_dim=-2) # -> (*, N_res, N_head*c)

        # broadcast T
        T = T.unsqueeze(-4).unsqueeze(-4)
        T = torch.broadcast_to(T, vp.shape[:-1]+(4,4)) # -> (*, N_head, N_point_values, N_res, 4, 4)

        # warp vp
        vp = warp_3d_point(T, vp) # -> (*, N_head, N_point_values, N_res, 3)

        # compute vp attention (*, N_head, N_res, N_res) * (*, N_head, N_point_values, N_res, 3)
        vp_out = torch.einsum('...hij,...hpjc->...hpic', att_scores, vp) # -> (*, N_head, N_point_values, N_res, 3)

        # warp vp attention with T-1
        vp_out = warp_3d_point(invert_4x4_transform(T), vp_out) # -> (*, N_head, N_point_values, N_res, 3)
        vp_out = torch.einsum('...hpic->...ichp', vp_out) # -> (*, N_res, 3, N_head, N_point_values)

    
        vp_out_norm = torch.linalg.vector_norm(vp_out, dim=-3, keepdim=True)

        # vp_out -> (*, N_res, N_head*3*N_point_values)
        vp_out = vp_out.flatten(start_dim=-3)

        vp_out_norm = vp_out_norm.flatten(start_dim=-3) # -> (*, N_res, N_head*N_point_values)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return v_out, vp_out, vp_out_norm, pairwise_out
        
        

    def forward(self, s, z, T):
        """
        Implements the forward pass for InvariantPointAttention, as specified in Algorithm 22.

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).
            T (torch.tensor): Backbone transforms of shape (*, N_res, 4, 4).

        Returns:
            torch.tensor: Output tensor of shape (*, N_res, c_s).
        """

        out = None
        
        ##########################################################################
        # TODO: Implement the forward pass by combining all the methods above.   #
        ##########################################################################

        q, k, v, qp, kp, vp = self.prepare_qkv(s)

        att_scores = self.compute_attention_scores(q, k, qp, kp, z, T)

        v_out, vp_out, vp_out_norm, pairwise_out = self.compute_outputs(att_scores, z, v, vp, T)

        out = torch.cat((v_out, vp_out, vp_out_norm, pairwise_out), dim=-1)

        out = self.linear_out(out)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out
        
