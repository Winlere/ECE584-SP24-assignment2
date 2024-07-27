import torch
import torch.nn as nn


class BoundHardTanh(nn.Hardtanh):
    def __init__(self,*args, **kwargs):
        super(BoundHardTanh, self).__init__(*args, **kwargs)

    @staticmethod
    def convert(act_layer):
        r"""Convert a HardTanh layer to BoundHardTanh layer

        Args:
            act_layer (nn.HardTanh): The HardTanh layer object to be converted.

        Returns:
            l (BoundHardTanh): The converted layer object.
        """
        # Done: Return the converted HardTanH
        l = BoundHardTanh(act_layer.min_val, act_layer.max_val, act_layer.inplace)
        assert act_layer.min_val == -1 and act_layer.max_val == 1, "HardTanh bounds are not -1 and 1"
        return l

    def boundpropogate(self, last_uA, last_lA, start_node=None):
        """
        Propagate upper and lower linear bounds through the HardTanh activation function
        based on pre-activation bounds.

        Args:
            last_uA (tensor): A (the coefficient matrix) that is bound-propagated to this layer
            (from the layers after this layer). It's exclusive for computing the upper bound.

            last_lA (tensor): A that is bound-propagated to this layer. It's exclusive for computing the lower bound.

            start_node (int): An integer indicating the start node of this bound propagation

        Returns:
            uA (tensor): The new A for computing the upper bound after taking this layer into account.

            ubias (tensor): The bias (for upper bound) produced by this layer.

            lA( tensor): The new A for computing the lower bound after taking this layer into account.

            lbias (tensor): The bias (for lower bound) produced by this layer.

        """
        # These are preactivation bounds that will be used for form the linear relaxation.
        preact_lb = self.lower_l
        preact_ub = self.upper_u

        preact_ub = torch.max(preact_ub, preact_lb + 1e-10) 
        # still sound, avoid division by 0
        """
         Hints: 
         1. Have a look at the section 3.2 of the CROWN paper [1] (Case Studies) as to how segments are made for multiple activation functions
         2. Look at the HardTanH graph, and see multiple places where the pre activation bounds could be located
         3. Refer the ReLu example in the class and the diagonals to compute the slopes/intercepts
         4. The paper talks about 3 segments S+, S- and S+- for sigmoid and tanh. You should figure your own segments based on preactivation bounds for hardtanh.
         [1] https://arxiv.org/pdf/1811.00866.pdf
        """

        # You should return the linear lower and upper bounds after propagating through this layer.
        # Upper bound: uA is the coefficients, ubias is the bias.
        # Lower bound: lA is the coefficients, lbias is the bias.
        
        upper_d = torch.zeros_like(preact_ub) # slope
        upper_b = torch.zeros_like(preact_ub) 
        
        lower_d = torch.zeros_like(preact_ub) # slope
        lower_b = torch.zeros_like(preact_ub) 
        
        def _abstraction_0(): # cover no step point
            masks_below_min = preact_ub <= self.min_val
            lower_d[masks_below_min] = upper_d[masks_below_min] = 0
            lower_b[masks_below_min] = upper_b[masks_below_min] = self.min_val
            del masks_below_min #avoid mistakes
            
            masks_between = (preact_lb > self.min_val) & (preact_ub < self.max_val)
            lower_d[masks_between] = upper_d[masks_between] = 1
            lower_b[masks_between] = upper_b[masks_between] = 0
            del masks_between
            
            masks_above_max = preact_lb >= self.max_val
            lower_d[masks_above_max] = upper_d[masks_above_max] = 0
            lower_b[masks_above_max] = upper_b[masks_above_max] = self.max_val
            del masks_above_max
        
        def _abstraction_1(): #cover exactly one step point
            masks_cover_min = (preact_lb <= self.min_val) & (preact_ub >= self.min_val) & (preact_ub < self.max_val)
            upper_d[masks_cover_min] = (preact_ub[masks_cover_min] - (-1)) / (preact_ub[masks_cover_min] - preact_lb[masks_cover_min])
            upper_b[masks_cover_min] = -1 * preact_ub[masks_cover_min] * upper_d[masks_cover_min] + preact_ub[masks_cover_min]
            lower_d[masks_cover_min] = upper_d[masks_cover_min] # optimizable,
            lower_b[masks_cover_min] = lower_d[masks_cover_min] - 1
            del masks_cover_min

            masks_cover_max = (preact_lb > self.min_val) & (preact_lb <= self.max_val) & (preact_ub >= self.max_val)
            lower_d[masks_cover_max] = (1 - preact_lb[masks_cover_max]) / (preact_ub[masks_cover_max] - preact_lb[masks_cover_max])
            lower_b[masks_cover_max] = -1 * preact_lb[masks_cover_max] * lower_d[masks_cover_max] + preact_lb[masks_cover_max]
            upper_d[masks_cover_max] = lower_d[masks_cover_max] # optimizable
            upper_b[masks_cover_max] = -1 * upper_d[masks_cover_max] + 1
            del masks_cover_max
        
        def _abstraction_2(): #cover exactly two step points
            masks_cover_min_max = (preact_lb <= self.min_val) & (preact_ub >= self.max_val)
            upper_d[masks_cover_min_max] = (1 - (-1)) / (1 - preact_lb[masks_cover_min_max]) # optimizable
            upper_b[masks_cover_min_max] = -1 * upper_d[masks_cover_min_max] + 1
            lower_d[masks_cover_min_max] = (1 - (-1)) / (preact_ub[masks_cover_min_max] - (-1)) # optimizable
            lower_b[masks_cover_min_max] = lower_d[masks_cover_min_max] - 1
            del masks_cover_min_max

        _abstraction_0() # cover no step point
        _abstraction_1() # cover exactly one step point
        _abstraction_2() # cover exactly two step points
        
        def _abnormal_detect():
            print(f"preact_lb={preact_lb}\n preact_ub={preact_ub}")
            print(f"upper_d={upper_d}\n upper_b={upper_b}")
            print(f"lower_d={lower_d}\n lower_b={lower_b}")
            test_val = (preact_lb + preact_ub) / 2
            print(f"test_val={test_val}")
            ub_eval = upper_d * test_val + upper_b
            lb_eval = lower_d * test_val + lower_b
            print(f"ub_eval={ub_eval}\n lb_eval={lb_eval}")
            abnormal = torch.gt(lb_eval, ub_eval + 1e-4)
            print(f"abnormal {abnormal}")
            print(f"abnormal_preact_lb={preact_lb[abnormal]}\n abnormal_preact_ub={preact_ub[abnormal]}")
            print(f"abnormal_upper_d={upper_d[abnormal]}\n abnormal_upper_b={upper_b[abnormal]}")
            print(f"abnormal_lower_d={lower_d[abnormal]}\n abnormal_lower_b={lower_b[abnormal]}")
            print(f"abnormal_test_val={test_val[abnormal]}")
            print(f"abnormal_ub_eval={ub_eval[abnormal]}\n abnormal_lb_eval={lb_eval[abnormal]}")
            assert torch.le(lb_eval, ub_eval + 1e-4).all(), f"lb <= ub not hold. \nlb: {lb_eval} \nub: {ub_eval}. \non {torch.gt(lb_eval, ub_eval)}. \n layer: {self}"
    
        # lower_b = torch.ones_like(lower_b) * -2
        # lower_d = torch.ones_like(lower_d) * 0
        
        # upper_b = torch.ones_like(upper_b) * 2
        # upper_d = torch.ones_like(upper_d) * 0

        uA = lA = None
        ubias = lbias = 0
        # Should be last_uA.shape=torch.Size([2, 64, 128]), upper_d.shape=torch.Size([2, 1, 128]), upper_b.shape=torch.Size([2, 128])
        # print(f"last_uA.shape={last_uA.shape}, upper_d.shape={upper_d.shape}, upper_b.shape={upper_b.shape}")
        upper_d = upper_d.unsqueeze(1)
        lower_d = lower_d.unsqueeze(1)
        
        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0)
            neg_uA = last_uA.clamp(max=0)
            # Choose upper or lower bounds based on the sign of last_A
            # New linear bound coefficent.
            uA = upper_d * pos_uA + lower_d * neg_uA
            # New bias term. Adjust shapes to use matmul (better way is to use einsum).
            mult_uA_pos = pos_uA.view(last_uA.size(0), last_uA.size(1), -1)
            mult_uA_neg = neg_uA.view(last_uA.size(0), last_uA.size(1), -1)
            ubias =         mult_uA_pos.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
            ubias = ubias + mult_uA_neg.matmul(lower_b.view(lower_b.size(0), -1, 1)).squeeze(-1)
        if last_lA is not None:
            pos_lA = last_lA.clamp(min=0)
            neg_lA = last_lA.clamp(max=0)
            # Choose upper or lower bounds based on the sign of last_A
            # New linear bound coefficent.
            lA = upper_d * neg_lA + lower_d * pos_lA
            # New bias term. Adjust shapes to use matmul (better way is to use einsum).
            mult_lA_pos = pos_lA.view(last_lA.size(0), last_lA.size(1), -1)
            mult_lA_neg = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias =         mult_lA_pos.matmul(lower_b.view(lower_b.size(0), -1, 1)).squeeze(-1)
            lbias = lbias + mult_lA_neg.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)

        return uA, ubias, lA, lbias

