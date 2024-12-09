The policy gradient with importance weighting, used for instance in PPO, is given by:
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ 
\frac{\nabla_\theta \pi_\theta(s_t, a_t)}{\pi_{\theta_{\text{old}}}(s_t, a_t)} A(s_t, a_t)
\right]
$$
However, so far we have studied policy gradient formulations containing the score function:
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) \cdot A(s, a) \right]
$$
Why is there no logarithm in Eq. (1)? Show that Eq. (1) is correct under the assumption that the difference in the state visitation distribution $\mu(s)$ between
the old and new policies can be ignored. Consider the expectation that needs to be computed, referencing slide 6 ("Recap: Policy Gradient") in lecture notes 8, 
but with respect to the old parameters.
