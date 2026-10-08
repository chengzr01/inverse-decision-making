# General Threshold Model

For each node \(v\):

- Let \(N(v)\) be its neighbors that can influence it.
- Define a function \(g_v(X)\in[0,1]\) for each subset \(X\subseteq N(v)\), representing the combined influence of those neighbors.
- Require **monotonicity**: if \(X\subseteq Y\), then \(g_v(X)\le g_v(Y)\). More active neighbors cannot reduce influence.
- Draw a threshold \(\theta_v\) uniformly from \([0,1]\), once at the beginning.

Starting from an initially active set \(S\), adoption proceeds in discrete rounds. An inactive node \(v\) becomes active when

\[ g_v(X)\ge\theta_v, \]

where \(X\) is its set of currently active neighbors. Once active, it stays active.

Unlike the Linear Threshold Model, this model can express rules such as “adopt when at least two relatives **and** three coworkers adopt.” The uniform threshold assumption is flexible: other threshold distributions can be incorporated by changing \(g_v\).

The key idea is that **influence must increase with adoption, but need not be additive**. Monotonicity alone does not require diminishing returns. 



In the model, the goal is to choose a **set of \(k\) initial adopters** that maximizes the expected total number of active nodes after diffusion ends. Writing this expected spread as \(f(S)\), the optimization problem is

\[ \max_{|S|=k} f(S). \]

The analysis concerns the influence of the set collectively, since different seeds can reach overlapping populations.

The chapter develops three main points:

1. **The general problem is computationally hard.** Finding the optimal seed set is NP-hard. With unrestricted monotone threshold functions \(g_v\), even approximating the optimum within a factor of \(n^{1-\varepsilon}\) is NP-hard for any fixed \(\varepsilon>0\). Critical-mass effects can make a particular combination of seeds enormously more effective than its individual members.

2. **Submodularity enables a good approximation.** If every local influence function \(g_v\) is submodular, then the global expected spread \(f\) is also submodular (Theorem 4.4). Thus, for \(X\subseteq Y\) and \(u\notin Y\),

   \[ f(X\cup\{u\})-f(X)\ \ge\ f(Y\cup\{u\})-f(Y). \]

   Adding another seed yields diminishing marginal benefit as the existing seed set grows. The Linear Threshold Model satisfies this condition.

3. **Greedy selection exploits this property.** Start with \(S=\varnothing\), then repeatedly add the node with the largest marginal increase in expected spread. With exact evaluations, the resulting set achieves at least

   \[ f(S_{\mathrm{greedy}})\ge(1-1/e)f(S^*)\approx0.632\,f(S^*). \]

   Exact evaluation of expected spread is itself computationally intractable, but sufficiently accurate estimates give a \(1-1/e-\varepsilon\) guarantee.

The central result is the **local-to-global connection**: diminishing returns in each node’s influence function produces diminishing returns in expected network-wide spread, making greedy seed selection provably effective. 