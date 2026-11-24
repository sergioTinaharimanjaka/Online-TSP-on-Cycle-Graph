# Online-TSP-on-Cycle-Graph
This is an implementation of the Algorithm $RandHeavyTest_{0.5}$ on the paper [link](https://arxiv.org/abs/2607.11203). In this Travelling Salesman Problem, given a cycle graph $G = (V, E)$ with nonnegative edges and source $s$. Each vertex represents a city, each edge the possible connection between two cities, and the weight of each edge the cost to be paid when moving between two cities(This is why it is nonnegative). The objective is to visit all cities(can be visited many times) starting from the given source $s$(vertex' index) and coming back to the source with a minimum cost. 

The agent does not know the graph in advance. They just discover two possibe edges(to the left or to the right) on the cycle graph at each step. They have to choose one and cannot catch-up in the future when doing the wrong choice.

The competitive ratio measure the efficiency of the algorithm. It is the cost found by the online algorithm divided by the cost of the offline algorithm noted $OPT$. The cost of the online algorithm($RandHeavyTest_{0.5}$) is given by the expectation since it is random. And $OPT = min\left(\sum_{e\in E}w(e), 2\sum_{e\in A \setminus h}w(e)\right)$ where $h$ is a heavy edge. An edge $h$ is heavy if $\sum_{e\in A \setminus h}w(e) < w(h).$

Let'say that we have the following flow at some step $t$ of the the algorithm. $s → a_t→ b_t→d_t→s.$ For this explanation, assume that $a_t$ is the path discovered by agent in the left(clockwise direction) at time $t$, $d_t$ in the counter clockwise direction (the boundary edge of $d_t$ is not visited yet) and $b_t$ the boundary edge at the branch $a_t$. It can be the opposite direction but the same principle.

At this time $t$, the agent chooses randomly between $b_t$ and the boundary edge on $d_t$ with a probability $p_{dir} =
1$ if $b_t \le 1.5a_t + d_t$ and  
$\frac{a_t}{2(b_t - a_t -d_t)}$ otherwise.



So, at this time $t$ if the probability is not $1$, two terms with probability $p_t$ and $1 - p_t$ appear to compute the expectation. And for a given graph with $n$ vertices, the algorithm can find $O(n^2)$ possible probabilities. It is quite hard to prove the bound of the competitive ratio using this $O(n^2)$ probabilities then the paper proposed the poetential function $\Phi$ which is a supermatingale. They proved that the bound for any configuration of $G$ will give a competitive ratio less than $\frac{(3 - \sqrt{2})^2}{8}$ by analyzing different value of $b_t$. 

This competitive ratio can be reduced to $1.307$ if we consider other edge in different position apart from $b_t$. They are the heaviest edge $b'$(which is the heavy edge if the cycle contains one) and $b''$ which is the second heaviest after the heaviest one. There is another proof for that one with the notion of $T_{stop}$ which will be published in another paper. For now, we just nedd the simulation.

The function $probOneStep$ update the branches $a$, $d$, the boundary edge $b$, the cost $C$paid by the agent after one step. The cycle graph is stored in a list $Graph$. $clockwise$ informs the direction of the last move. $False$ if it was in the left and $True$ otherwise. 

$Graph[i]$ refers to the edge before the vertex $i$ in the counter clockwise direction if the boolean clockwise is $False$.
It is the edge before the vertex $i$ in the clockwise direction if clockwise is $True$.

$ag pos$ is the index of Graph such that $Graph\[ag pos\] = b$. Similar to $b$, $back pos$ is the index of Graph such that $Graph\[back pos\]$ is the unvisited vertex of the branch $d$. Initially $ag pos = back pos = s.$

$probOneStep$ is just computing the new cost after one step according to the random value compared with the $pdir$ at this step, not the expectation.


```python
import numpy as np
```
```python
def probOneStep(G, N, a, b, d, ag_pos, back_pos, clockwise, C, rng):
  if b <= 1.5 * a + d:#Move direct
    C += b
    p = 1
    if clockwise == False:
      a = a + G[ag_pos]
      b = G[(ag_pos - 1) % N]
      ag_pos = (ag_pos - 1) % N
    else:#clockwise = True
      a = a + G[(ag_pos + 1) % N]
      b = G[(ag_pos + 2) % N]
      ag_pos = (ag_pos + 1) % N
  else:#Backtracking with probability 1 - p
    p = a / (2 * (b - a - d))
    if clockwise == False:
      #C += p * b + (1 - p) * (a + d)
      if rng.uniform(low=0, high=1) <= 1 - p:
        C += a + d
        temp = back_pos
        back_pos = ag_pos
        temp_d = d
        d = a + b
        a = temp_d#We are adding the last unvisited edge of the previous d.
        ag_pos = (temp + 1) % N
        b = G[(ag_pos + 1) % N]
        clockwise = True
      else:#go direct
        #print("Go direct")
        C += b
        a = a + G[ag_pos]
        b = G[(ag_pos - 1) % N]
        ag_pos = (ag_pos - 1) % N
    else:#clockwise is True
      if rng.uniform(low=0, high=1) <= 1 - p:
        C += a + d
        temp = back_pos
        back_pos = ag_pos
        temp_d = d
        d = a + b
        a = temp_d
        ag_pos = (temp - 1) % N
        b = G[ag_pos]
        clockwise = False
      else:#go direct
        #print("Go direct")
        C += b
        a = a + G[(ag_pos + 1) % N]
        b = G[(ag_pos + 2) % N]
        ag_pos = (ag_pos + 1) % N
  #print("p = ", p, "a = ", a, " b = ", b, " d = ", d, " ag_pos = ", ag_pos, " back_pos = ", back_pos, " clockwise = ", clockwise, "Cost = ", C)
  return G, a, b, d, ag_pos, back_pos, clockwise, C
```




The function $findOpt$ compute $OPT$.






```python
def find_opt(G):#G is a numpy array
  summation = np.sum(G)
  heaviest = np.max(G)
  return min(2 * (summation - heaviest), summation)
  ```

$Randheavytets05(G, source)$ repeats the step $n - 2$ times where $n$ is the number of edges. $n-2$ because the algorithm choose with probability $1$ the cheapest between the $2$ proposed edges in the begining and when the last edge is discovered at the end of the algorithm, the agent goes back by choosing the cheapest branch.  But there is some required initialization to be consistent with the index and direction.

```python
def Randheavytest05(G, source):
  N  = len(G)
  G = np.array(G)
  s = 0 #must be < N
  clockwise = False
  #Assume that when moving to the left branch, index decreases, and increases in the opposite direction.
  #First move should be to the left for reference. The array Graph is adapted to that.
  if G[source] > G[source + 1]:
    L = list(G)
    L.reverse()
    G = np.array(L)
    source = N - 1 - source
    clockwise = False

  a, b, d = G[source % N], G[(source - 1) % N], G[(source + 1) % N]#It is correct since clockwise = False
  ag_pos, back_pos = (source - 1) % N, source#definition of ag_pos and back_pos are inside the function prob_one_step
  C = a # Cost paid by the algorithm
  rng = np.random.default_rng()
  for i in range(N - 2):
    G, a, b, d, ag_pos, back_pos, clockwise, C = probOneStep(G, N, a, b, d, ag_pos, back_pos, clockwise, C, rng)
  det = min(a, b + d)
  C += det
  OPT = findOpt(G)
  return C / OPT
```

