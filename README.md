# Online-TSP-on-Cycle-Graph
This is an implementation of the Algorithm $RandHeavyTest_{0.5}$ on the paper [link](https://arxiv.org/abs/2607.11203). In this Travelling Salesman Problem, given a cycle graph $G = (V, E)$ with nonnegative edges and source $s$. Each vertex represents a city, each edge the possible connection between two cities, and the weight of each edge the cost to be paid when moving between two cities(This is why it is nonnegative). The objective is to visit all cities(can be visited many times) starting from the given source $s$(vertex' index) and coming back to the source with a minimum cost. 

The agent does not know the graph in advance. They just discover two possibe edges(to the left or to the right) on the cycle graph at each step. They have to choose one and cannot catch-up in the future when doing the wrong choice.

The competitive ratio measure the efficiency of the algorithm. It is the cost found by the online algorithm divided by the cost of the offline algorithm noted $OPT$. The cost of the online algorithm($RandHeavyTest_{0.5}$) is given by the expectation since it is random. And $OPT = min\left(\sum_{e\in E}w(e), 2\sum_{e\in A \setminus h}w(e)\right)$ where $h$ is a heavy edge. An edge $h$ is heavy if $\sum_{e\in A \setminus h}w(e) < w(h).$

```python
def find_opt(G):#G is a numpy array
  summation = np.sum(G)
  heaviest = np.max(G)
  return min(2 * (summation - heaviest), summation)
  ```
