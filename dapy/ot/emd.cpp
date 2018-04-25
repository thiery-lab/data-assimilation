/* This file is a C++ wrapper function for computing the transportation cost
 * between two vectors given a cost matrix.
 *
 * It was written by Antoine Rolet (2014) and mainly consists of a wrapper
 * of the code written by Nicolas Bonneel available on this page
 *          http://people.seas.harvard.edu/~nbonneel/FastTransport/
 *
 * It was then modified to make it more amenable to Python inline calling.
 *
 * Further small modifications and cleanup by Matt Graham.
 *
 * Please give relevant credit to the original author (Nicolas Bonneel) if
 * you use this code for a publication.
 *
 */

#include "emd.h"


/* Compute the earth mover distance and optimal transport matrix given
 * two 1D arrays of marginal probabilities X and Y and a 2D array D defining
 * the cost matrix. Note X, Y and D must all be C-contiguous and row-major.
 * Computed optimal transport matrix is written to 2D array pointed to by G
 * and optimal transport cost written to cost argument. Dual vector variables
 * corresponding to X and Y are written to alpha and beta respectively.
 */
int emd(int n1, int n2, double *X, double *Y, double *D, double *G,
        double* alpha, double* beta, double *cost, int maxIter)  {

    int n, m, cur;

    typedef FullBipartiteDigraph Digraph;
 
    DIGRAPH_TYPEDEFS(FullBipartiteDigraph);

    // Get the number of non zero coordinates in X and Y

    n=0;
    for (int i=0; i<n1; i++) {
        double val=*(X+i);
        if (val>0) {
            n++;
        } else if(val<0){
			return INFEASIBLE;
		}
    }

    m=0;
    for (int i=0; i<n2; i++) {
        double val=*(Y+i);
        if (val>0) {
            m++;
        } else if(val<0){
			return INFEASIBLE;
		}
    }

    // Define the graph

    std::vector<int> indI(n), indJ(m);
    std::vector<double> weights1(n), weights2(m);
    Digraph di(n, m);
    NetworkSimplexSimple<Digraph,double,double, node_id_type> net(
        di, true, n+m, n*m, maxIter);

    // Set supply and demand, don't account for 0 values (faster)

    cur=0;
    for (int i=0; i<n1; i++) {
        double val=*(X+i);
        if (val>0) {
            weights1[ cur ] = val;
            indI[cur++]=i;
        }
    }

    // Demand is actually negative supply...

    cur=0;
    for (int i=0; i<n2; i++) {
        double val=*(Y+i);
        if (val>0) {
            weights2[ cur ] = -val;
            indJ[cur++]=i;
        }
    }

    net.supplyMap(&weights1[0], n, &weights2[0], m);

    // Set the cost of each edge
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            double val=*(D+indI[i]*n2+indJ[j]);
            net.setCost(di.arcFromId(i*m+j), val);
        }
    }

    // Solve the problem with the network simplex algorithm

    int ret=net.run();
    if (ret==(int)net.OPTIMAL || ret==(int)net.MAX_ITER_REACHED) {
        *cost = 0;
        Arc a; di.first(a);
        for (; a != INVALID; di.next(a)) {
            int i = di.source(a);
            int j = di.target(a);
            double flow = net.flow(a);
            *cost += flow * (*(D+indI[i]*n2+indJ[j-n]));
            *(G+indI[i]*n2+indJ[j-n]) = flow;
            *(alpha + indI[i]) = -net.potential(i);
            *(beta + indJ[j-n]) = net.potential(j);
        }
    }

    return ret;
}
