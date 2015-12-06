#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <math.h>

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>


/* Size of input graphs - note this code will only work for size smaller than 62 */
#define N 19

/* Size of graph with just two vertices in one bipartition */
#define NSMALL 17

/* Size/valency of the original graph */
#define NTOT 76 
#define VAL 30

#define BIPART_MIN 15
#define BIPART_MAX 18

/* Forbidden eigenvalue of the star complement */
#define SC_EIG 2

/* number of edges that can be added to our graph when extending */
#define NEDGES (( BIPART_MAX-BIPART_MIN + 1) * (BIPART_MAX-BIPART_MIN)/2)

#define EPS 0.000001

/* graph6 related things */
#define G6LEN(n)  (((n)*((n)-1)/2+5)/6+1) 
#define BIAS6 63
#define TOPBIT6 32

/* The current graph (either read or expanded) given as an adjacency matrix */
static gsl_matrix *adj;
static gsl_matrix *det_adj;
static gsl_matrix *partitioned_adj;

static gsl_matrix *adj_small;
static gsl_matrix *partitioned_adj_small;

static gsl_permutation *p; 
static gsl_vector_complex *eval;
static gsl_eigen_nonsymm_workspace *w;

static gsl_vector_complex *eval_small;
static gsl_eigen_nonsymm_workspace *w_small;


/* After each call to adjtog6, gcode stores the graph6 
   string of the graph represented by adj */
static char gcode[G6LEN(N)+3]; 

unsigned  edges[NEDGES][2]; 

const int eigenvalues[NTOT] = {30, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8};


/* convert nauty graph to graph6 string, including \n and \0 */
char *adjtog6() {
    int i,j,k;
    char *p,x;

    p = gcode;
    *p++ = BIAS6+N;

    k = 6;
    x = 0;
    
    for (j = 1; j < N; ++j) {
        for (i = 0; i < j; ++i) {
            x <<= 1;
            if (gsl_matrix_get(adj, j,i) == 1) {
                x |= 1;
            }                

            if (--k == 0) {
                *p++ = BIAS6 + x;
                k = 6;
                x = 0;
            }
        }
    }

    if (k != 6) {
        *p++ = BIAS6 + (x << k);
    }        

    *p++ = '\n';
    *p = '\0';

    return gcode;
}

/* 
    Returns 1 if and only if the graph G represented by adj
 
  1. does not have SC_EIG as an eigenvalue
  2. it is a good interlacing candidate for the 
     eigenvalues of the original graph.

  The function expects partitioned_adj to be a matrix of size m_size+1   
*/

/* TODO make it work so that it iterates over half the matrix */
static gsl_matrix *partitioned_am(gsl_matrix *adj, gsl_matrix *partitioned_adj, unsigned n) {

    unsigned i,j;

    unsigned total_edges = 0; 

    gsl_matrix_set_zero(partitioned_adj);

    for (i= 0; i < n; i++) {
        unsigned deg = 0;
        for (j = 0; j < n; j++) {
            unsigned el = gsl_matrix_get(adj,i,j);
            if (el)  {
                gsl_matrix_set(partitioned_adj, i,j,1);
                deg+=1;
            }   
        }
        gsl_matrix_set(partitioned_adj,i, n, VAL-deg);
        gsl_matrix_set(partitioned_adj, n,i, (double)(VAL-deg)/(NTOT-n));
        total_edges += deg;
    }
    gsl_matrix_set(partitioned_adj,n, n, (double) 2*(NTOT*VAL/2 + total_edges/2 - n*VAL)/(NTOT-n));

    return partitioned_adj;
}

static int cmp(const void* elem1, const void* elem2) {

    const double *a = elem1, *b = elem2; 
    
    return *a > *b ? -1 : *a < *b ? 1 : 0;
}

static double *spectrum(gsl_matrix *m,gsl_vector_complex *eval,  
        gsl_eigen_nonsymm_workspace *w, unsigned size) {

    /* N+1 is an upper bound on eigs, in practice it can be smaller */
    static double eigs[N+1];
    unsigned i;

    gsl_eigen_nonsymm (m, eval, w);    

    for(i = 0; i < size ; i++) {
        gsl_complex eval_i = gsl_vector_complex_get (eval, i);
        eigs[i] = GSL_REAL(eval_i);
    }

    qsort(eigs,size, sizeof(double),cmp); 
    return eigs;
}

/* 
   This function returns 1 if and only if 
   the seqeucne eigs_sc interlaces eigs 
*/
static unsigned does_interlace(double eigs_sc[N+1], unsigned n) {

    unsigned i;

    for (i = 0; i < n; i++) {

        double expr = eigenvalues[i]-eigs_sc[i];

        if (expr <= EPS && fabs(expr) >= EPS) { 
            return 0;    
        }

        expr = eigs_sc[i] - eigenvalues[NTOT-n+i];

        if (expr <= EPS && fabs(expr) >= EPS) {
            return 0;
        } 
    }

    return 1; 
}


static unsigned is_valid_sc_cand(gsl_matrix *adj) {

    unsigned i;

    gsl_matrix_memcpy(det_adj, adj);

    for (i = 0; i < N; i++) {
        gsl_matrix_set(det_adj,i,i,-SC_EIG);   
    }

    int signum;

    gsl_linalg_LU_decomp(det_adj, p , &signum);
    double  det = gsl_linalg_LU_det(det_adj, signum);

    if (fabs(det) < EPS) {
        return 0;
    }        

    /* we now make sure the interlacing is satisfied */
    gsl_matrix *m = partitioned_am(adj,partitioned_adj,N);

    double *eigs = spectrum(m,eval, w, N+1);
    
    return does_interlace(eigs,N+1);
}

/* Given the current graph stored in adj this function
   computes the number of valid edges that can be added
   to its bipartition 

   If there is a pair of vertices inducing a bipartition
   that does not interlace as well after adding an edge,
   then we return NEDGES+1.

   forced_edges is a bitmask containing the edges that must
   be present in any extension.
*/
static unsigned valid_edges(unsigned *forced_edges) {
    
    unsigned i,j, ret = 0;
    unsigned k;

    *forced_edges = 0;

    for (i = BIPART_MIN ; i <= BIPART_MAX; i++) {
        for (j = i+1; j <= BIPART_MAX; j++) {
            /* We have chosen the edge (i,j). We now create the adjacency matrix
               of the graph having just i,j in the respective bipartition and
               check if it interlaces the original graph. */
            gsl_matrix_set_zero(adj_small);

            for (k = 0; k < BIPART_MIN; k++) {
                double val = gsl_matrix_get(adj, k,i);
                if (val) {
                    gsl_matrix_set(adj_small, k,BIPART_MIN,1);
                    gsl_matrix_set(adj_small, BIPART_MIN, k,1);
                }

                val = gsl_matrix_get(adj, k,j);
                if (val) {
                    gsl_matrix_set(adj_small, k, BIPART_MIN+1,1);
                    gsl_matrix_set(adj_small, BIPART_MIN+1, k,1);
                }

            }
            /* An edge can be present in the bipartition or not. If both
               cases do not interlace then the graph in question is not
               good. */
            
            gsl_matrix *m = partitioned_am(adj_small, partitioned_adj_small, NSMALL);
            double *eigs = spectrum(m, eval_small, w_small, NSMALL+1);

            unsigned no_edge = does_interlace(eigs, NSMALL+1);

            gsl_matrix_set(adj_small, BIPART_MIN+1, BIPART_MIN, 1);
            gsl_matrix_set(adj_small, BIPART_MIN, BIPART_MIN+1, 1);
        
            m = partitioned_am(adj_small, partitioned_adj_small, NSMALL);
            
            eigs = spectrum(m, eval_small, w_small, NSMALL+1);

            unsigned edge = does_interlace(eigs, NSMALL+1);

            if (no_edge == 0 && edge == 0) {
                return NEDGES+1;
            }

            if (edge) {     
                if (no_edge == 0) {
                    *forced_edges |= (1<<ret);
                }
                edges[ret][0] = i;
                edges[ret][1] = j;
                ret++;
            }
        }
    }
    return ret;
}

static void expand() {

    unsigned i,j,x,y;

    unsigned forced_edges;
    unsigned num_cand_edges = valid_edges(&forced_edges);

    /* THIS graph is not extendible. */
    if (num_cand_edges == NEDGES+1) {
        return;
    }

    for (i = 0; i < 1U<<num_cand_edges; i++) {
        if ( (i & forced_edges) != forced_edges) {
            continue;
        }

        unsigned edges_set[NEDGES][2];
        unsigned num_edges_set = 0;

        for (j = 0; j < num_cand_edges; j++) {
            if (i & (1<<j)) {
                x = edges[j][0];
                y = edges[j][1];
                gsl_matrix_set(adj, x,y, 1);
                gsl_matrix_set(adj, y,x, 1);

                edges_set[num_edges_set][0] = x;
                edges_set[num_edges_set][1] = y;

                num_edges_set++;
            }
        }        
        
        /* here we have an expanded graph */
        if (is_valid_sc_cand(adj)) {
            printf("%s", adjtog6());
        }

        /* ... cleanup ...*/
        for (j = 0; j < num_edges_set;j++) {
            x = edges_set[j][0];
            y = edges_set[j][1];

            gsl_matrix_set(adj, x, y, 0);
            gsl_matrix_set(adj, y, x, 0);
        }
    }
}

static void stringtoadj(char *s) {

    char *p;
    int i,j,k,x = 0;

    /* Clear the adjacency matrix */
    gsl_matrix_set_zero(adj);

    p = s + 1;

    k = 1;
    for (j = 1; j < N; ++j) {
        for (i = 0; i < j; ++i) {
            if (--k == 0) {
                k = 6;
                x = *(p++) - BIAS6;
            }
	    
            if (x & TOPBIT6) {
                gsl_matrix_set(adj,i,j,1);
                gsl_matrix_set(adj,j,i,1);
            }
            x <<= 1;
        }
    }
}
static unsigned long long nproc = 0;

#define CHUNK 71146749

void sig_handler(int signo) {
    fprintf(stderr, "Current progress %.2f\n", 100.0*nproc/CHUNK);
}


int main(int argc, char *argv[]) {

    static FILE *infile;

    if (argc < 1) {
        return 1; 
    } 

    char line[G6LEN(N)+2];
    
    infile = fopen(argv[1], "r");

    adj = gsl_matrix_calloc(N, N);
    det_adj = gsl_matrix_calloc(N, N);
    partitioned_adj = gsl_matrix_calloc(N+1, N+1);

    adj_small = gsl_matrix_alloc(NSMALL, NSMALL);
    partitioned_adj_small = gsl_matrix_alloc(NSMALL+1, NSMALL+1);
    
    p = gsl_permutation_calloc(N);
    
    eval = gsl_vector_complex_alloc(N+1);
    w = gsl_eigen_nonsymm_alloc(N+1);

    eval_small = gsl_vector_complex_alloc(NSMALL+1);
    w_small = gsl_eigen_nonsymm_alloc(NSMALL+1);
    
    if (!w_small || !eval_small ||!adj_small ||!partitioned_adj_small) {
        fprintf(stderr, "Error allocating required space.");
        return 1;
    }    

    if (!infile || !adj || !p || !partitioned_adj || !eval) { 
        fprintf(stderr, "Error allocating required space.");
        return 1;
    }

    signal(SIGUSR1, sig_handler);

    while (1) {
        if (fgets(line, sizeof(line),infile) == NULL) 
            break;

        stringtoadj(line);
        expand();
        nproc++;
	}

    return 0;
}
