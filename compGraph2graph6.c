#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>

/* Size of input graphs - note this code will only work for size smaller than 62 */
#define N 21

/* Maximum order we expect for the the comparability graph */
#define MAX_ORDER 45000

/* Forbidden eigenvalue of the star complement */
#define SC_EIG 2

/* Specified precision used while doing floating point operations */
#define EPS 0.000001

/* graph6 related things */
#define SMALLN 62
#define G6LEN(n)  (((n)*((n)-1)/2+5)/6+(n<=SMALLN?1:4))
#define BIAS6 63
#define TOPBIT6 32
#define SMALLISHN 258047
#define MAXBYTE 126
#define C6MASK 63

static gsl_matrix *mat,*mat_inv;
static gsl_permutation *perm;

static gsl_vector *vecs[1<<N];
static gsl_vector *vecs_prod[1<<N];

static gsl_vector *vec_j;

/* storing the current graph6 string of our graph */
static char line[G6LEN(N)+2];

static char gcode[G6LEN(MAX_ORDER)+3]; 

/* number of currently processed graphs. */ 
static unsigned long nproc = 0;

static FILE *outFile;
static FILE *outDict;

static void stringtomat(char *s) {

	char *p;
	int i,j,k,x = 0;

    /* Clear the adjacency matrix */
    gsl_matrix_set_zero(mat);

    p = s + 1;

    k = 1;
    
    gsl_matrix_set(mat, 0, 0, SC_EIG);

    for (j = 1; j < N; ++j) {
        gsl_matrix_set(mat,j,j, SC_EIG);
        for (i = 0; i < j; ++i) {

            if (--k == 0) {
		        k = 6;
		        x = *(p++) - BIAS6;
            }
	    
            if (x & TOPBIT6) {
                gsl_matrix_set(mat,i,j,-1);
                gsl_matrix_set(mat,j,i,-1);
            }
            x <<= 1;
        }
    }

    int signum;
    gsl_linalg_LU_decomp(mat, perm , &signum);
    gsl_linalg_LU_invert(mat, perm, mat_inv);
}

/* Some graph6 string thingie */
static void encodegraphsize(int n, char **pp) {
    char *p;

    p = *pp;
    if (n <= SMALLN) 
        *p++ = BIAS6 + n;
    else {
        *p++ = MAXBYTE;
        *p++ = BIAS6 + (n >> 12);
        *p++ = BIAS6 + ((n >> 6) & C6MASK);
        *p++ = BIAS6 + (n & C6MASK);
    }
    *pp = p;
}

/* Given that mat_inv is the M = SC_EIG*I - A we 
   compute all binary vectors x such that x M x^t == SC_EIG
   and x M j^t == -1. 

   Finally for any pair x,y of such vectors we add an edge to our 
   if x M y^t is either 0 or -1.
*/

/* This is a global thingie since declaring it localy as 
 * an array of size 1<<N breaks the stack limit */
gsl_vector **verts;

static void constructGraph() {

    double res;
    unsigned i,j;

    /* After the first iteration this holds the number of 
       vertices of the obtained graph. The respective vertices
       are stored in vecs_prod[0]...vecs_prod[cache_size-1].
    */ 
    unsigned cache_size = 0; 

    for (i = 0; i < 1<<N; i++) {
        gsl_blas_dsymv(CblasUpper,1, mat_inv, vecs[i], 0, vecs_prod[cache_size]);

        gsl_blas_ddot(vecs_prod[cache_size], vec_j, &res);
    
        if (fabs(res+1) < EPS) {
            gsl_blas_ddot(vecs_prod[cache_size], vecs[i], &res);

            if (fabs(res-SC_EIG) < EPS) {
                verts[cache_size] = vecs[i];
                cache_size+=1;
            }                
        }
    }

    if (cache_size >= MAX_ORDER) {
        printf("Error. Obtained graph exceeds our limits. It has size %u\n", cache_size);
        exit(EXIT_FAILURE);
    }

    char *p = gcode;
    encodegraphsize(cache_size,&p);

    int k = 6,x = 0;

    for (i = 1; i < cache_size; i++) {
        for (j = 0; j < i; j++) {
            x <<= 1;
            gsl_blas_ddot(vecs_prod[i], verts[j], &res);

            /* We have an edge */
            if (fabs(res) <= EPS || fabs(res+1) <= EPS) {
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

    fputs(gcode,outFile);
    fputs(line,outDict);
    fputs(gcode,outDict);
}


static void init_vectors(void) {
    
    unsigned i,j;

    vec_j = gsl_vector_alloc(N);

    if (!vec_j) {
        puts("ERROR allocating vector space");
        exit(EXIT_FAILURE);
    }

    gsl_vector_set_all(vec_j, 1);

    for (i = 0; i < 1<<N; i++) {
        vecs[i] = gsl_vector_calloc(N);
        vecs_prod[i] = gsl_vector_alloc(N);

        if (!vecs[i] || !vecs_prod[i]) {
            puts("Error allocating vector space.");
            exit(EXIT_FAILURE);
        }

        /* We fill the i'th vector of vecs */
        for (j = 0; j < N; j++) {
            if ( i & (1<<j) ) {
                gsl_vector_set(vecs[i],j,1);
            }
        }
    }   
}


int main(int argc, char **argv) {
    
    static FILE *infile;

	if (argc < 2) {
        return 1; 

    } 
    
    infile = fopen(argv[1], "r");

    mat = gsl_matrix_alloc(N, N);
    mat_inv = gsl_matrix_alloc(N, N);
    perm = gsl_permutation_calloc(N);

    verts = malloc(sizeof(gsl_vector *) *(1<<N));

    if (!infile || !mat || !mat_inv || !perm || !verts) {
        puts("Error allocatin computational space.");
        exit(EXIT_FAILURE);
    }

    init_vectors();

    unsigned pr = 1;
    char buf[512];

    snprintf(buf, sizeof(buf), "%s.out", argv[1]);
    outFile = fopen(buf, "w");

    snprintf(buf, sizeof(buf), "%s.dict", argv[1]);
    outDict = fopen(buf, "w");
    
    while (1) {
        if (!fgets(line, sizeof(line), infile)) 
            break;

        stringtomat(line);
        constructGraph();
        nproc++;
        if (nproc % 183 == 0) {
            printf("%u\n", pr);
            pr += 1;
        }
	}
    printf("Successfuly processed: %lu graphs.\n" , nproc);
    return 0;
}
