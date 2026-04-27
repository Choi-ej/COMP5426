#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <stdint.h>
#include <sys/resource.h>
#ifdef __linux__
#include <sched.h>   // for CPU pinning
#endif

#define MAX_THREADS 64

/* ---------- Thread Data ---------- */
typedef struct {
    int start_row, end_row; // row partitioning. Each thread processess only these rows -> avoid race conditions
    int rows, common, cols; // rows, common(shared dimmension in inner loop), columns
    // generic pointers to support float or double
    // matrices
    void *A;  // Left matrix
    void *B;  // Right matrix
    void *C;  // Result matrix
    int use_double; // datatype switch
} ThreadData;

/* ---------- Time ---------- */
double get_time() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

/* ---------- Safe Parsing ---------- */ // convert string -> int. valid input (non-numeric, negative val)
int parse_int(const char *s) {
    char *end;
    long val = strtol(s, &end, 10);
    if (*end != '\0' || val <= 0) return -1;
    return (int)val;
}

/* ---------- Random ---------- */
double rand_unit() {
    return (double)arc4random() / (double)UINT32_MAX; //generates values in [0,1]
}

/* ---------- Worker ---------- */ //parallael logic
/*
   Performs matrix multiplication:
   C = A × B

   A: rows × common
   B: common × cols
   C: rows × cols
*/
void* worker(void *arg) {
    ThreadData *d = (ThreadData*)arg;

#ifdef __linux__
    /* additional requirement: binds thread to a specific CPU core to maximise speedup
    cache locality, performance stability */ 
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(d->start_row % sysconf(_SC_NPROCESSORS_ONLN), &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif

    if (!d->use_double) { // float
        float *A = (float*)d->A;
        float *B = (float*)d->B;
        float *C = (float*)d->C;
        // i->k->j order, cache efficiency.
        // A accessed row-wise, B acessed row wise
        for (int i = d->start_row; i < d->end_row; i++) {
            for (int k = 0; k < d->common; k++) {
                // reduces repeated memory access
                float a_val = A[i * d->common + k];
                for (int j = 0; j < d->cols; j++) {
                    C[i * d->cols + j] += a_val * B[k * d->cols + j];
                }
            }
        }
    } else {
        double *A = (double*)d->A;
        double *B = (double*)d->B;
        double *C = (double*)d->C;

        for (int i = d->start_row; i < d->end_row; i++) {
            for (int k = 0; k < d->common; k++) {
                double a_val = A[i * d->common + k];
                for (int j = 0; j < d->cols; j++) {
                    C[i * d->cols + j] += a_val * B[k * d->cols + j];
                }
            }
        }
    }
    return NULL;
}

/* ---------- Parallel ---------- */
void parallel_mul(int threads, int rows, int common, int cols,
                  void *A, void *B, void *C, int use_double) {

    pthread_t tids[MAX_THREADS];
    ThreadData td[MAX_THREADS];

    int actual = (threads > rows) ? rows : threads;

    for (int i = 0; i < actual; i++) {
        // work distribution. evenly distribute, no overlap
        // 1D row partitioning
        td[i].start_row = i * rows / actual;
        td[i].end_row   = (i + 1) * rows / actual;

        td[i].rows = rows;
        td[i].common = common;
        td[i].cols = cols;

        td[i].A = A;
        td[i].B = B;
        td[i].C = C;
        td[i].use_double = use_double;
        // starts threads
        pthread_create(&tids[i], NULL, worker, &td[i]);
    }

    for (int i = 0; i < actual; i++) {
        // all threads complete before proceeding
        pthread_join(tids[i], NULL);
    }
}

/* ---------- Sequential ---------- */
void sequential_mul(int rows, int common, int cols,
                    void *A, void *B, void *C, int use_double) {

    ThreadData td = {0, rows, rows, common, cols, A, B, C, use_double};
    // This is sequential version but reusing multiplicating part
    worker(&td);
}

/* ---------- MAIN ---------- */
int main(int argc, char *argv[]) {
    // verifying 6 arguments
    if (argc != 7) {
        fprintf(stderr, "Usage: %s m k l n <float|double> threads\n", argv[0]);
        return EXIT_FAILURE;
    }

    int m = parse_int(argv[1]);
    int k = parse_int(argv[2]);
    int l = parse_int(argv[3]);
    int n = parse_int(argv[4]);
    int threads = parse_int(argv[6]);
    // validates numerical inputs
    if (m <= 0 || k <= 0 || l <= 0 || n <= 0 ||
        threads <= 0 || threads > MAX_THREADS) {
        fprintf(stderr, "Error: Invalid arguments.\n");
        return EXIT_FAILURE;
    }

    int use_double;
    // validating data type float or double
    if (strcmp(argv[5], "float") == 0) use_double = 0;
    else if (strcmp(argv[5], "double") == 0) use_double = 1;
    else {
        fprintf(stderr, "Error: datatype must be float or double.\n");
        return EXIT_FAILURE;
    }

    size_t sz = use_double ? sizeof(double) : sizeof(float);

    /* ---------- Allocate ---------- */
    // Allocates A, B, C, D_par (parallel result), D_seq(sequential result)
    // dynamic allocagtion and contiguous memory improves performance
    void *A = malloc((size_t)m * k * sz);
    void *B = malloc((size_t)k * l * sz);
    void *C = malloc((size_t)l * n * sz);
    void *D_par = malloc((size_t)m * n * sz);
    void *D_seq = malloc((size_t)m * n * sz);

    if (!A || !B || !C || !D_par || !D_seq) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    /* ---------- Initialize ---------- */
    if (!use_double) {
        float *fa=A,*fb=B,*fc=C;
        for (size_t i=0;i<(size_t)m*k;i++) fa[i]=rand_unit();
        for (size_t i=0;i<(size_t)k*l;i++) fb[i]=rand_unit();
        for (size_t i=0;i<(size_t)l*n;i++) fc[i]=rand_unit();
    } else {
        double *da=A,*db=B,*dc=C;
        for (size_t i=0;i<(size_t)m*k;i++) da[i]=rand_unit();
        for (size_t i=0;i<(size_t)k*l;i++) db[i]=rand_unit();
        for (size_t i=0;i<(size_t)l*n;i++) dc[i]=rand_unit();
    }

    /* ---------- Choose Order ---------- */
    // choose (AXB)XC or AX(BXC) based on operations
    long cost1 = (long)m*k*l + (long)m*l*n;
    long cost2 = (long)k*l*n + (long)m*k*n;

    int mode = (cost1 <= cost2) ? 0 : 1;

    size_t t_size = (mode == 0) ? (size_t)m*l : (size_t)k*n;
    void *T = malloc(t_size * sz); //T is temporary

    if (!T) {
        fprintf(stderr, "Temp allocation failed\n");
        return EXIT_FAILURE;
    }

    /* ---------- PARALLEL ---------- */
    memset(T, 0, t_size * sz);
    memset(D_par, 0, (size_t)m*n*sz);
    // end-start time => computation time
    double start_p = get_time();

    if (mode == 0) {
        parallel_mul(threads, m, k, l, A, B, T, use_double);
        parallel_mul(threads, m, l, n, T, C, D_par, use_double);
    } else {
        parallel_mul(threads, k, l, n, B, C, T, use_double);
        parallel_mul(threads, m, k, n, A, T, D_par, use_double);
    }

    double end_p = get_time();

    /* ---------- SEQUENTIAL ---------- */
    memset(T, 0, t_size * sz);
    memset(D_seq, 0, (size_t)m*n*sz);

    double start_s = get_time();

    if (mode == 0) {
        sequential_mul(m, k, l, A, B, T, use_double);
        sequential_mul(m, l, n, T, C, D_seq, use_double);
    } else {
        sequential_mul(k, l, n, B, C, T, use_double);
        sequential_mul(m, k, n, A, T, D_seq, use_double);
    }

    double end_s = get_time();

    /* ---------- Verify ---------- */
    double eps = use_double ? 1e-8 : 1e-3;
    int correct = 1;

    for (size_t i = 0; i < (size_t)m*n; i++) {
        double diff = use_double ?
            fabs(((double*)D_par)[i] - ((double*)D_seq)[i]) :
            fabs(((float*)D_par)[i] - ((float*)D_seq)[i]);

        if (diff > eps) { correct = 0; break; }
    }

    printf("Verification: %s\n", correct ? "SUCCESS" : "FAIL");
    printf("Parallel Time: %f\n", end_p - start_p);
    printf("Sequential Time: %f\n", end_s - start_s);
    // performance check: sequential / parallel
    if ((end_p - start_p) > 0)
        printf("Speedup: %f\n", (end_s - start_s) / (end_p - start_p));
    // retrieves peak memory usage
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    // RAM usage calculation
    printf("Max RAM: %.2f MB\n", usage.ru_maxrss / (1024.0 * 1024.0));
    // memory cleanup
    free(A); free(B); free(C);
    free(T); free(D_par); free(D_seq);

    return 0;
}