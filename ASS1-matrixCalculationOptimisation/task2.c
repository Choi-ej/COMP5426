#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <stdint.h>
#include <sys/resource.h>
#ifdef __linux__
#include <sched.h>   // for thread affinity
#endif

#define MAX_THREADS 64

/* ---------- Thread Data ---------- */
typedef struct {
    int start_row, end_row; // row partitioning. Each thread processess only these rows -> avoid race conditions
    int rows, common, cols; // rows, common(shared dimmension in inner loop), colubmns
    // generic pointers to support float or double
    void *A;  // Left matrix
    void *B;  // Right matrix
    void *C;  // Result matrix
    int use_double;
} ThreadData;

/* ---------- Time ---------- */
double get_time() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

/* ---------- Safe Parsing ---------- */
int parse_int(const char *s) {
    char *end;
    long val = strtol(s, &end, 10);
    if (*end != '\0' || val <= 0) return -1;
    return (int)val;
}

/* ---------- Random ---------- */
double rand_unit() {
    return (double)arc4random() / (double)UINT32_MAX;
}

/* ---------- Worker ---------- */
/*
   Performs matrix multiplication:
   C = A × B

   A: rows × common
   B: common × cols
   C: rows × cols
*/
// original worker part for sequential
void* worker(void *arg) {
    ThreadData *d = (ThreadData*)arg;

#ifdef __linux__
    /* Optional: pin thread to CPU core */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(d->start_row % sysconf(_SC_NPROCESSORS_ONLN), &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif

    if (!d->use_double) {
        float *A = (float*)d->A;
        float *B = (float*)d->B;
        float *C = (float*)d->C;

        for (int i = d->start_row; i < d->end_row; i++) {
            for (int k = 0; k < d->common; k++) {
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
// this part is unroll part for parallel version. loop unrolling factor = 4
void* worker_unroll(void *arg) {
    ThreadData *d = (ThreadData*)arg;
    int common = d->common;
    int c = d->cols;

    // precomputed limits -> loops run in multiples of 4. And leftover elements handled seperately
    // because loop unrolling requires processing 4 elements at a time and avoid out of bounds access
    int k_limit = (common / 4) * 4;
    int j_limit = (c / 4) * 4;

    if (!d->use_double) {
        float *A = (float*)d->A;
        float *B = (float*)d->B;
        float *C = (float*)d->C;

        for (int i = d->start_row; i < d->end_row; i++) {
            // Precomputing row offsets reduces redundant index calculations and improves execution efficiency.
            int row_i_C = i * c;
            int row_i_A = i * common;
            // this is unrolling k (first level). k++ to k+=4. Each iteration processes 4 values of k at once
            for (int k = 0; k < k_limit; k += 4) {
                // 4 consecutive elements from matrix A, enable simultaneous computation, reducing memory access overhead
                float a0 = A[row_i_A + k];
                float a1 = A[row_i_A + k + 1];
                float a2 = A[row_i_A + k + 2];
                float a3 = A[row_i_A + k + 3];
                // points to one row of B
                float *b0 = &B[k * c];
                float *b1 = &B[(k + 1) * c];
                float *b2 = &B[(k + 2) * c];
                float *b3 = &B[(k + 3) * c];

                int j = 0;
                // unrolling j-loop (second level)
                for (; j < j_limit; j += 4) {
                    int idx = row_i_C + j;
                    // computes 4 * 4 = 16 multiplications per iteration at once.
                    // reduces loop overhead and increases instruction-level parallelism. 
                    C[idx]     += a0*b0[j]   + a1*b1[j]   + a2*b2[j]   + a3*b3[j];
                    C[idx + 1] += a0*b0[j+1] + a1*b1[j+1] + a2*b2[j+1] + a3*b3[j+1];
                    C[idx + 2] += a0*b0[j+2] + a1*b1[j+2] + a2*b2[j+2] + a3*b3[j+2];
                    C[idx + 3] += a0*b0[j+3] + a1*b1[j+3] + a2*b2[j+3] + a3*b3[j+3];
                }

                // j remainder (handles c % 4 != 0)
                for (; j < c; j++) {
                    C[row_i_C + j] += a0*b0[j] + a1*b1[j] + a2*b2[j] + a3*b3[j];
                }
            }

            // k remainder (handles common % 4 != 0)
            for (int k = k_limit; k < common; k++) {
                float a = A[row_i_A + k];
                float *bk = &B[k * c];
                for (int j = 0; j < c; j++) {
                    C[row_i_C + j] += a * bk[j];
                }
            }
        }

    } else {
        double *A = (double*)d->A;
        double *B = (double*)d->B;
        double *C = (double*)d->C;

        for (int i = d->start_row; i < d->end_row; i++) {
            int row_i_C = i * c;
            int row_i_A = i * common;

            for (int k = 0; k < k_limit; k += 4) {
                double a0 = A[row_i_A + k];
                double a1 = A[row_i_A + k + 1];
                double a2 = A[row_i_A + k + 2];
                double a3 = A[row_i_A + k + 3];

                double *b0 = &B[k * c];
                double *b1 = &B[(k + 1) * c];
                double *b2 = &B[(k + 2) * c];
                double *b3 = &B[(k + 3) * c];

                int j = 0;
                for (; j < j_limit; j += 4) {
                    int idx = row_i_C + j;

                    C[idx]     += a0*b0[j]   + a1*b1[j]   + a2*b2[j]   + a3*b3[j];
                    C[idx + 1] += a0*b0[j+1] + a1*b1[j+1] + a2*b2[j+1] + a3*b3[j+1];
                    C[idx + 2] += a0*b0[j+2] + a1*b1[j+2] + a2*b2[j+2] + a3*b3[j+2];
                    C[idx + 3] += a0*b0[j+3] + a1*b1[j+3] + a2*b2[j+3] + a3*b3[j+3];
                }

                for (; j < c; j++) {
                    C[row_i_C + j] += a0*b0[j] + a1*b1[j] + a2*b2[j] + a3*b3[j];
                }
            }

            for (int k = k_limit; k < common; k++) {
                double a = A[row_i_A + k];
                double *bk = &B[k * c];
                for (int j = 0; j < c; j++) {
                    C[row_i_C + j] += a * bk[j];
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
        td[i].start_row = i * rows / actual;
        td[i].end_row   = (i + 1) * rows / actual;

        td[i].rows = rows;
        td[i].common = common;
        td[i].cols = cols;

        td[i].A = A;
        td[i].B = B;
        td[i].C = C;
        td[i].use_double = use_double;
        // unroll version for only parallel computation (k-loop outer, j-loop inner)
        pthread_create(&tids[i], NULL, worker_unroll, &td[i]);
    }

    for (int i = 0; i < actual; i++) {
        pthread_join(tids[i], NULL);
    }
}


/* ---------- Sequential ---------- */
void sequential_mul(int rows, int common, int cols,
                    void *A, void *B, void *C, int use_double) {

    ThreadData td = {0, rows, rows, common, cols, A, B, C, use_double};
    worker(&td);
}

/* ---------- MAIN ---------- */
int main(int argc, char *argv[]) {

    if (argc != 7) {
        fprintf(stderr, "Usage: %s m k l n <float|double> threads\n", argv[0]);
        return EXIT_FAILURE;
    }

    int m = parse_int(argv[1]);
    int k = parse_int(argv[2]);
    int l = parse_int(argv[3]);
    int n = parse_int(argv[4]);
    int threads = parse_int(argv[6]);

    if (m <= 0 || k <= 0 || l <= 0 || n <= 0 ||
        threads <= 0 || threads > MAX_THREADS) {
        fprintf(stderr, "Error: Invalid arguments.\n");
        return EXIT_FAILURE;
    }

    int use_double;
    if (strcmp(argv[5], "float") == 0) use_double = 0;
    else if (strcmp(argv[5], "double") == 0) use_double = 1;
    else {
        fprintf(stderr, "Error: datatype must be float or double.\n");
        return EXIT_FAILURE;
    }

    size_t sz = use_double ? sizeof(double) : sizeof(float);

    /* ---------- Allocate ---------- */
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
    long cost1 = (long)m*k*l + (long)m*l*n;
    long cost2 = (long)k*l*n + (long)m*k*n;

    int mode = (cost1 <= cost2) ? 0 : 1;

    size_t t_size = (mode == 0) ? (size_t)m*l : (size_t)k*n;
    void *T = malloc(t_size * sz);

    if (!T) {
        fprintf(stderr, "Temp allocation failed\n");
        return EXIT_FAILURE;
    }

    /* ---------- PARALLEL ---------- */
    memset(T, 0, t_size * sz);
    memset(D_par, 0, (size_t)m*n*sz);

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

        double base = use_double ?
        fabs(((double*)D_seq)[i]) :
        fabs(((float*)D_seq)[i]);

        if (diff / (base + 1e-6) > eps){ correct = 0; break; }
        
    }

    printf("Verification: %s\n", correct ? "SUCCESS" : "FAIL");
    printf("Parallel Time: %f\n", end_p - start_p);
    printf("Sequential Time: %f\n", end_s - start_s);

    if ((end_p - start_p) > 0)
        printf("Speedup: %f\n", (end_s - start_s) / (end_p - start_p));

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    printf("Max RAM: %.2f MB\n", usage.ru_maxrss / (1024.0 * 1024.0));

    free(A); free(B); free(C);
    free(T); free(D_par); free(D_seq);

    return 0;
}