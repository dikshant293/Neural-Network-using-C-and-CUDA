#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <arpa/inet.h> // Include this for little-endian conversion
#include <cblas.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>

#define IMAGE_MAGIC_NUMBER 2051
#define LABEL_MAGIC_NUMBER 2049
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define NUM_IMAGES 60000
#define NUM_VALIDATION_IMAGES 10000
#define NUM_TRAIN_IMAGES (NUM_IMAGES-NUM_VALIDATION_IMAGES)
#define NUM_TEST_IMAGES 10000
#define NUM_DIGITS 10

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define ABS(x) (((x) < (0)) ? (-x) : (x))

#define BLOCK_SIZE 16

int no_BLAS = 0;
int use_GPU = 0;
// Allocate memory for matrices A, B, and C on device
float *d_A, *d_B, *d_C;

// Kernel for matrix-matrix multiplication
__global__ void multiply_kernel(float *A, int transA, float *B, int transB, float *C, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    if (i < M && j < N) {
        float sum = 0.0;
        for (int k = 0; k < K; ++k) {
            a = transA ? A[k * M + i] : A[i * K + k];
            b = transB ? B[j * K + k] : B[k * N + j];
            sum += a * b;
        }
        C[i * N + j] = sum;
    }
}


void gpu_multiply_matrices(float* h_A, int transA, float* h_B, int transB, float* h_C, int m, int n ,int k){
    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    if(no_BLAS){
        // Launch kernel for mative GPU matrix multiplication
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

        multiply_kernel<<<numBlocks, threadsPerBlock>>>(d_A, transA, d_B, transB, d_C, m, n, k);
    }
    else{
        // use CuBLASS for matrix multiplication
        cublasSgemm(
                    (transB ? 'T' : 'N'),
                    (transA ? 'T' : 'N'),
                    n,m,k,
                    1.0f,
                    d_B, (transB ? k : n),
                    d_A, (transA ? m : k),
                    0.0f,
                    d_C, n
                );
    }
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
}
// CPU native matrix multiplication
void multiply_matrices(float* mat1, int transA, float* mat2, int transB, float* result, int M, int N ,int K) {
    float a,b;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            result[i * N + j] = 0.0f;
            for (int k = 0; k < K; ++k) {
                a = transA ? mat1[k * M + i] : mat1[i * K + k];
                b = transB ? mat2[j * K + k] : mat2[k * N + j];
                result[i * N + j] += a * b;
            }
        }
    }
}
// daxpy for matrix
void daxpy(float alpha, float *x, float *y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}
// copy matrix
void copy(float alpha, float *x, float *y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = alpha * x[i];
    }
}
// apply sigmoid to a value
float sigmoid(float x){
    return 1.0f/(1.0f+exp(-1.0f*x));
}
// apply sigmoid to a matrix
void sigmoid_matrix(float *ans, float *mat, int rows, int cols){
    for(int j=0;j<cols;j++){
        for(int i=0;i<rows;i++){
            ans[j+cols*i]=sigmoid(mat[j+cols*i]);
        }
    }
}
// apply relu to a matrix
void relu_matrix(float *ans, float *mat, int rows, int cols){
    for(int j=0;j<cols;j++){
        for(int i=0;i<rows;i++){
            ans[j+cols*i]=MAX(0.0f,mat[j+cols*i]);
        }
    }
}
// apply softmax to a matrix
void softmax_matrix(float *ans, float *mat, int rows, int cols){
    float sum = 0.0f,max = 0.0f;
    for(int j=0;j<cols;j++){
        sum = 0.0f;
        max = 0.0f;
        for(int i=0;i<rows;i++){
            max=MAX(max,mat[j+cols*i]);
        }
        for(int i=0;i<rows;i++){
            sum+=exp(mat[j+cols*i]-max);
        }
        for(int i=0;i<rows;i++){
            ans[j+cols*i]=exp(mat[j+cols*i]-max)/sum;
        }
    }
}
// print image to terminal
void print_image(int idx, float *images, float* labels){
    printf("Label of idx = %d image:\n", idx);
    for(int i=0;i<NUM_DIGITS;i++)
        printf("%0.1lf ",labels[NUM_DIGITS*idx + i]);
    printf("\n");
    for(int i=0;i<IMAGE_HEIGHT;i++){
        for(int j=0;j<IMAGE_WIDTH;j++){
            float val = images[idx*IMAGE_HEIGHT*IMAGE_WIDTH + i*IMAGE_HEIGHT+j];
            if(val) printf("%0.2lf ",val);
            else printf(".... ");
        }
        printf("\n");
    }
}
// to shuffle the training images at the start of each epoch to randomize batch selection
void shuffle(int *array, size_t n){
    if(n > 1){
        size_t i;
        for (i = 0; i < n - 1; i++){
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}
// forward propagation
void fwd_prop(float **layers, float **z_vals, float* labels, int total_layers,int batch_size, int *layer_dims, float **weights, float **biases, int total_nodes, float **batch_biases){
    int m,n,k;

    for(int i=1;i<total_layers;i++){
        batch_biases[i] = batch_biases[i-1] + layer_dims[i-1]*batch_size;
        for(int j=0;j<layer_dims[i];j++){
            for(k=0;k<batch_size;k++){
                batch_biases[i][j*batch_size+k] = biases[i][j];
            }
        }
    }
    for(int l=1;l<total_layers;l++){
        m=layer_dims[l];
        n=batch_size;
        k=layer_dims[l-1];
        // if using GPU
        if(use_GPU) gpu_multiply_matrices(weights[l],0,layers[l-1],0,z_vals[l],m,n,k);
        else{ // if using CPU
            if(no_BLAS){ // native CPU
                multiply_matrices(weights[l],0,layers[l-1],0,z_vals[l],m,n,k);
            }
            else{ // CPU BLAS
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, weights[l], k, layers[l-1], n, 0.0, z_vals[l], n);
            }
        }
        if(no_BLAS) daxpy(1.0f,batch_biases[l],z_vals[l],m*n);
        else        cblas_saxpy(m*n, 1.0, batch_biases[l], 1, z_vals[l], 1);

        if(l==total_layers-1){ // final layer
            softmax_matrix(layers[l],z_vals[l],m,n);
        }
        else{ // not final layer
            relu_matrix(layers[l],z_vals[l],m,n);
        }
    }
}

// uses the weights and bises to infer the given images and check accuracy against the provided labels
float accuracy(float **layers, float *images, int total_layers, int *layer_dims, float **weights, float **biases, float* labels, int nsamples){
    int correct = 0;
    for(int b=0;b<nsamples;b++){
        for(int i=0;i<layer_dims[0];i++){
            layers[0][i] = images[b*IMAGE_HEIGHT*IMAGE_WIDTH + i];
        }
        int m,n,k;
        for(int l=1;l<total_layers;l++){
            float *temp = (float*)malloc(layer_dims[l]*sizeof(float));
            m=layer_dims[l];
            n=1;
            k=layer_dims[l-1];
            // similar to forward prop, choose appropriate function as per run type
            if(use_GPU) gpu_multiply_matrices(weights[l],0,layers[l-1],0,temp,m,n,k);
            else{
                if(no_BLAS){
                    multiply_matrices(weights[l],0,layers[l-1],0,temp,m,n,k);
                }
                else{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, weights[l], k, layers[l-1], n, 0.0, temp, n);
                }
            }
            if(no_BLAS) daxpy(1.0f,biases[l],temp,m);
            else        cblas_saxpy(m, 1.0, biases[l], 1, temp, 1);

            
            if(l==total_layers-1){
                softmax_matrix(layers[l],temp,m,n);
            }
            else{
                relu_matrix(layers[l],temp,m,n);
            }
            free(temp);
        }
        
        int y_predict = 0;
        float val = layers[total_layers-1][0];
        for(int i=1;i<NUM_DIGITS;i++){
            if(layers[total_layers-1][i]>val){
                val = layers[total_layers-1][i];
                y_predict = i;
            }
        }
        correct += (labels[b*NUM_DIGITS + y_predict]==1.0f);
    }
    float acc = (float)((float)correct / (float)nsamples);
    return acc;
}
// box muller - random number generator - norm with given mean and variance
float rand_normal(float mean, float variance) {
    static int generate = 0;
    static float z1;
    float u1, u2, z0;
    generate = !generate;
    if (!generate)
        return z1 * variance + mean;
    do {
        u1 = (float)rand() / RAND_MAX;
        u2 = (float)rand() / RAND_MAX;
        z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
        z1 = sqrt(-2.0 * log(u1)) * sin(2 * M_PI * u2);
    } while (z0 * z0 + z1 * z1 >= 1 || z0 == 0);
    return z0 * variance + mean;
}

int main(int argc, char** argv){
    int nl,nh,ne,nb,print_acc_per_epoch=1,print_val_loss=1,seed = 0;
    float alpha;
    nl = atoi(argv[1]);
    nh = atoi(argv[2]);
    ne = atol(argv[3]);
    nb = atoi(argv[4]);
    alpha = atof(argv[5]);
    if(argc>6){
        no_BLAS = atoi(argv[6]);
    }
    if(argc>7){
        use_GPU = atoi(argv[7]);
    }
    if(argc>8){
        print_acc_per_epoch = atoi(argv[8]);
    }
    if(argc>9){
        print_val_loss = atoi(argv[9]);
    }
    if(argc>10){
        seed = atoi(argv[10]);
    }
    srand(seed); // if seed is inputted from CLI and not equal to 0, then per epoch cost printing will be suppressed 
    printf("nl = %d nh = %d ne = %d nb = %d alpha = %lf\n",nl,nh,ne,nb,alpha);
    printf(no_BLAS ? "NO BLAS\n" : "USING BLAS\n");
    printf(use_GPU ? "USING GPU\n" : "USING CPU\n");
    printf(print_acc_per_epoch ? "printing accuracy scores each epoch\n" : "NOT printing accuracy scores each epoch\n");
    printf(print_val_loss ? "printing validation loss each epoch\n" : "NOT printing validation loss each epoch\n");
    int stat = 0;
    printf("%d\n",stat);

    // input files and output file for writing validation loss
    FILE *image_file = fopen("train-images-idx3-ubyte", "rb");
    FILE *label_file = fopen("train-labels-idx1-ubyte", "rb");
    FILE *validation_file = fopen("validation_loss.dat","w");
    
    fprintf(validation_file,"%d %d %d %d %lf %d\n",nl,nh,ne,nb,alpha,no_BLAS*2+use_GPU);

    // Read image file header
    int magic_number, num_images, num_rows, num_cols;
    stat = fread(&magic_number, sizeof(magic_number), 1, image_file);
    stat = fread(&num_images, sizeof(num_images), 1, image_file);
    stat = fread(&num_rows, sizeof(num_rows), 1, image_file);
    stat = fread(&num_cols, sizeof(num_cols), 1, image_file);

    // Convert from big-endian to little-endian if needed
    magic_number = ntohl(magic_number);
    num_images = ntohl(num_images);
    num_rows = ntohl(num_rows);
    num_cols = ntohl(num_cols);

    // Check if the file format is correct
    if (magic_number != IMAGE_MAGIC_NUMBER || num_images != NUM_IMAGES || num_rows != IMAGE_HEIGHT || num_cols != IMAGE_WIDTH) {
        printf("Invalid image file.\n");
        return 1;
    }

    // Read label file header
    stat = fread(&magic_number, sizeof(magic_number), 1, label_file);
    stat = fread(&num_images, sizeof(num_images), 1, label_file);

    // Convert from big-endian to little-endian if needed
    magic_number = ntohl(magic_number);
    num_images = ntohl(num_images);

    // Check if the file format is correct
    if (magic_number != LABEL_MAGIC_NUMBER || num_images != NUM_IMAGES) {
        printf("Invalid label file.\n");
        return 1;
    }
    
    // Read images and labels
    uint8_t x,y;
    float *images = (float*)malloc(NUM_IMAGES * IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float));
    float *labels = (float*)malloc(NUM_DIGITS * NUM_IMAGES * sizeof(float));

    for (int i = 0; i < NUM_IMAGES; i++) {
        for(int j=0;j<IMAGE_HEIGHT;j++){
            for(int k=0;k<IMAGE_WIDTH;k++){
                stat = fread(&x, sizeof(x), 1, image_file);
                images[i*IMAGE_HEIGHT*IMAGE_WIDTH+j*IMAGE_HEIGHT+k] = ntohs(x)>>8;
                images[i*IMAGE_HEIGHT*IMAGE_WIDTH+j*IMAGE_HEIGHT+k] /= 255.0f;
            }
        }
        stat = fread(&y, sizeof(y), 1, label_file);
        y = ntohs(y)>>8;
        for(int j=0;j<NUM_DIGITS;j++)
            labels[i*NUM_DIGITS+j] = (float)(j==y);
    }

    
    float *validation_images = &images[NUM_TRAIN_IMAGES * IMAGE_WIDTH * IMAGE_HEIGHT];
    float *validation_labels = &labels[NUM_TRAIN_IMAGES * NUM_DIGITS];
    
    fclose(image_file);
    fclose(label_file);

    image_file = fopen("t10k-images-idx3-ubyte", "rb");
    label_file = fopen("t10k-labels-idx1-ubyte", "rb");
    
    // Read image file header
    stat = fread(&magic_number, sizeof(magic_number), 1, image_file);
    stat = fread(&num_images, sizeof(num_images), 1, image_file);
    stat = fread(&num_rows, sizeof(num_rows), 1, image_file);
    stat = fread(&num_cols, sizeof(num_cols), 1, image_file);

    // Convert from big-endian to little-endian if needed
    magic_number = ntohl(magic_number);
    num_images = ntohl(num_images);
    num_rows = ntohl(num_rows);
    num_cols = ntohl(num_cols);

    // printf("magic_number = %d num_images = %d num_rows = %d num_cols = %d\n",magic_number,num_images,num_rows,num_cols);
    
    // Check if the file format is correct
    if (magic_number != IMAGE_MAGIC_NUMBER || num_images != NUM_TEST_IMAGES || num_rows != IMAGE_HEIGHT || num_cols != IMAGE_WIDTH) {
        printf("Invalid image file.\n");
        return 1;
    }

    // Read label file header
    stat = fread(&magic_number, sizeof(magic_number), 1, label_file);
    stat = fread(&num_images, sizeof(num_images), 1, label_file);

    // Convert from big-endian to little-endian if needed
    magic_number = ntohl(magic_number);
    num_images = ntohl(num_images);

    // Check if the file format is correct
    if (magic_number != LABEL_MAGIC_NUMBER || num_images != NUM_TEST_IMAGES) {
        printf("Invalid label file.\n");
        return 1;
    }
    
    // Read images and labels
    float *test_images = (float*)malloc(NUM_TEST_IMAGES * IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float));
    float *test_labels = (float*)malloc(NUM_DIGITS * NUM_TEST_IMAGES * sizeof(float));

    for (int i = 0; i < NUM_TEST_IMAGES; i++) {
        for(int j=0;j<IMAGE_HEIGHT;j++){
            for(int k=0;k<IMAGE_WIDTH;k++){
                stat = fread(&x, sizeof(x), 1, image_file);
                test_images[i*IMAGE_HEIGHT*IMAGE_WIDTH+j*IMAGE_HEIGHT+k] = ntohs(x)>>8;
                test_images[i*IMAGE_HEIGHT*IMAGE_WIDTH+j*IMAGE_HEIGHT+k] /= 255.0f;
            }
        }
        stat = fread(&y, sizeof(y), 1, label_file);
        y = ntohs(y)>>8;
        for(int j=0;j<NUM_DIGITS;j++)
            test_labels[i*NUM_DIGITS+j] = (float)(j==y);
    }
    
    fclose(image_file);
    fclose(label_file);


    // layers-> with activation values
    // z_vals-> before activation
    float **layers,**z_vals; 
    int total_layers = 2+nl, total_nodes = IMAGE_HEIGHT*IMAGE_HEIGHT + nl*nh + NUM_DIGITS;
    
    // layer dims = 784(input) 10 10 10(output)
    int *layer_dims = (int*)malloc(total_layers*sizeof(int));

    layers = (float**)malloc(total_layers*sizeof(float*));
    layers[0] = (float*)malloc(total_nodes*sizeof(float));
    z_vals = (float**)malloc(total_layers*sizeof(float*));
    z_vals[0] = (float*)malloc(total_nodes*sizeof(float));
    layer_dims[0] = IMAGE_HEIGHT*IMAGE_HEIGHT;

    for(int i=1;i<total_layers;i++){
        layer_dims[i] = i==total_layers-1 ? NUM_DIGITS : nh;
    }
    for(int i=1;i<total_layers;i++){
        layers[i] = layers[i-1] + layer_dims[i-1];
        z_vals[i] = z_vals[i-1] + layer_dims[i-1];
    }
    int num_weights = 0, num_biases = 0;
    for(int i=1;i<total_layers;i++){
        num_weights+=layer_dims[i]*layer_dims[i-1];
        num_biases+=layer_dims[i];
    }
    float **weights, **biases;
    weights = (float**)malloc((total_layers)*sizeof(float*));
    weights[0] = (float*)malloc(num_weights*sizeof(float));
    biases = (float**)malloc((total_layers)*sizeof(float*));
    biases[0] = (float*)malloc(total_nodes*sizeof(float));

    for(int i=0;i<total_layers;i++)
        printf("%d ",layer_dims[i]);
    printf("\n");

    // initialize weights and biases
    for(int i=1;i<total_layers;i++){
        weights[i] = i==1 ? weights[0] : weights[i-1] + layer_dims[i-1]*layer_dims[i-2];
        biases[i] = biases[i-1] + layer_dims[i-1];
        for(int j=0;j<layer_dims[i]*layer_dims[i-1];j++){
            weights[i][j] = rand_normal(0,2.0f/(float)layer_dims[i-1]);
        }
        for(int j=0;j<layer_dims[i];j++){
            biases[i][j] = 0.0f;
        }
    }

    int maxdim = layer_dims[0];
    for(int i=0;i<total_layers;i++) maxdim = MAX(maxdim,layer_dims[i]);

    // malloc all required arrays on CPU and GPU beforehand
    if(use_GPU){
        cudaMalloc((void **)&d_A, maxdim*maxdim*nb * sizeof(float));
        cudaMalloc((void **)&d_B, maxdim*maxdim*nb * sizeof(float));
        cudaMalloc((void **)&d_C, maxdim*maxdim*nb * sizeof(float));
    }
    float **batch_layers, **batch_z, *batch_labels, **batch_error;
    batch_layers = (float**)malloc(total_layers*sizeof(float*));
    batch_layers[0] = (float*)malloc(total_nodes*nb*sizeof(float));
    batch_error = (float**)malloc(total_layers*sizeof(float*));
    batch_error[0] = (float*)malloc(total_nodes*nb*sizeof(float));
    batch_z = (float**)malloc(total_layers*sizeof(float*));
    batch_z[0] = (float*)malloc(total_nodes*nb*sizeof(float));
    batch_labels = (float*)malloc(layer_dims[total_layers-1]*nb*sizeof(float));
    float *batch_diff = (float*)malloc(layer_dims[total_layers-1]*nb*sizeof(float));
    float **weight_error;
    weight_error = (float**)malloc((total_layers)*sizeof(float*));
    weight_error[0] = (float*)malloc(num_weights*sizeof(float));

    float *temp_arr = (float*)malloc(maxdim*nb*sizeof(float));
    
    float **batch_biases;
    batch_biases = (float**)malloc((total_layers)*sizeof(float*));
    batch_biases[0] = (float*)malloc(total_nodes*nb*sizeof(float));

    // for randomizing the batches for each epoch
    int *index_arr;
    index_arr = (int*)malloc(NUM_TRAIN_IMAGES*sizeof(int));
    for(int i=0;i<NUM_TRAIN_IMAGES;i++) index_arr[i]=i;

    printf("~~~~~~~~~~~~~~~~~~~~~~~~LEARNING~~~~~~~~~~~~~~~~~~~~~~~~\n");
    float grind_avg = 0.0f;
    double total_t1, total_t2;
    total_t1 = omp_get_wtime();
    for(int e=1;e<=ne;e++){ //epoch
        double epoch_time = omp_get_wtime();
        if(seed==0) printf("\n---------------epoch %d---------------\n",e);
        shuffle(index_arr,NUM_TRAIN_IMAGES);
        for(int start = 0;start<NUM_TRAIN_IMAGES;start+=nb){ //batch
            int end = MIN(start+nb,NUM_TRAIN_IMAGES);
            float C=0.0f;
            double batch_t;
            int batch_size = end-start;
            float multiplier = alpha/batch_size;
            
            // initialize batch matrices for training
            for(int i=1;i<total_layers;i++){
                batch_layers[i] = batch_layers[i-1] + layer_dims[i-1]*batch_size;
                batch_z[i] = batch_z[i-1] + layer_dims[i-1]*batch_size;
                batch_error[i] = batch_error[i-1] + layer_dims[i-1]*batch_size;
            }
            for(int idx=start;idx<end;idx++){
                int b = index_arr[idx];
                for(int i=0;i<layer_dims[0];i++){
                    batch_layers[0][(idx-start) + i*batch_size] = images[b*layer_dims[0] + i];
                }
                for(int i=0;i<layer_dims[total_layers-1];i++){
                    batch_labels[(idx-start) + i*batch_size] = labels[b*layer_dims[total_layers-1] + i];
                }
            }
            // initialize error matrices
            for(int i=1;i<total_layers;i++){
                weight_error[i] = i==1 ? weight_error[0] : weight_error[i-1] + layer_dims[i-1]*layer_dims[i-2];
                for(int j=0;j<layer_dims[i]*layer_dims[i-1];j++){
                    weight_error[i][j] = 0.0f;
                }
            }
            // only print grind rate for first batch of the epoch, its almost same for all so no need to print all batch grind rates
            if(start==0)
                batch_t = omp_get_wtime();
            
            fwd_prop(batch_layers, batch_z, batch_labels, total_layers, batch_size, layer_dims, weights, biases, total_nodes, batch_biases);
            
            // calculate cross entropy loss
            if(no_BLAS){
                copy(1.0f,batch_layers[total_layers-1],batch_diff,layer_dims[total_layers-1]*batch_size);
                daxpy(-1.0,batch_labels,batch_diff,layer_dims[total_layers-1]*batch_size);
            }else{
                cblas_scopy(layer_dims[total_layers-1]*batch_size, batch_layers[total_layers-1], 1, batch_diff, 1);
                cblas_saxpy(layer_dims[total_layers-1]*batch_size, -1.0, batch_labels, 1, batch_diff, 1); // batch_diff = a_L - y
            }

            for(int i=0;i<batch_size;i++){
                for(int j=0;j<layer_dims[total_layers-1];j++){
                    float x = batch_labels[i+j*batch_size]==0.0f ? 0.0f : -batch_labels[i+j*batch_size]*log(batch_layers[total_layers-1][i+j*batch_size])/batch_size;
                    C+=x;
                }
            }
            
            // back propagate
            int m,n,k;
            for(int l = total_layers-1;l>=1;l--){
                if(l==total_layers-1){
                    for(int i=0;i<layer_dims[total_layers-1]*batch_size;i++){
                        batch_error[l][i] = batch_diff[i];
                    }
                }
                else{
                    m=layer_dims[l];
                    n=batch_size;
                    k=layer_dims[l+1];

                    if(use_GPU) gpu_multiply_matrices(weights[l+1],1,batch_error[l+1],0,temp_arr,m,n,k);
                    else{
                        if(no_BLAS){
                            multiply_matrices(weights[l+1],1,batch_error[l+1],0,temp_arr,m,n,k);
                        }
                        else{
                            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0, weights[l+1], m, batch_error[l+1], n, 0.0, temp_arr, n);
                        }
                    }

                    for(int i=0;i<layer_dims[l]*batch_size;i++){
                        batch_error[l][i] = temp_arr[i] * (batch_layers[l][i]>0.0f);
                    }
                }
            }

            // claculate weight error
            for(int l=1;l<total_layers;l++){
                m=layer_dims[l];
                n=layer_dims[l-1];
                k=batch_size;

                if(use_GPU) gpu_multiply_matrices(batch_error[l],0,batch_layers[l-1],1,weight_error[l],m,n,k);
                else{
                    if(no_BLAS){
                        multiply_matrices(batch_error[l],0,batch_layers[l-1],1,weight_error[l],m,n,k);
                    }
                    else{
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0, batch_error[l], k, batch_layers[l-1], k, 0.0, weight_error[l], n);
                    }
                }
            }
            // accumulate errors of the batch into the 0th sample
            for(int l=1;l<total_layers;l++){
                for(int i=0;i<layer_dims[l];i++){
                    for(int j=1;j<batch_size;j++){
                        batch_error[l][i*batch_size]+=batch_error[l][i*batch_size+j];
                    }
                }
            }
            // batch processing now complete, back propagate and update weights and biases
            for(int l=1;l<total_layers;l++){
                for(int i=0;i<layer_dims[l]*layer_dims[l-1];i++){
                    weights[l][i] -= multiplier * weight_error[l][i]; 
                }
                for(int i=0;i<layer_dims[l];i++){
                    biases[l][i] -= multiplier * batch_error[l][i*batch_size];
                }
            }
            // only printing when seed==0, When seed!=0, means we are testing effects of randomization so cost values are not needed
            if(start==0 && seed==0){
                printf("fwd prop time %lf seconds, grind rate = %0.2lf images/sec\n",omp_get_wtime()-batch_t, (double)batch_size / (omp_get_wtime()-batch_t));
                printf("Train cost = %lf\n",C);
                grind_avg += (double)batch_size / (omp_get_wtime()-batch_t);
            }
            
            
        }
        if(seed == 0)
            printf("epoch time %lf seconds\n",omp_get_wtime()-epoch_time);
        
        // print validation loss at the end of epoch
        if(print_val_loss){
            float C=0.0f;
            for(int start = 0;start<NUM_VALIDATION_IMAGES;start+=nb){ //batch
                int end = MIN(start+nb,NUM_VALIDATION_IMAGES);
                int batch_size = end-start;
                
                for(int i=0;i<total_nodes*batch_size;i++){
                    batch_layers[0][i] = 0.0f;
                    batch_error[0][i] = 0.0f;
                    batch_z[0][i] = 0.0f;
                }
                for(int i=0;i<layer_dims[total_layers-1]*batch_size;i++){
                    batch_diff[i] = 0.0f;
                }

                for(int i=1;i<total_layers;i++){
                    batch_layers[i] = batch_layers[i-1] + layer_dims[i-1]*batch_size;
                    batch_z[i] = batch_z[i-1] + layer_dims[i-1]*batch_size;
                    batch_error[i] = batch_error[i-1] + layer_dims[i-1]*batch_size;
                }
                for(int idx=start;idx<end;idx++){
                    int b = idx;
                    for(int i=0;i<layer_dims[0];i++){
                        batch_layers[0][(idx-start) + i*batch_size] = validation_images[b*layer_dims[0] + i];
                    }
                    for(int i=0;i<layer_dims[total_layers-1];i++){
                        batch_labels[(idx-start) + i*batch_size] = validation_labels[b*layer_dims[total_layers-1] + i];
                    }
                }
                
                fwd_prop(batch_layers, batch_z, batch_labels, total_layers, batch_size, layer_dims, weights, biases, total_nodes, batch_biases);
                
                
                if(no_BLAS){
                    copy(1.0f,batch_layers[total_layers-1],batch_diff,layer_dims[total_layers-1]*batch_size);
                    daxpy(-1.0,batch_labels,batch_diff,layer_dims[total_layers-1]*batch_size);
                }else{
                    cblas_scopy(layer_dims[total_layers-1]*batch_size, batch_layers[total_layers-1], 1, batch_diff, 1);
                    cblas_saxpy(layer_dims[total_layers-1]*batch_size, -1.0, batch_labels, 1, batch_diff, 1); // batch_diff = a_L - y
                }
                
                for(int i=0;i<batch_size;i++){
                    for(int j=0;j<layer_dims[total_layers-1];j++){
                        float x = batch_labels[i+j*batch_size]==0.0f ? 0.0f : -batch_labels[i+j*batch_size]*log(batch_layers[total_layers-1][i+j*batch_size]);
                        C+=x/(float)NUM_VALIDATION_IMAGES;
                    }
                }
            }
            printf("Validation cost = %lf\n",C);
            fprintf(validation_file,"%lf\n",C);
        }
        // print test and train acc
        if(print_acc_per_epoch){
            printf("train acc = %0.2lf %\n",100.0f*accuracy(layers,images,total_layers,layer_dims,weights,biases,labels,NUM_IMAGES));
            printf("test acc = %0.2lf %\n",100.0f*accuracy(layers,test_images,total_layers,layer_dims,weights,biases,test_labels,NUM_TEST_IMAGES));
        }
    }

    total_t2 = omp_get_wtime();
    printf("\nTotal time to train %0.2lf seconds\n",total_t2-total_t1);
    
    // print final result with configuraion information and accuracies
    printf("Avg Grind rate  = %lf images/sec\n",grind_avg/(float)ne);
    printf(no_BLAS ? "NO BLAS\n" : "USING BLAS\n");
    printf(use_GPU ? "USING GPU\n" : "USING CPU\n");
    fclose(validation_file);
    printf("seed = %d\n",seed);
    printf("FINAL ACCURACY\n");
    printf("train acc = %0.2lf %\n",100.0f*accuracy(layers,images,total_layers,layer_dims,weights,biases,labels,NUM_IMAGES));
    printf("test acc = %0.2lf %\n",100.0f*accuracy(layers,test_images,total_layers,layer_dims,weights,biases,test_labels,NUM_TEST_IMAGES));
    printf("\n\n\n");
    
    // free all allocated memory on CPU and GPU
    if(use_GPU){
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    free(batch_biases[0]);
    free(batch_biases);

    free(temp_arr);
    free(batch_diff);
    free(batch_labels);
    free(batch_layers[0]);
    free(batch_layers);
    free(batch_error[0]);
    free(batch_error);
    free(batch_z[0]);
    free(batch_z);
    free(weight_error[0]);
    free(weight_error);

    free(layers[0]);
    free(layers);
    free(weights[0]);
    free(weights);
    free(biases[0]);
    free(biases);
    free(z_vals[0]);
    free(z_vals);
    free(images);
    free(labels);
    free(test_images);
    free(test_labels);
    free(layer_dims);
    free(index_arr);
    return 0;
}