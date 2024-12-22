#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define IMAGE_SIZE 784  // 28x28 pixels
#define WEIGHT_SIZE (IMAGE_SIZE + 1)  // Including bias
#define TRAIN_RATIO 0.8
#define LEARNING_RATE 1e-6
#define STEP_SIZE 1e-3
#define EPOCHS 20
#define MAX_LINE_LENGTH 4096
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8
#define NUM_PREDICTIONS 20

typedef struct {
    int label;               // -1 or 1 for class label
    double pixels[IMAGE_SIZE]; // Pixel values (normalized to 0-1)
} ImageData;

// Read CSV data and store it in ImageData array
int read_csv(const char* filename, ImageData** data, int* num_samples) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening CSV file");
        return -1;
    }

    char line[MAX_LINE_LENGTH];
    int count = 0;

    if (fgets(line, MAX_LINE_LENGTH, file) == NULL) { // Skip header
        perror("Error reading header");
        fclose(file);
        return -1;
    }

    int capacity = 100;
    *data = malloc(capacity * sizeof(ImageData));
    if (!*data) {
        perror("Error allocating memory");
        fclose(file);
        return -1;
    }

    while (fgets(line, MAX_LINE_LENGTH, file)) {
        line[strcspn(line, "\n")] = 0; // Trim newline

        if (strlen(line) == 0) continue;

        if (count >= capacity) {
            capacity *= 2;
            *data = realloc(*data, capacity * sizeof(ImageData));
            if (!*data) {
                perror("Error reallocating memory");
                fclose(file);
                return -1;
            }
        }

        char* token = strtok(line, ",");
        if (!token) continue;

        (*data)[count].label = atoi(token) == 1 ? 1 : -1;  // T-shirt: -1, Pants: 1
        int pixel_count = 0;
        token = strtok(NULL, ",");
        while (token && pixel_count < IMAGE_SIZE) {
            (*data)[count].pixels[pixel_count] = atof(token) / 255.0;
            pixel_count++;
            token = strtok(NULL, ",");
        }

        if (pixel_count == IMAGE_SIZE) {
            count++;
        } else {
            fprintf(stderr, "Skipping line %d: incorrect pixel count\n", count + 2);
        }
    }

    fclose(file);
    *num_samples = count;
    return 0;
}

// Shuffle indices for random training/testing split
void shuffle_indices(int* indices, int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

// Compute prediction for a given sample
double predict(double* x, double* w) {
    double dot_product = w[0];  // Bias term
    for (int i = 0; i < IMAGE_SIZE; i++) {
        dot_product += x[i] * w[i + 1];
    }
    return tanh(dot_product);  // Apply tanh to the dot product
}

// Compute loss for labels in range [-1, 1]
double compute_loss(double* y, double* y_pred, int m) {
    double loss = 0;
    for (int i = 0; i < m; i++) {
        double error = y_pred[i] - y[i];  // Difference between predicted and actual labels
        loss += error * error;  // Sum of squared errors
    }
    loss /= m;  // Compute the mean squared error
    return loss;  // Return the root mean square error
}

void initialize_weights(double* w, int size) {
    double range = sqrt(6.0 / (IMAGE_SIZE + 1));  // Xavier initialization
    for (int i = 0; i < size; i++) {
        w[i] = ((double)rand() / RAND_MAX) * 2 * range - range;  // Random values within [-range, range]
    }
}

// Function to evaluate accuracy
double evaluate_accuracy(ImageData* data, double* w, int num_samples) {
    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        double pred = predict(data[i].pixels, w);
        int predicted_label = (pred > 0) ? 1 : -1;  // Classify using threshold 0
        if (predicted_label == data[i].label) {
            correct++;
        }
    }
    return (double)correct / num_samples;
}

// Evaluate and print predictions
void print_predictions(ImageData* data, double* w, int num_samples) {
    printf("\nSample Predictions:\n");
    int count_tshirts = 0, count_pants = 0;
    for (int i = 0; i < num_samples && (count_tshirts < NUM_PREDICTIONS / 2 || count_pants < NUM_PREDICTIONS / 2); i++) {
        double pred = predict(data[i].pixels, w);
        int true_label = data[i].label;
        if (true_label == -1 && count_tshirts < NUM_PREDICTIONS / 2) {
            printf("T-shirt Sample %d: True Label = %d, Predicted Value = %.3f\n", count_tshirts + 1, true_label, pred);
            count_tshirts++;
        } else if (true_label == 1 && count_pants < NUM_PREDICTIONS / 2) {
            printf("Pants Sample %d: True Label = %d, Predicted Value = %.3f\n", count_pants + 1, true_label, pred);
            count_pants++;
        }
    }
}
int is_file_empty(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        return 1;  // File doesn't exist or can't be opened, treat as empty
    }
    fseek(file, 0, SEEK_END);  // Move to the end of the file
    long size = ftell(file);
    fclose(file);
    return size == 0;  // File is empty if size is 0
}
void epoch_loss(const char* filename, int epoch, double loss) {
    FILE* file = fopen(filename, "a"); // Append mode
    if (!file) {
        perror("Error opening log file");
        return;
    }

    // Check if file is empty
    if (is_file_empty(filename)) {
        fprintf(file, "Epoch,Loss\n");  // Write header only if the file is empty
    }

    // Write the epoch and loss data
    fprintf(file, "%d,%.6f\n", epoch, loss);
    fclose(file);
}
void time_loss(const char* filename, double loss, double time) {
    FILE* file = fopen(filename, "a"); // Append mode
    if (!file) {
        perror("Error opening log file");
        return;
    }

    // Check if file is empty
    if (is_file_empty(filename)) {
        fprintf(file, "Time,Loss\n");  // Write header only if the file is empty
    }

    // Write the epoch and time data
    fprintf(file, "%.6f,%.6f\n", time, loss);
    fclose(file);
}
void gradient_descent(ImageData* data, double* w, double* y, double* y_pred, int m, int epoch, ImageData* test_data, int test_count, const char* time_log, const char* loss_log, double* total_time) {
    double dw[WEIGHT_SIZE] = {0};

    clock_t start_time = clock();  // Start time for the epoch

    // Compute predictions and gradients
    for (int i = 0; i < m; i++) {
        y_pred[i] = predict(data[i].pixels, w);  // Get prediction
        double error = y_pred[i] - y[i];        // Compute the error

        dw[0] += error;  // Gradient for bias
        for (int j = 0; j < IMAGE_SIZE; j++) {
            dw[j + 1] += error * data[i].pixels[j];  // Gradient for weights
        }
    }

    // Update weights
    for (int i = 0; i < WEIGHT_SIZE; i++) {
        w[i] -= STEP_SIZE * dw[i] / m;  // Apply learning rate
    }

    clock_t end_time = clock();  // End time for the epoch
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    *total_time += elapsed_time;  // Accumulate total time

    // Compute loss
    double loss = sqrt(compute_loss(y, y_pred, m));

    // Log data
    time_loss(time_log, loss, *total_time);
    epoch_loss(loss_log, epoch, loss);

    // Print test accuracy
    double test_accuracy = evaluate_accuracy(test_data, w, test_count);
    printf("GD - Epoch: %d, Loss: %.6f, Test Accuracy: %.3f, Time: %.6f\n", epoch, loss, test_accuracy, elapsed_time);
}
void stochastic_gradient_descent(ImageData* data, double* w, double* y, double* y_pred, int m, int epoch, ImageData* test_data, int test_count, const char* time_log, const char* loss_log, double* total_time) {
    clock_t start_time = clock();  // Start time for the epoch

    int indices[m];
    for (int i = 0; i < m; i++) {
        indices[i] = i;
    }
    shuffle_indices(indices, m);

    // Update weights using each sample
    for (int k = 0; k < m; k++) {
        int idx = indices[k];
        double* x = data[idx].pixels;
        double true_label = y[idx];

        double pred = predict(x, w);
        double error = pred - true_label;

        w[0] -= LEARNING_RATE * error;  // Update bias
        for (int j = 0; j < IMAGE_SIZE; j++) {
            w[j + 1] -= LEARNING_RATE * error * x[j];  // Update weights
        }
    }

    clock_t end_time = clock();  // End time for the epoch
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    *total_time += elapsed_time;  // Accumulate total time

    // Compute loss
    for (int i = 0; i < m; i++) {
        y_pred[i] = predict(data[i].pixels, w);
    }
    double loss = compute_loss(y, y_pred, m);

    // Log data
    time_loss(time_log, loss, *total_time);
    epoch_loss(loss_log, epoch, loss);

    // Print test accuracy
    double test_accuracy = evaluate_accuracy(test_data, w, test_count);
    printf("SGD - Epoch: %d, Loss: %.6f, Test Accuracy: %.3f, Time: %.6f\n", epoch, loss, test_accuracy, elapsed_time);
}
void adam_optimizer(ImageData* data, double* w, double* y, double* y_pred, int m, int epoch, ImageData* test_data, int test_count, const char* time_log, const char* loss_log, double* total_time) {
    static double m_t[WEIGHT_SIZE] = {0};
    static double v_t[WEIGHT_SIZE] = {0};
    static double t = 0;

    double dw[WEIGHT_SIZE] = {0};
    clock_t start_time = clock();  // Start time for the epoch

    // Compute gradients
    for (int i = 0; i < m; i++) {
        y_pred[i] = predict(data[i].pixels, w);
        double error = y_pred[i] - y[i];

        dw[0] += error;
        for (int j = 0; j < IMAGE_SIZE; j++) {
            dw[j + 1] += error * data[i].pixels[j];
        }
    }

    // Update moments and weights
    t++;
    for (int i = 0; i < WEIGHT_SIZE; i++) {
        m_t[i] = BETA1 * m_t[i] + (1 - BETA1) * dw[i];
        v_t[i] = BETA2 * v_t[i] + (1 - BETA2) * dw[i] * dw[i];

        double m_t_hat = m_t[i] / (1 - pow(BETA1, t));
        double v_t_hat = v_t[i] / (1 - pow(BETA2, t));

        w[i] -= STEP_SIZE * m_t_hat / (sqrt(v_t_hat) + EPSILON);
    }

    clock_t end_time = clock();  // End time for the epoch
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    *total_time += elapsed_time;  // Accumulate total time

    // Compute loss
    double loss = sqrt(compute_loss(y, y_pred, m));

    // Log data
    time_loss(time_log, loss, *total_time);
    epoch_loss(loss_log, epoch, loss);

    // Print test accuracy
    double test_accuracy = evaluate_accuracy(test_data, w, test_count);
    printf("Adam - Epoch: %d, Loss: %.6f, Test Accuracy: %.3f, Time: %.6f\n", epoch, loss, test_accuracy, elapsed_time);
}
int main() {
    const char* filename = "C:\\Users\\ENES\\Desktop\\ödev\\Denemeler\\DiffProje\\fashion_mnist.csv"; // CSV file path
    ImageData* data = NULL;
    int num_samples = 0;

    // Read CSV file
    if (read_csv(filename, &data, &num_samples) != 0) {
        fprintf(stderr, "Failed to read CSV file\n");
        return 1;
    }

    printf("Read %d samples from %s\n", num_samples, filename);

    // Split into training and testing sets
    int total_samples = num_samples;
    int train_count = total_samples * TRAIN_RATIO;
    int test_count = total_samples - train_count;
    int indices[total_samples];

    for (int i = 0; i < total_samples; i++) {
        indices[i] = i;
    }

    srand(time(NULL));
    shuffle_indices(indices, total_samples);

    // Separate training and test data using shuffled indices
    ImageData* train_data = malloc(train_count * sizeof(ImageData));
    ImageData* test_data = malloc(test_count * sizeof(ImageData));

    for (int i = 0; i < train_count; i++) {
        train_data[i] = data[indices[i]]; // Train data based on shuffled indices
    }
    for (int i = 0; i < test_count; i++) {
        test_data[i] = data[indices[train_count + i]]; // Test data based on shuffled indices
    }

    double y[train_count];
    double y_pred[train_count];

    for (int i = 0; i < train_count; i++) {
        y[i] = train_data[i].label;
    }
    double total_time;
    // Loop through 5 different initial weights
    for (int run = 0; run < 5; run++) {
        printf("\nRun %d/5\n", run + 1);

        // Initialize weights
        double w[WEIGHT_SIZE] = {0};
        double w_stored[WEIGHT_SIZE] = {0};
        initialize_weights(w_stored, WEIGHT_SIZE);
        
        // Gradient Descent Logs
        char gd_time_log[50], gd_loss_log[50];
        sprintf(gd_time_log, "gd_time_log_run%d.csv", run + 1);
        sprintf(gd_loss_log, "gd_loss_log_run%d.csv", run + 1);

        memcpy(w, w_stored, sizeof(w_stored));  // Reset weights before starting GD
        printf("\nTraining with Gradient Descent...\n");
        total_time = 0;
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            gradient_descent(train_data, w, y, y_pred, train_count, epoch, test_data, test_count, gd_time_log, gd_loss_log,&total_time);
        }
        double gd_accuracy = evaluate_accuracy(test_data, w, test_count);
        printf("GD Final Test Accuracy: %.3f\n", gd_accuracy);

        // Stochastic Gradient Descent Logs
        char sgd_time_log[50], sgd_loss_log[50];
        sprintf(sgd_time_log, "sgd_time_log_run%d.csv", run + 1);
        sprintf(sgd_loss_log, "sgd_loss_log_run%d.csv", run + 1);

        memcpy(w, w_stored, sizeof(w_stored));  // Reset weights before starting SGD
        printf("\nTraining with Stochastic Gradient Descent...\n");
        total_time = 0;
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            stochastic_gradient_descent(train_data, w, y, y_pred, train_count, epoch, test_data, test_count, sgd_time_log, sgd_loss_log,&total_time);
        }
        double sgd_accuracy = evaluate_accuracy(test_data, w, test_count);
        printf("SGD Final Test Accuracy: %.3f\n", sgd_accuracy);

        // Adam Optimizer Logs
        char adam_time_log[50], adam_loss_log[50];
        sprintf(adam_time_log, "adam_time_log_run%d.csv", run + 1);
        sprintf(adam_loss_log, "adam_loss_log_run%d.csv", run + 1);

        memcpy(w, w_stored, sizeof(w_stored));  // Reset weights before starting ADAM
        printf("\nTraining with Adam Optimizer...\n");
        total_time = 0;
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            adam_optimizer(train_data, w, y, y_pred, train_count, epoch, test_data, test_count, adam_time_log, adam_loss_log,&total_time);
        }
        double adam_accuracy = evaluate_accuracy(test_data, w, test_count);
        printf("Adam Final Test Accuracy: %.3f\n", adam_accuracy);
    }

    // Free memory
    free(data);
    free(train_data);
    free(test_data);

    return 0;
}