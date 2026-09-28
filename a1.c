/*============================================================================*
 * FILE: a1.c *
 * Skeleton code: COMP10002 Assignment 1 2026 *
 * Written by: Dr. Shaanan Cohney and Kacie Beckett *
 * Attention Is All You Need (Single-Head Attention with KV Cache) *
 * Edited by: [Yuxiang Sun, 1677061] *
 *============================================================================*/
/*==========================================================*
 * COMPLEXITY ANALYSIS *
 *==========================================================*/




/*==========================================================*
 * PREPROCESSOR DIRECTIVES *
 *==========================================================*/

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOKENS 64
#define MAX_D 64
#define MAX_GEN 32

#define NO_SOFTMAX 0
#define APPLY_SOFTMAX 1

#define MAX_TEXT_SIZE 64
#define MAX_TOKEN_LENGTH 20

/* Build the scanf format for a string reading into a fixed size char array.
 This should be used for Stage 0 when reading the prompt.*/
#define STRINGIZE(x) #x
#define STR(x) STRINGIZE(x)
#define TOKEN_STR_SCANF_FORMAT "%" STR(MAX_TOKEN_LENGTH) "s "

/*==========================================================*
 * FUNCTION PROTOTYPES *
 *==========================================================*/

/* Read all network parameters and matrices from standard input */
void read_input(int *n, int *d, int *g, int *text_len,
                char embedding_table[MAX_TEXT_SIZE][MAX_TOKEN_LENGTH + 1],
                int mask[MAX_TOKENS],
                double prompt[MAX_TOKENS][MAX_D],
                double gen[MAX_GEN][MAX_D],
                double wq[MAX_D][MAX_D], double wk[MAX_D][MAX_D],
                double wv[MAX_D][MAX_D]);

/* Extract unique tokens and sort them, and print their in one-hot coding style*/
void create_embeddings(
    char embedding_table[MAX_TEXT_SIZE][MAX_TOKEN_LENGTH + 1], int text_len);

/* Multiply the input matrix by a projection weight */
void compute_projection(int count, int d, double src[MAX_TOKENS][MAX_D],
                        double w[MAX_D][MAX_D],
                        double dest[MAX_TOKENS][MAX_D]);

/* Calculate dot products, apply mask and stable softmax */
void compute_attention_scores_or_weights_prompt(
    int n, int d, const int mask[MAX_TOKENS],
    double q[MAX_TOKENS][MAX_D], double k[MAX_TOKENS][MAX_D],
    double scores_or_weights[MAX_TOKENS][MAX_TOKENS],
    int apply_softmax);

/* Multiply the attention weights by the value matrix */
void compute_attention_output_prompt(
    int n, int d, double weights[MAX_TOKENS][MAX_TOKENS],
    double v[MAX_TOKENS][MAX_D], double out[MAX_TOKENS][MAX_D]);

/* Compute the next generated output using cached keys and values */

long compute_generation_with_cache(
    int n, int d, int g, int t, const int mask[MAX_TOKENS],
    double gen[MAX_GEN][MAX_D], double wq[MAX_D][MAX_D],
    double wk[MAX_D][MAX_D], double wv[MAX_D][MAX_D],
    double k_cache[MAX_TOKENS + MAX_GEN][MAX_D],
    double v_cache[MAX_TOKENS + MAX_GEN][MAX_D],
    double output[MAX_D]);

int compare_tokens(const void *a, const void *b);

double clamp_near_zero(double value);

void print_vector(int d, const double row[]);

void print_attention_matrix(int n,
                            const double matrix[MAX_TOKENS][MAX_TOKENS]);

void print_embedding_matrix(int n, int d,
                            const double matrix[MAX_TOKENS][MAX_D]);
/*==========================================================*
 * MAIN LOOP *
 *==========================================================*/

int main(void)
{
    int num = 0, dimension = 0, num_gen = 0;     /*ps:num is the number of tokens, dimension is dimension of embeddings on matrix, num_gen is the number of tokens to generate*/
    int text_len = 0;
    int mask[MAX_TOKENS] = {0};
    double prompt[MAX_TOKENS][MAX_D] = {{0}};
    double gen[MAX_GEN][MAX_D] = {{0}};
    double wq[MAX_D][MAX_D] = {{0}};
    double wk[MAX_D][MAX_D] = {{0}};
    double wv[MAX_D][MAX_D] = {{0}};
    char embedding_table[MAX_TEXT_SIZE][MAX_TOKEN_LENGTH + 1] = {{0}};

    read_input(&num, &dimension, &num_gen, &text_len, embedding_table, mask, prompt, gen, wq, wk, wv);

    /*Stage 1: Create Embeddings*/
    printf("Stage 1: Create Embeddings\n");
    create_embeddings(embedding_table, text_len);
    /* For stage 1 you must print the embedding table yourself.
    The later stages are done for you. */

    double q_matrix[MAX_TOKENS][MAX_D] = {{0}};
    double k_matrix[MAX_TOKENS + MAX_GEN][MAX_D] = {{0}};
    double v_matrix[MAX_TOKENS + MAX_GEN][MAX_D] = {{0}};
    /*Stage 2: Projections*/
    printf("Stage 2: Projections\n");

    compute_projection(num, dimension, prompt, wq, q_matrix);
    printf("Q Projection:\n");
    print_embedding_matrix(num, dimension, q_matrix);

    compute_projection(num, dimension, prompt, wk, k_matrix);
    printf("K Projection:\n");
    print_embedding_matrix(num, dimension, k_matrix);

    compute_projection(num, dimension, prompt, wv, v_matrix);
    printf("V Projection:\n");
    print_embedding_matrix(num, dimension, v_matrix);
    
    /*Stage 3: Attention Scores (Prompt)*/
    printf("Stage 3: Attention Scores (Prompt)\n");
    double scores[MAX_TOKENS][MAX_TOKENS] = {{0}};
    compute_attention_scores_or_weights_prompt(
        num, dimension, mask, q_matrix, k_matrix, scores, NO_SOFTMAX);
    print_attention_matrix(num, scores);

    /*Stage 4: Attention Weights (Prompt)*/
    printf("Stage 4: Attention Weights (Prompt)\n");
    double weights[MAX_TOKENS][MAX_TOKENS] = {{0}};
    compute_attention_scores_or_weights_prompt(
        num, dimension, mask, q_matrix, k_matrix, weights, APPLY_SOFTMAX);
    print_attention_matrix(num, weights);

    /*Stage 5: Attention Output (Prompt)*/
    printf("Stage 5: Attention Output (Prompt)\n");
    double out_prompt[MAX_TOKENS][MAX_D] = {{0}};
    compute_attention_output_prompt(num, dimension, weights, v_matrix, out_prompt);
    print_embedding_matrix(num, dimension, out_prompt);

    /*Stage 6: Generated Outputs*/
    printf("Stage 6: Generated Outputs\n");
    for (int t = 0; t < num_gen; t++)
    {
        double gen_output[MAX_D] = {0};
        long dot_products = compute_generation_with_cache(
            num, dimension, num_gen, t, mask, gen, wq, wk, wv, k_matrix, v_matrix,
            gen_output);
        printf("Gen %d: ", t);
        print_vector(dimension, gen_output);
        printf("Dot products computed: %ld\n", dot_products);
    }

    return 0;
}

/*==========================================================*
 * FUNCTIONS TO IMPLEMENT *
 *==========================================================*/

 /* Read all network parameters and matrices from standard input */
void read_input(int *num, int *dimension, int *num_gen, int *text_len,
                char embedding_table[MAX_TEXT_SIZE][MAX_TOKEN_LENGTH + 1],
                int mask[MAX_TOKENS],
                double prompt[MAX_TOKENS][MAX_D],
                double gen[MAX_GEN][MAX_D],
                double wq[MAX_D][MAX_D], double wk[MAX_D][MAX_D],
                double wv[MAX_D][MAX_D])
{
    scanf("%d %d %d %d", num, dimension, num_gen, text_len);      /*the same as the fist one naming, except text_len means the length of the input text*/

    for (int i = 0; i < *text_len; i++)
    {
        scanf(TOKEN_STR_SCANF_FORMAT, embedding_table[i]);
    }

    for (int i = 0; i < *num; i++)
    {
        scanf("%d", &mask[i]);
    }

    for (int i = 0; i < *num; i++)
    {
        for (int j = 0; j < *dimension; j++)
        {
            scanf("%lf", &prompt[i][j]);
        }
    }

    for (int i = 0; i < *num_gen; i++)
    {
        for (int j = 0; j < *dimension; j++)
        {
            scanf("%lf", &gen[i][j]);
        }
    }

    for (int i = 0; i < *dimension; i++)
    {
        for (int j = 0; j < *dimension; j++)
        {
            scanf("%lf", &wq[i][j]);
        }
    }

    for (int i = 0; i < *dimension; i++)
    {
        for (int j = 0; j < *dimension; j++)
        {
            scanf("%lf", &wk[i][j]);
        }
    }

    for (int i = 0; i < *dimension; i++)
    {
        for (int j = 0; j < *dimension; j++)
        {
            scanf("%lf", &wv[i][j]);
        }
    }
}

/* Extract unique tokens, sort them, and print their one-hot embeddings */
void create_embeddings(
    char embedding_table[MAX_TEXT_SIZE][MAX_TOKEN_LENGTH + 1], int text_len)
{

    char unique_tokens[MAX_TEXT_SIZE][MAX_TOKEN_LENGTH + 1];
    int unique_count = 0;

    for (int i = 0; i < text_len; i++)
    {
        int j;
        for (j = 0; j < unique_count; j++)
        {
            if (strcmp(embedding_table[i], unique_tokens[j]) == 0) /*strcmp, a little trick I learned on an article on github.(I have no ideo how to do the comparison without it so I use it and the strcpy latter)*/
            {
                break;
            }
        }
        if (j == unique_count)
        {
            strcpy(unique_tokens[unique_count], embedding_table[i]);
            unique_count++;
        }
    }

    qsort(unique_tokens, unique_count, sizeof(unique_tokens[0]), compare_tokens);
    
    for (int i = 0; i < unique_count; i++)
    {
        printf("\"%s\" -> (" , unique_tokens[i]);
        for (int j = 0; j < unique_count; j++)
        {
            if (i == j)
            {
                if (j == unique_count - 1)
                {
                    printf("1)\n");
                }
                else
                {
                    printf("1 ");
                }
            }
            else
            {
                if (j == unique_count - 1)
                {
                    printf("0)\n");
                }
                else
                {
                    printf("0 ");
                }
            }
        }
    }
}

/* Multiply the input matrix by a projection weight matrix */
void compute_projection(int count, int d, double src[MAX_TOKENS][MAX_D],
                        double w[MAX_D][MAX_D],
                        double dest[MAX_TOKENS][MAX_D])
{

    for (int i = 0; i < count; i++)
    {
        for (int j = 0; j < d; j++)
        {
            double sum = 0.0;
            for (int u = 0; u < d; u++)
            {
                double left_value = src[i][u];
                double right_value = w[u][j];
                double multiplied_matrix = left_value * right_value;
                sum = sum + multiplied_matrix;
            }
            dest[i][j] = sum;
        }
    }
}

/* Calculate dot products, apply masks, and optionally apply stable softmax */
void compute_attention_scores_or_weights_prompt(int n, int d, const int mask[MAX_TOKENS], double q[MAX_TOKENS][MAX_D], double k[MAX_TOKENS][MAX_D], double scores_or_weights[MAX_TOKENS][MAX_TOKENS], int apply_softmax)
{
    double sqrt_d = sqrt(d);
    for (int i = 0; i < n; i++)
    {
        double row_max = -INFINITY;
        for (int j = 0; j < n; j++)
        {
            if (j > i || mask[j] == 0 || mask[i] == 0)
            {
                scores_or_weights[i][j] = -INFINITY;
            }
            else
            {
                double dot_product = 0.0;
                for (int u = 0; u < d; u++)
                {
                    double query_val = q[i][u];
                    double key_val = k[j][u];
                    double multiplied_val = query_val * key_val;
                    dot_product = dot_product + multiplied_val;
                }
                scores_or_weights[i][j] = dot_product / sqrt_d;
            }

            if (apply_softmax && scores_or_weights[i][j] > row_max)
            {
                row_max = scores_or_weights[i][j];
            }
        }
        if (apply_softmax)
        {
            double sum_exp = 0.0;
            for (int j = 0; j < n; j++)
            {
                if (scores_or_weights[i][j] == -INFINITY)
                {
                    scores_or_weights[i][j] = 0.0; 
                }
                else
                {
                    scores_or_weights[i][j] = exp(scores_or_weights[i][j] - row_max);
                    sum_exp += scores_or_weights[i][j];
                }
            }
            for (int j = 0; j < n; j++)
            {
                if (sum_exp > 0.0)
                {
                    scores_or_weights[i][j] = scores_or_weights[i][j] / sum_exp;
                }
                else
                {
                    scores_or_weights[i][j] = 0.0;
                }
            }
        }
    }
}

/* Multiply the attention weights by the value matrix to get the final output */
void compute_attention_output_prompt(
    int n, int d, double weights[MAX_TOKENS][MAX_TOKENS],
    double v[MAX_TOKENS][MAX_D], double out[MAX_TOKENS][MAX_D])
{

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < d; j++)
        {
            double sum = 0.0;
            for (int u = 0; u < n; u++)
            {
                sum += weights[i][u] * v[u][j];
            }
            out[i][j] = sum;
        }
    }
}

/* Compute the next generated output using cached keys and values */
long compute_generation_with_cache(
    int n, int d, int g, int t, const int mask[MAX_TOKENS],
    double gen[MAX_GEN][MAX_D], double wq[MAX_D][MAX_D],
    double wk[MAX_D][MAX_D], double wv[MAX_D][MAX_D],
    double k_cache[MAX_TOKENS + MAX_GEN][MAX_D],
    double v_cache[MAX_TOKENS + MAX_GEN][MAX_D],
    double output[MAX_D])
{

    long dot_products_count = 0;
    int current_pos = n + t;
    double q_t[MAX_D] = {0};
    for (int j = 0; j < d; j++)
    {
        for (int u = 0; u < d; u++)
        {
            q_t[j] += gen[t][u] * wq[u][j];
            
        }
        dot_products_count++;
    }
    for (int j = 0; j < d; j++)
    {
        for (int u = 0; u < d; u++)
        {
            k_cache[current_pos][j] += gen[t][u] * wk[u][j];
            v_cache[current_pos][j] += gen[t][u] * wv[u][j];
           
        } 
        dot_products_count += 2;
    }
    double scores[MAX_TOKENS + MAX_GEN] = {0};
    double row_max = -INFINITY;
    double sqrt_d = sqrt(d);

    for (int j = 0; j <= current_pos; j++)
    {
        // Causal mask is intrinsically handled because we only loop up to current_pos
        // Padding mask for prompt tokens
        if (j < n && mask[j] == 0)
        {
            scores[j] = -INFINITY;
        }
        else
        {
            double dot_product = 0.0;
            dot_products_count++;
            for (int u = 0; u < d; u++)
            {
                dot_product += q_t[u] * k_cache[j][u];
                
            }
            scores[j] = dot_product / sqrt_d;
        }

        if (scores[j] > row_max)
        {
            row_max = scores[j];
        }
    }

    // 4. Stable Softmax
    double sum_exp = 0.0;
    for (int j = 0; j <= current_pos; j++)
    {
        if (scores[j] != -INFINITY)
        {
            scores[j] = exp(scores[j] - row_max);
            sum_exp += scores[j];
        }
        else
        {
            scores[j] = 0.0;
        }
    }

    for (int j = 0; j <= current_pos; j++)
    {
        if (sum_exp > 0.0)
        {
            scores[j] /= sum_exp;
        }
        else
        {
            scores[j] = 0.0;
        }
    }
    for (int j = 0; j < d; j++)
    {
        output[j] = 0.0;
        dot_products_count++;
        for (int u = 0; u <= current_pos; u++)
        {
            output[j] += scores[u] * v_cache[u][j];
            
        }
    }
    return dot_products_count;
}

/*==========================================================*
 * HELPER FUNCTIONS *
 *==========================================================*/

/* Optional helper for qsort based token sorting */
int compare_tokens(const void *a, const void *b)
{
    return strcmp((const char *)a, (const char *)b);
}

/* Set small doubles to 0.0 to remove floating point noise */
double clamp_near_zero(double value)
{
    if (fabs(value) < 0.0005) /*calculate the noise needs it to be an absolute number, we can use the if statement to do it but fabs can make it simpler*/
    {
        return 0.0;
    }
    return value;
}

/* print an n by n attention matrix with each row on a new line */
void print_attention_matrix(int n, const double matrix[MAX_TOKENS][MAX_TOKENS])
{
    for (int i = 0; i < n; i++)
    {
        print_vector(n, matrix[i]);
    }
}

/* print an n by d embedding matrix with each row on a new line */
void print_embedding_matrix(int n, int d,
                            const double matrix[MAX_TOKENS][MAX_D])
{
    for (int i = 0; i < n; i++)
    {
        print_vector(d, matrix[i]);
    }
}

/* print a vector on a new row with each value seperated by a space, and small
 values clamped to 0.0*/
void print_vector(int d, const double row[])
{
    for (int i = 0; i < d; i++)
    {
        double value = clamp_near_zero(row[i]);
        if (i == d - 1)
        {
            printf("%.3f", value);
        }
        else
        {
            printf("%.3f ", value);
        }
    }
    printf("\n");
}