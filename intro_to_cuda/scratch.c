#include <stdio.h>
#include <stdlib.h>

void test1() {
    int three = 3; // Fixed here
    int *a = &three; // to here
    *a = *a + 2;
    printf("%d\n", *a);
}

void test2() {
    int *a, *b; // Fixed here
    a = (int *) malloc(sizeof (int));
    b = (int *) malloc(sizeof (int));

    if (!(a && b)) {
        printf("Out of memory\n");
        exit(-1);
    }
    *a = 2;
    *b = 3;
}

void test3() {
    int i, *a = (int *) malloc(1000 * sizeof(int)); // Fixed here

    if (!a) {
        printf("Out of memory\n");
        exit(-1);
    }
    for (i = 0; i < 1000; i++)
        *(i + a) = i;
}

void test4() {
    int **a = (int **) malloc(3 * sizeof (int *));
    for (int i = 0; i < 3; i++) { // Fix added here
      a[i] = (int *) malloc(100 * sizeof(int)); // and here
    }
    a[1][1] = 5;
}

void test5() {
    int *a = (int *) malloc(sizeof (int));
    scanf("%d", a);
    if (!*a) // Fixed here
        printf("Value is 0\n");
}

void pad() {
    uint blur_v_size = 4;
    
    for (uint thread_index = 0; thread_index < 10; thread_index++) {
        uint max = blur_v_size;
        if (blur_v_size > thread_index) {
            max = thread_index + 1;
        }
        printf("Max: %u\n", max);
        for (int j = 0; j < max; j++) {
            printf("Idx: %u \t Access: %u\n", thread_index, thread_index - j);
            gpu_out_data[thread_index] += gpu_raw_data[thread_index - j] * blur_v[j];
        }
    }
    
}

int main() {
  printf("Scratch Work:\n");
  test1();
  test2();
  test3();
  test4();
  test5();
  pad();
  return 0;
}