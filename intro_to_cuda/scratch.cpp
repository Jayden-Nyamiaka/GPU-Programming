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

/*
void scratch_pad() {
    unsigned int blur_v_size = 5;
    
    for (unsigned int thread_index = 2; thread_index < 6; thread_index++) {
        unsigned int max = blur_v_size;
        if (blur_v_size > thread_index) {
            max = thread_index + 1;
        }
        printf("Idx: %u \t Max: %u\n", thread_index, max);
        for (int j = 0; j < max; j++) {
            printf("Access: %u\n", thread_index - j);
        }
    }
    
}
*/

int main() {
  printf("Scratch Work:\n");
  test1();
  test2();
  test3();
  test4();
  test5();
  //scratch_pad();
  return 0;
}