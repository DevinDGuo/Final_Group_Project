#ifndef MY_BARRIER_H
#define MY_BARRIER_H

#include <pthread.h>
#include <stdint.h>

#define MY_BARRIER_SERIAL_THREAD 1

typedef struct my_barrier {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    volatile uint32_t flag;
    size_t count;
    size_t num;
} my_barrier_t;

int my_barrier_init(my_barrier_t *bar, int attr, int num);
int my_barrier_wait(my_barrier_t *bar);

#endif // MY_BARRIER_H
