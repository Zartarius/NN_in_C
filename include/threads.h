#ifndef THREADS_H
#define THREADS_H

#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#include <stdint.h>

typedef HANDLE thread_t;
typedef DWORD thread_func_return_t; // Ensure this matches LPTHREAD_START_ROUTINE

typedef LPVOID thread_func_param_t;

#define THREAD_CREATE(thread, func, arg) \
do { \
    thread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(func), arg, 0, NULL); \
    if ((thread) == NULL) { \
        printf("Failed to create thread. Error: %d\n", GetLastError()); \
        exit(1); \
    } \
} while (0)

#define THREAD_JOIN_AND_CLOSE(thread, num_threads) \
do { \
    WaitForMultipleObjects(num_threads, threads, TRUE, INFINITE); \
    for (int i = 0; i < (num_threads); i++) { \
        CloseHandle(threads[i]); \
    } \
} while(0)

#define THREAD_JOIN(thread, result)  \
do { \
    DWORD waitStatus = WaitForSingleObject(thread, INFINITE); \
    if (waitStatus == WAIT_OBJECT_0) { \
        DWORD exitCode; \
        GetExitCodeThread(thread, &exitCode); \
        result = (void *)(uintptr_t)exitCode; \
    } else { \
        result = NULL; \
    } \
} while (0)

#define THREAD_EXIT ExitThread(0)

#define THREAD_CLOSE(thread) CloseHandle(thread)

#else
#include <errno.h>
#include <pthread.h>

typedef pthread_t thread_t;
typedef void *thread_func_return_t;
typedef void *thread_func_param_t;

#define THREAD_CREATE(thread, func, arg) \
do { \
    if ( pthread_create(&thread, NULL, func, arg) != 0) { \
        perror("pthread_create"); \
        exit(1); \
    } \
} while (0)

#define THREAD_JOIN_AND_CLOSE(thread, num_threads) \
do { \
    for (size_t i = 0; i < (num_threads); i++) { \
        pthread_join(threads[i], NULL); \
    } \
} while(0)

#define THREAD_JOIN(thread, result)  \
do { \
    void *ret_val; \
    if (pthread_join(thread, &ret_val) == 0) { \
        result = ret_val; \
    } else { \
        result = NULL; \
    } \
} while (0)

#define THREAD_EXIT pthread_exit(NULL)

#define THREAD_CLOSE(thread)
#endif

#endif // THREADS_H
