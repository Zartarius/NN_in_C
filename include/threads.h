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
    thread = CreateThread(NULL, 256 * 1024, (LPTHREAD_START_ROUTINE)(func), arg, 0, NULL); \
    if ((thread) == NULL) { \
        char message[256]; \
FormatMessage( \
    FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, \
    NULL, \
    GetLastError(), \
    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), \
    message, \
    sizeof(message), \
    NULL \
); \
printf("Error: %s\n", message); \
        printf("Failed to create thread. Error: %lu\n", GetLastError()); \
        exit(1); \
    } \
} while (0)

#define THREAD_JOIN_AND_CLOSE(thread, num_threads) \
do { \
    for(size_t i = 0; i < num_threads; i++) { \
    DWORD waitResult = WaitForSingleObject(threads[i], INFINITE); \
    if (waitResult == WAIT_FAILED) { \
        fprintf(stderr, "Failed to wait for thread %zu: %lu\n", i, GetLastError()); \
        exit(1); \
    } \
    } \
    for (size_t i = 0; i < (num_threads); i++) { \
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
