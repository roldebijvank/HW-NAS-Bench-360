// Generated with AI.


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/wait.h>

// Set to 0,1 seconds (0 second and 100.000 microseconds)
#define INTERVAL_SEC  0
#define INTERVAL_USEC 100000

// Global static variables access by the alarm handler. 
static char* arguments[15];
static char* program;

// Signalhandler for SIGALRM
void alarm_handler(int signum) {
    pid_t pid = fork();

    if (pid < 0) {
        perror("Fork failed");
        exit(1);
    }

    if (pid == 0) {
        // Child process: execute external program
        execv(program, arguments);

        // When execvp fails
        perror("Execvp failed");
        exit(1);
    } else {
        // Parent process: clear child process (non-blocking)
        // setitimer restarts itself
        waitpid(-1, NULL, WNOHANG);
    }
}

int main(int argc, char* argv[]) {
    struct itimerval timer;

    printf("Periodic application started. Interval: %d sec and %d usec.\n", INTERVAL_SEC, INTERVAL_USEC);

    // Copy arguments of this program to global variables.
    program = argv[1];
    int i;
    for (i=1; i<argc && i<15; i++) {
      arguments[i-1]=argv[i];
    }
    arguments[i-1]=NULL;

    // Connect handler to SIGALRM signal
    signal(SIGALRM, alarm_handler);

    // Define the periodic interval
    timer.it_interval.tv_sec = INTERVAL_SEC;
    timer.it_interval.tv_usec = INTERVAL_USEC;

    // Define the initial delay time
    timer.it_value.tv_sec = INTERVAL_SEC;
    timer.it_value.tv_usec = INTERVAL_USEC;

    // Start the realtime timer
    if (setitimer(ITIMER_REAL, &timer, NULL) < 0) {
        perror("Setitimer configuration fialed");
        exit(1);
    }

    // Endless loop to repeat the call.
    while (1) {
        pause(); // Wait on signal
    }

    return 0;
}

