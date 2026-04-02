// tools/server/cuda-checkpoint.h
// Fork/exec wrapper for NVIDIA cuda-checkpoint CLI binary.
// Used by llama-server to self-freeze and by watchdog to thaw.
#pragma once

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Run cuda-checkpoint with given action on pid.
// action: "lock" or "unlock"
// Returns 0 on success, -1 on fork failure, child exit code otherwise.
static inline int run_cuda_checkpoint(const char * action, pid_t pid) {
    char pid_str[32];
    snprintf(pid_str, sizeof(pid_str), "%d", pid);

    pid_t child = fork();
    if (child < 0) {
        perror("fork");
        return -1;
    }
    if (child == 0) {
        // Child: exec cuda-checkpoint
        execlp("cuda-checkpoint", "cuda-checkpoint",
               "--action", action, "--pid", pid_str, (char *)NULL);
        perror("execlp cuda-checkpoint");
        _exit(127);
    }
    // Parent: wait for child
    int status = 0;
    waitpid(child, &status, 0);
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return -1;
}

// Check if cuda-checkpoint binary exists in PATH.
static inline bool cuda_checkpoint_available() {
    return system("command -v cuda-checkpoint > /dev/null 2>&1") == 0;
}
