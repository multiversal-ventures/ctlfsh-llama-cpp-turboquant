// llama-freeze-watchdog: proxy that thaws a frozen llama-server on incoming connections.
// Usage: llama-freeze-watchdog [--listen ADDR:PORT] [--backend ADDR:PORT] -- <llama-server args...>
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static pid_t                  g_child_pid  = -1;
static volatile sig_atomic_t  g_child_frozen = 0;
static volatile sig_atomic_t  g_shutdown     = 0;

static const char * g_listen_addr  = "0.0.0.0";
static int          g_listen_port  = 8080;
static const char * g_backend_addr = "127.0.0.1";
static int          g_backend_port = 2853;

// --- signal handlers ---

static void on_sigusr1(int sig) { (void)sig; g_child_frozen = 1; }
static void on_shutdown(int sig) { (void)sig; g_shutdown = 1; }
static void on_sigchld(int sig) {
    (void)sig;
    int status;
    // Reap all finished children. Only shutdown if llama-server child dies.
    // cuda-checkpoint and proxy fork children also trigger SIGCHLD.
    pid_t pid;
    while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
        if (pid == g_child_pid) {
            fprintf(stderr, "[watchdog] server child %d exited (status %d)\n",
                    g_child_pid, WIFEXITED(status) ? WEXITSTATUS(status) : -1);
            g_shutdown = 1;
        }
        // proxy children and cuda-checkpoint children are silently reaped
    }
}

// --- cuda-checkpoint CLI wrapper ---

static int cuda_ckpt(const char * action, pid_t pid) {
    char pid_str[32];
    snprintf(pid_str, sizeof(pid_str), "%d", pid);
    pid_t child = fork();
    if (child < 0) return -1;
    if (child == 0) {
        execlp("cuda-checkpoint", "cuda-checkpoint",
               "--action", action, "--pid", pid_str, (char *)NULL);
        _exit(127);
    }
    int status;
    waitpid(child, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// --- health check ---

static int wait_for_health(int timeout_ms) {
    int elapsed = 0;
    while (elapsed < timeout_ms) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(g_backend_port);
        inet_pton(AF_INET, g_backend_addr, &addr.sin_addr);

        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            const char * req = "GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n";
            if (write(fd, req, strlen(req)) > 0) {
                char buf[512];
                int n = read(fd, buf, sizeof(buf) - 1);
                close(fd);
                if (n > 0) {
                    buf[n] = '\0';
                    if (strstr(buf, "\"ok\"")) return 0;
                }
            } else {
                close(fd);
            }
        } else {
            close(fd);
        }
        usleep(50000);
        elapsed += 50;
    }
    return -1;
}

// --- proxy ---

static long proxy_fds(int client_fd, int backend_fd) {
    long total = 0;
    struct pollfd fds[2];
    fds[0].fd = client_fd;  fds[0].events = POLLIN;
    fds[1].fd = backend_fd; fds[1].events = POLLIN;
    char buf[8192];
    int client_done = 0;

    while (1) {
        int ret = poll(fds, 2, 120000);
        if (ret <= 0) break;

        // Client → backend (request body)
        if (fds[0].revents & POLLIN) {
            int n = read(client_fd, buf, sizeof(buf));
            if (n <= 0) {
                // Client done sending — half-close to backend
                shutdown(backend_fd, SHUT_WR);
                client_done = 1;
                fds[0].fd = -1; // stop polling client
            } else {
                write(backend_fd, buf, n);
                total += n;
            }
        }

        // Backend → client (response)
        if (fds[1].revents & POLLIN) {
            int n = read(backend_fd, buf, sizeof(buf));
            if (n <= 0) break; // backend done = response complete
            write(client_fd, buf, n);
            total += n;
        }

        // Client hung up — stop reading client but keep reading backend
        if (!client_done && (fds[0].revents & (POLLHUP | POLLERR))) {
            shutdown(backend_fd, SHUT_WR);
            client_done = 1;
            fds[0].fd = -1;
        }

        // Backend hung up or errored — we're done
        if (fds[1].revents & (POLLHUP | POLLERR)) {
            // Drain any remaining data
            while (1) {
                int n = read(backend_fd, buf, sizeof(buf));
                if (n <= 0) break;
                write(client_fd, buf, n);
                total += n;
            }
            break;
        }
    }
    return total;
}

// --- helpers ---

static long ms_since(struct timespec * t0) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0->tv_sec) * 1000 + (t1.tv_nsec - t0->tv_nsec) / 1000000;
}

static void usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s [options] -- <llama-server command...>\n\n"
        "  --listen ADDR:PORT   Listen address (default 0.0.0.0:8080)\n"
        "  --backend ADDR:PORT  Backend address (default 127.0.0.1:2853)\n"
        "  --                   Separator; everything after is the child command\n\n"
        "The watchdog starts the child process, waits for it to be healthy,\n"
        "then listens for connections. When the child self-freezes (via --cuda-freeze),\n"
        "the watchdog thaws it on the next incoming connection.\n", prog);
}

// --- main ---

int main(int argc, char ** argv) {
    int child_start = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--") == 0) {
            child_start = i + 1;
            break;
        }
        if (strcmp(argv[i], "--listen") == 0 && i + 1 < argc) {
            char * s = argv[++i];
            char * colon = strrchr(s, ':');
            if (colon) { *colon = '\0'; g_listen_addr = s; g_listen_port = atoi(colon + 1); }
            else { g_listen_port = atoi(s); }
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            char * s = argv[++i];
            char * colon = strrchr(s, ':');
            if (colon) { *colon = '\0'; g_backend_addr = s; g_backend_port = atoi(colon + 1); }
            else { g_backend_port = atoi(s); }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]); return 0;
        } else {
            fprintf(stderr, "[watchdog] unknown option: %s\n", argv[i]);
            usage(argv[0]); return 1;
        }
    }

    if (child_start < 0 || child_start >= argc) {
        fprintf(stderr, "[watchdog] error: no child command after --\n");
        usage(argv[0]);
        return 1;
    }

    // Signals
    signal(SIGUSR1, on_sigusr1);
    signal(SIGTERM, on_shutdown);
    signal(SIGINT,  on_shutdown);
    signal(SIGCHLD, on_sigchld);
    signal(SIGPIPE, SIG_IGN);

    // Fork child
    g_child_pid = fork();
    if (g_child_pid < 0) { perror("fork"); return 1; }
    if (g_child_pid == 0) {
        execvp(argv[child_start], &argv[child_start]);
        perror("execvp");
        _exit(127);
    }

    fprintf(stderr, "[watchdog] child PID %d, waiting for ready...\n", g_child_pid);

    if (wait_for_health(120000) != 0) {
        fprintf(stderr, "[watchdog] child failed to start within 120s\n");
        kill(g_child_pid, SIGTERM);
        return 1;
    }
    fprintf(stderr, "[watchdog] child ready\n");

    // Listen socket
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) { perror("socket"); return 1; }
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in laddr;
    memset(&laddr, 0, sizeof(laddr));
    laddr.sin_family = AF_INET;
    laddr.sin_port   = htons(g_listen_port);
    inet_pton(AF_INET, g_listen_addr, &laddr.sin_addr);

    if (bind(listen_fd, (struct sockaddr *)&laddr, sizeof(laddr)) < 0) { perror("bind"); return 1; }
    if (listen(listen_fd, 32) < 0) { perror("listen"); return 1; }

    fprintf(stderr, "[watchdog] %s:%d -> %s:%d (child PID %d)\n",
            g_listen_addr, g_listen_port, g_backend_addr, g_backend_port, g_child_pid);

    // Main loop — fork per connection for concurrent requests
    while (!g_shutdown) {
        struct pollfd pfd = { listen_fd, POLLIN, 0 };
        if (poll(&pfd, 1, 1000) <= 0) continue;

        struct sockaddr_in caddr;
        socklen_t clen = sizeof(caddr);
        int client_fd = accept(listen_fd, (struct sockaddr *)&caddr, &clen);
        if (client_fd < 0) continue;

        // Thaw if frozen (only in main process, before forking)
        if (g_child_frozen) {
            struct timespec t0;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            int rc = cuda_ckpt("unlock", g_child_pid);
            if (rc != 0) {
                fprintf(stderr, "[watchdog] thaw failed (rc=%d)\n", rc);
                close(client_fd);
                continue;
            }
            g_child_frozen = 0;
            if (wait_for_health(5000) != 0) {
                fprintf(stderr, "[watchdog] child not healthy after thaw\n");
                close(client_fd);
                continue;
            }
            fprintf(stderr, "[watchdog] thawed in %ldms\n", ms_since(&t0));
        }

        // Fork a proxy child to handle this connection
        pid_t proxy_pid = fork();
        if (proxy_pid < 0) {
            perror("fork proxy");
            close(client_fd);
            continue;
        }
        if (proxy_pid == 0) {
            // Proxy child: connect to backend, shuttle bytes, exit
            close(listen_fd);
            // Reset signals to default in proxy child
            signal(SIGUSR1, SIG_DFL);
            signal(SIGCHLD, SIG_DFL);

            struct timespec t0;
            clock_gettime(CLOCK_MONOTONIC, &t0);

            int bfd = socket(AF_INET, SOCK_STREAM, 0);
            if (bfd < 0) { close(client_fd); _exit(1); }

            struct sockaddr_in baddr;
            memset(&baddr, 0, sizeof(baddr));
            baddr.sin_family = AF_INET;
            baddr.sin_port   = htons(g_backend_port);
            inet_pton(AF_INET, g_backend_addr, &baddr.sin_addr);

            if (connect(bfd, (struct sockaddr *)&baddr, sizeof(baddr)) < 0) {
                close(bfd); close(client_fd); _exit(1);
            }

            proxy_fds(client_fd, bfd);
            fprintf(stderr, "[watchdog] request done (%ldms)\n", ms_since(&t0));
            close(bfd);
            close(client_fd);
            _exit(0);
        }

        // Parent: close client fd (proxy child owns it now)
        close(client_fd);
    }

    // Shutdown
    fprintf(stderr, "[watchdog] shutting down\n");
    if (g_child_frozen) {
        cuda_ckpt("unlock", g_child_pid);
        g_child_frozen = 0;
    }
    kill(g_child_pid, SIGTERM);
    waitpid(g_child_pid, NULL, 0);
    close(listen_fd);
    return 0;
}
