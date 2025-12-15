#define _POSIX_C_SOURCE 200112L

#include "runtime_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void mpi_die_abort(MPI_Comm comm, const char *msg)
{
    if (!msg) msg = "mpi_die_abort";
    fprintf(stderr, "%s\n", msg);
    MPI_Abort(comm, EXIT_FAILURE);
}

int runtime_default_threads(void)
{
#ifdef _SC_NPROCESSORS_ONLN
    long t = sysconf(_SC_NPROCESSORS_ONLN);
    return (t > 0) ? (int)t : 1;
#else
    return 1;
#endif
}
