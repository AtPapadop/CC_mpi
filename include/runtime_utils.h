#ifndef RUNTIME_UTILS_H
#define RUNTIME_UTILS_H

#include <mpi.h>

/* Centralized runtime helpers shared across MPI codepaths. */
void mpi_die_abort(MPI_Comm comm, const char *msg);
int runtime_default_threads(void);

#endif /* RUNTIME_UTILS_H */
