#include "boundary_sort.h"

#include <string.h>

static inline uint64_t be_key(const BoundaryEdge *e)
{
    return (((uint64_t)e->remote) << 32) | (uint64_t)e->rep;
}

static inline unsigned be_digit(const BoundaryEdge *e, int shift)
{
    return (unsigned)((be_key(e) >> shift) & 0xFFu);
}

static void be_insertion_sort(BoundaryEdge *a, uint64_t n)
{
    for (uint64_t i = 1; i < n; ++i)
    {
        BoundaryEdge x = a[i];
        uint64_t kx = be_key(&x);
        uint64_t j = i;
        while (j > 0)
        {
            uint64_t kj = be_key(&a[j - 1]);
            if (kj <= kx) break;
            a[j] = a[j - 1];
            --j;
        }
        a[j] = x;
    }
}

static void be_afsort_rec(BoundaryEdge *a, uint64_t n, int shift)
{
    if (n <= 64 || shift < 0)
    {
        be_insertion_sort(a, n);
        return;
    }

    uint32_t count[256];
    uint32_t start[256];
    uint32_t next[256];

    memset(count, 0, sizeof(count));

    for (uint64_t i = 0; i < n; ++i)
        count[be_digit(&a[i], shift)]++;

    uint32_t sum = 0;
    for (int b = 0; b < 256; ++b)
    {
        start[b] = sum;
        sum += count[b];
        next[b] = start[b];
    }

    for (int b = 0; b < 256; ++b)
    {
        uint32_t end = start[b] + count[b];
        while (next[b] < end)
        {
            uint32_t i = next[b];
            unsigned d = be_digit(&a[i], shift);
            if ((int)d == b)
            {
                next[b]++;
            }
            else
            {
                uint32_t j = next[d]++;
                BoundaryEdge tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
            }
        }
    }

    if (shift == 0) return;
    int nshift = shift - 8;

    for (int b = 0; b < 256; ++b)
    {
        uint32_t c = count[b];
        if (c <= 1) continue;
        be_afsort_rec(a + start[b], (uint64_t)c, nshift);
    }
}

void boundary_edges_sort(BoundaryEdge *edges, uint64_t count)
{
    if (!edges || count <= 1) return;
    be_afsort_rec(edges, count, 56);
}

uint64_t boundary_edges_dedup(BoundaryEdge *edges, uint64_t count)
{
    if (!edges || count <= 1) return count;

    uint64_t write = 1;
    BoundaryEdge prev = edges[0];

    for (uint64_t i = 1; i < count; ++i)
    {
        BoundaryEdge cur = edges[i];
        if (cur.remote != prev.remote || cur.rep != prev.rep)
        {
            edges[write++] = cur;
            prev = cur;
        }
    }

    return write;
}
