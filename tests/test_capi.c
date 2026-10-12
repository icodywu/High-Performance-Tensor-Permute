/* The C interface, compiled as C: a 2 x 3 x 4 tensor of floats permuted to 4 x 2 x 3, and the error codes. */
#include "hptp.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    float src[24], dst[24];
    const int64_t shape[3] = {2, 3, 4}, perm[3] = {2, 0, 1}, bad[3] = {2, 2, 1};
    int i, j, k, rc, fails = 0;
    for (i = 0; i < 24; i++) src[i] = (float)i;
    rc = hptp_permute(src, dst, sizeof(float), 3, shape, perm, 2);
    if (rc != HPTP_OK) {
        printf("FAIL: hptp_permute returned %d (%s)\n", rc, hptp_strerror(rc));
        fails++;
    }
    for (k = 0; k < 4; k++)
        for (i = 0; i < 2; i++)
            for (j = 0; j < 3; j++)
                if (dst[(k * 2 + i) * 3 + j] != src[(i * 3 + j) * 4 + k]) {
                    printf("FAIL: dst[%d][%d][%d]\n", k, i, j);
                    fails++;
                }
    rc = hptp_permute(src, dst, sizeof(float), 3, shape, bad, 1);
    if (rc != HPTP_ERR_PERM || strcmp(hptp_strerror(rc), "perm is not a permutation of 0..ndim-1") != 0) {
        printf("FAIL: invalid permutation gave %d\n", rc);
        fails++;
    }
    if (hptp_auto_threads(1000) != 1 || hptp_auto_threads((size_t)1 << 30) < 1) {
        printf("FAIL: hptp_auto_threads\n");
        fails++;
    }
    if (strcmp(hptp_version(), HPTP_VERSION_STRING) != 0) {
        printf("FAIL: hptp_version\n");
        fails++;
    }
    printf("C interface %s: %s\n", hptp_version(), fails ? "FAILED" : "ok");
    return fails != 0;
}
