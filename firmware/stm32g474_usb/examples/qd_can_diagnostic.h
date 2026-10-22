#ifndef QD_CAN_DIAGNOSTIC_H
#define QD_CAN_DIAGNOSTIC_H

#include "qd_protocol.h"

typedef struct {
    uint32_t bus;
    uint32_t arbitration_id;
    uint32_t joint_index;
    uint8_t data[8];
} qd_can_diagnostic_frame;

/* Bare-board diagnostic encoding only, not production motor scaling.
 * Accepts disabled commands with zero velocity, gains, and requested effort.
 * Position must be finite and within [-12.5, 12.5]. The 12-bit midpoint
 * fields do not guarantee exactly zero physical velocity or torque.
 * Returns false for invalid/null arguments without changing out. */
bool qd_can_build_diagnostic(
    const qd_command *command,
    qd_can_diagnostic_frame out[QD_JOINT_COUNT]);

#endif
