#ifndef QD_AK_VIRTUAL_H
#define QD_AK_VIRTUAL_H

#include "qd_protocol.h"

enum qd_ak_mode { QD_AK_MIT = 0, QD_AK_SERVO = 1 };

typedef struct qd_ak_frame {
    uint32_t bus;
    uint32_t arbitration_id;
    uint32_t joint_index;
    uint32_t extended;
    uint32_t dlc;
    uint8_t data[8];
} qd_ak_frame;

/* Fixed virtual profiles: bus 1 AK70-10 legacy, buses 2/3 AK10-9 V3.
 * Commands must be disabled with zero velocity, gains, and requested effort.
 * This is kinematic protocol emulation, not a force plant or motor driver.
 * Every failure leaves the output unchanged. */
bool qd_ak_build_command(unsigned mode, const qd_command *command,
                         unsigned joint, qd_ak_frame *out);
bool qd_ak_virtual_reply(unsigned mode, const qd_ak_frame *received,
                         qd_ak_frame *out);
/* Updates only the addressed joint. Velocity and effort are virtual zeros. */
bool qd_ak_decode_reply(unsigned mode, const qd_ak_frame *received,
                        qd_measurement *out);

#endif
