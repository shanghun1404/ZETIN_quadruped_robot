#include "qd_can_diagnostic.h"

#include <math.h>

bool qd_can_build_diagnostic(
    const qd_command *command,
    qd_can_diagnostic_frame out[QD_JOINT_COUNT])
{
    if (command == NULL || out == NULL || command->enabled != 0u)
        return false;

    /* Validate the complete command before writing any output frame. */
    for (size_t i = 0; i < QD_JOINT_COUNT; ++i) {
        if (!isfinite(command->position[i]) ||
            command->position[i] < -12.5f || command->position[i] > 12.5f ||
            command->velocity[i] != 0.0f || command->kp[i] != 0.0f ||
            command->kd[i] != 0.0f || command->effort[i] != 0.0f)
            return false;
    }

    for (size_t i = 0; i < QD_JOINT_COUNT; ++i) {
        const double scaled =
            ((double)command->position[i] + 12.5) * 65535.0 / 25.0;
        /* The validated interval is nonnegative, so the cast takes floor. */
        const uint32_t position = scaled <= 0.0 ? 0u :
                                  scaled >= 65535.0 ? 65535u :
                                  (uint32_t)scaled;

        out[i] = (qd_can_diagnostic_frame){
            .bus = (uint32_t)(i % 3u) + 1u,
            .arbitration_id = (uint32_t)(i / 3u) + 1u,
            .joint_index = (uint32_t)i,
            .data = {
                (uint8_t)(position >> 8), (uint8_t)position,
                0x80, 0, 0, 0, 0x08, 0
            }
        };
    }
    return true;
}
