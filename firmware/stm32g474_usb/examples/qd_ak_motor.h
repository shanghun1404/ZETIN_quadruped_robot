#ifndef QD_AK_MOTOR_H
#define QD_AK_MOTOR_H

#include "qd_protocol.h"

/* Real AK actuator MIT adapter behind the qd_board callbacks.
 * Bit layouts and float/uint conversions follow CubeMars manuals:
 *   LEGACY: "AK Series Module Driver User Manual" V1.0.15.X, section 5.3
 *   V3:     "AKA Series Module Product Manual" V3.0.0, sections 4.2, 4.3.1
 * Ranges, reply identifiers and feedback scales are NOT built in. The V3
 * manual's range table and example code disagree, and neither manual defines
 * the V3 upload identifier or output-shaft feedback units. Every value comes
 * from qd_ak_motor_config, measured on the installed units. */

#define QD_AK_CONFIG_VERIFIED 0x414b5631u /* "AKV1": set only after measurement */
#define QD_AK_FRESH_MS 10u                /* two 5 ms ticks per reply */

enum qd_ak_protocol { QD_AK_LEGACY_MIT = 1, QD_AK_V3_MIT = 2 };

typedef struct {
    uint32_t bus;      /* FDCAN 1..3 */
    uint32_t extended;
    uint32_t id;
    uint32_t dlc;
    uint8_t data[8];
} qd_ak_can_frame;

/* Canonical (CAD/policy) = sign * (motor - zero). The verified flag asserts
 * that MIT position/velocity/torque are already output-shaft SI values. */
typedef struct {
    uint32_t protocol;  /* qd_ak_protocol */
    uint32_t bus;       /* 1..3 */
    uint32_t motor_id;  /* driver ID 1..255 */
    uint32_t reply_id;  /* exact identifier of this motor's reply frame */
    float sign;         /* +1 or -1 */
    float zero;         /* motor position (rad) at canonical zero */
    /* MIT command span: symmetric [-max, max]; gains [0, max]. */
    float position_max, velocity_max, torque_max, kp_max, kd_max;
    /* LEGACY: symmetric full scale of reply position/velocity/effort fields.
     * V3: output-shaft rad, rad/s and N*m per int16 count of the upload. */
    float reply_position, reply_velocity, reply_effort;
} qd_ak_joint_config;

typedef struct {
    uint32_t verified;  /* QD_AK_CONFIG_VERIFIED */
    qd_ak_joint_config joint[QD_JOINT_COUNT];
} qd_ak_motor_config;

typedef bool (*qd_ak_send)(void *user, const qd_ak_can_frame *frame);

typedef struct {
    float position, velocity, effort; /* canonical SI */
    uint32_t stamp_ms;
    int32_t temperature_c;
    uint32_t error_code;
    bool seen;
    bool motor_mode; /* LEGACY: enter frame sent since the last rest frame */
} qd_ak_joint_state;

typedef struct {
    qd_ak_motor_config config;
    bool config_valid;
    qd_ak_send send;
    void *send_user;
    uint32_t now_ms;
    qd_ak_joint_state joint[QD_JOINT_COUNT];
    uint32_t rejected_frames;
    uint32_t send_failures;
} qd_ak_motor;

bool qd_ak_config_valid(const qd_ak_motor_config *config);
/* An invalid or unverified config leaves the adapter never ready. */
void qd_ak_motor_init(qd_ak_motor *motor, const qd_ak_motor_config *config,
                      qd_ak_send send, void *send_user);
/* Call with the monotonic millisecond clock before controller processing.
 * A reply older than QD_AK_FRESH_MS stays stale until the next reply. */
void qd_ak_motor_clock(qd_ak_motor *motor, uint32_t now_ms);
/* Foreground only; stamps with the last clock. Returns true only for a
 * well-formed reply from a configured motor. */
bool qd_ak_motor_receive(qd_ak_motor *motor, const qd_ak_can_frame *frame);
/* Call every 5 ms while disarmed: LEGACY exit-motor-mode frame,
 * V3 MIT frame with zero gains, velocity and torque. */
bool qd_ak_motor_poll(qd_ak_motor *motor);
/* Canonical inputs. Out-of-span or non-finite values are rejected, not clipped. */
bool qd_ak_motor_encode(const qd_ak_joint_config *joint, float position,
                        float velocity, float kp, float kd, float effort,
                        qd_ak_can_frame *out);
qd_board qd_ak_motor_board(qd_ak_motor *motor);

#endif
