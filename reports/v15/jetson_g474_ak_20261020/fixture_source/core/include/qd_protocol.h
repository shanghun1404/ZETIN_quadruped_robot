#ifndef QD_PROTOCOL_H
#define QD_PROTOCOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define QD_JOINT_COUNT 12u
#define QD_HEADER_SIZE 22u
#define QD_MAX_PAYLOAD 256u
#define QD_MAX_FRAME (QD_HEADER_SIZE + QD_MAX_PAYLOAD + 2u)
#define QD_COMMAND_SIZE 244u
#define QD_FEEDBACK_SIZE 156u
#define QD_WATCHDOG_MS 100u

enum qd_type { QD_HELLO = 1, QD_HELLO_ACK = 2, QD_COMMAND = 3,
               QD_FEEDBACK = 4, QD_STOP = 5 };
enum qd_fault { QD_FAULT_WATCHDOG = 1u, QD_FAULT_COMMAND = 2u,
                QD_FAULT_GUARD = 4u, QD_FAULT_BOARD = 8u,
                QD_FAULT_STATE = 16u, QD_FAULT_IO = 32u };
enum qd_result { QD_IGNORED = 0, QD_ACCEPTED = 1, QD_REJECTED = 2,
                 QD_REPLY = 3 };

typedef struct {
    uint8_t type;
    uint32_t session;
    uint32_t sequence;
    uint64_t stamp_ns;
    uint16_t payload_length;
    uint8_t payload[QD_MAX_PAYLOAD];
} qd_frame;

/* Canonical SI output-shaft coordinates: FL/FR/RL/RR x HAA/HFE/KFE. */
typedef struct {
    uint32_t enabled;
    float position[QD_JOINT_COUNT];
    float velocity[QD_JOINT_COUNT];
    float kp[QD_JOINT_COUNT];
    float kd[QD_JOINT_COUNT];
    float effort[QD_JOINT_COUNT];
} qd_command;

typedef struct {
    float position[QD_JOINT_COUNT];
    float velocity[QD_JOINT_COUNT];
    float effort[QD_JOINT_COUNT];
} qd_measurement;

typedef struct {
    uint32_t acknowledged_sequence;
    uint32_t fault_flags;
    uint32_t armed;
    qd_measurement measured;
} qd_feedback;

typedef void (*qd_frame_callback)(void *user, const qd_frame *frame);
typedef struct {
    uint8_t bytes[QD_MAX_FRAME];
    size_t used;
    uint32_t rejected_frames;
    uint32_t emitted_frames;
} qd_parser;

/* All four callbacks must exist before an enabled command is accepted.
 * ready confirms completed CAN, calibration and motor-enable integration.
 * read returns fresh actual measurements in the canonical coordinates.
 * apply receives a validated command and the checked total PD+FF torque.
 * It must return false if any motor command could not be delivered.
 * disarm must synchronously revoke motor enable, including partial apply. */
typedef struct {
    void *user;
    bool (*ready)(void *user);
    bool (*read)(void *user, qd_measurement *measured);
    bool (*apply)(void *user, const qd_command *command,
                  const float total_torque[QD_JOINT_COUNT]);
    void (*disarm)(void *user);
} qd_board;

typedef struct {
    qd_board board;
    uint32_t session;
    uint32_t acknowledged_sequence;
    uint32_t tx_sequence;
    uint32_t last_command_ms;
    uint32_t fault_flags;
    bool have_command;
    bool armed;
    bool measurement_valid;
    qd_command command;
    qd_measurement measured;
    float total_torque[QD_JOINT_COUNT];
} qd_controller;

uint16_t qd_crc16(const uint8_t *data, size_t length);
/* These operate on structural framing. Numeric COMMAND validation belongs
 * in qd_unpack_command/controller so a bad command can disarm the MCU. */
size_t qd_encode_frame(const qd_frame *frame, uint8_t *out, size_t capacity);
bool qd_decode_frame(const uint8_t *bytes, size_t length, qd_frame *out);
void qd_parser_init(qd_parser *parser);
void qd_parser_feed(qd_parser *parser, const uint8_t *bytes, size_t length,
                    qd_frame_callback callback, void *user);
bool qd_pack_command(const qd_command *command, uint8_t out[QD_COMMAND_SIZE]);
bool qd_unpack_command(const uint8_t *payload, size_t length, qd_command *out);
bool qd_pack_feedback(const qd_feedback *feedback, uint8_t out[QD_FEEDBACK_SIZE]);
bool qd_unpack_feedback(const uint8_t *payload, size_t length, qd_feedback *out);
bool qd_sequence_newer(uint32_t candidate, uint32_t previous);
bool qd_guard_command(const qd_command *command, const qd_measurement *measured,
                      float total_torque[QD_JOINT_COUNT]);
void qd_controller_init(qd_controller *controller, const qd_board *board);
/* Monotonic uint32 milliseconds may wrap; poll more often than 2^31 ms.
 * HELLO emits HELLO_ACK via reply; the current session is idempotent and
 * only a changed nonzero session resets command/fault state.
 * Other frames leave reply untouched. */
enum qd_result qd_controller_on_frame(qd_controller *controller,
                                     const qd_frame *frame, uint32_t now_ms,
                                     qd_frame *reply);
/* Invoke at least every 5 ms and immediately before processing a frame.
 * Rechecks actual q/dq and re-applies only while armed. */
void qd_controller_tick(qd_controller *controller, uint32_t now_ms);
void qd_controller_fault(qd_controller *controller, uint32_t fault_flags);
/* No synthetic measurements: returns false until board.read succeeds. */
bool qd_controller_feedback(qd_controller *controller, uint64_t stamp_ns,
                            qd_frame *out);

#endif
