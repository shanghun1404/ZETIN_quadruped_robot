#include "qd_ak_motor.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static unsigned failed;
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: FAIL %s\n", __func__, __LINE__, #x); \
    ++failed; return; \
} } while (0)

/* TEST ONLY: power-of-two spans make hand-computed codes exact.
 * These are not motor facts and must not be used as a robot config. */
static qd_ak_motor_config test_config(void)
{
    qd_ak_motor_config config;
    memset(&config, 0, sizeof(config));
    config.verified = QD_AK_CONFIG_VERIFIED;
    for (unsigned i = 0; i < QD_JOINT_COUNT; ++i) {
        qd_ak_joint_config *j = &config.joint[i];
        bool legacy = i % 3u == 0u;
        j->protocol = legacy ? QD_AK_LEGACY_MIT : QD_AK_V3_MIT;
        j->bus = i % 3u + 1u;
        j->motor_id = i / 3u + 1u;
        j->reply_id = legacy ? 0u : 0x2900u | j->motor_id;
        j->sign = 1.0f;
        j->position_max = 16.0f;
        j->velocity_max = 32.0f;
        j->torque_max = 64.0f;
        j->kp_max = 512.0f;
        j->kd_max = 8.0f;
        j->reply_position = legacy ? 16.0f : 0.125f;
        j->reply_velocity = legacy ? 32.0f : 0.25f;
        j->reply_effort = legacy ? 64.0f : 0.0625f;
    }
    return config;
}

typedef struct {
    qd_ak_can_frame frames[256];
    unsigned count;
    unsigned attempts;
    unsigned fail_at; /* 1-based attempt that fails; 0 = never */
} capture;

static bool capture_send(void *user, const qd_ak_can_frame *frame)
{
    capture *c = user;
    ++c->attempts;
    if (c->attempts == c->fail_at)
        return false;
    if (c->count < sizeof(c->frames) / sizeof(c->frames[0]))
        c->frames[c->count++] = *frame;
    return true;
}

static bool is_legacy_special(const qd_ak_can_frame *f, uint8_t code)
{
    static const uint8_t ff[7] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    return f->extended == 0u && f->dlc == 8u && memcmp(f->data, ff, 7) == 0 &&
           f->data[7] == code;
}

static bool is_v3_rest(const qd_ak_can_frame *f)
{
    /* kp=kd=0, position/velocity/torque at the manual's zero code. */
    static const uint8_t zero[8] = {0, 0, 0, 0x80, 0x00, 0x80, 0x08, 0x00};
    return f->extended == 1u && (f->id >> 8) == 8u && memcmp(f->data, zero, 8) == 0;
}

static qd_ak_can_frame v3_reply(unsigned joint, int16_t pos, int16_t spd, int16_t cur,
                                uint8_t temp, uint8_t error)
{
    qd_ak_can_frame f = {0};
    f.bus = joint % 3u + 1u;
    f.extended = 1;
    f.id = 0x2900u | (joint / 3u + 1u);
    f.dlc = 8;
    f.data[0] = (uint8_t)((uint16_t)pos >> 8);
    f.data[1] = (uint8_t)pos;
    f.data[2] = (uint8_t)((uint16_t)spd >> 8);
    f.data[3] = (uint8_t)spd;
    f.data[4] = (uint8_t)((uint16_t)cur >> 8);
    f.data[5] = (uint8_t)cur;
    f.data[6] = temp;
    f.data[7] = error;
    return f;
}

static qd_ak_can_frame legacy_reply(unsigned joint, uint16_t p, uint16_t v, uint16_t i,
                                    uint8_t temp, uint8_t error)
{
    qd_ak_can_frame f = {0};
    f.bus = joint % 3u + 1u;
    f.id = 0;
    f.dlc = 8;
    f.data[0] = (uint8_t)(joint / 3u + 1u);
    f.data[1] = (uint8_t)(p >> 8);
    f.data[2] = (uint8_t)p;
    f.data[3] = (uint8_t)(v >> 4);
    f.data[4] = (uint8_t)((v & 0xfu) << 4 | i >> 8);
    f.data[5] = (uint8_t)i;
    f.data[6] = temp;
    f.data[7] = error;
    return f;
}

static void feed_all(qd_ak_motor *motor)
{
    for (unsigned j = 0; j < QD_JOINT_COUNT; ++j) {
        qd_ak_can_frame f = j % 3u == 0u ? legacy_reply(j, 0x8000, 0x800, 0x800, 65, 0)
                                         : v3_reply(j, 0, 0, 0, 25, 0);
        (void)qd_ak_motor_receive(motor, &f);
    }
}

static void manual_bit_layouts_match_hand_computed_vectors(void)
{
    /* p=0xa010 v=0x601 kp=0x208 kd=0xc00 t=0xa01 under the test spans. */
    static const uint8_t v3[8] = {0x20, 0x8c, 0x00, 0xa0, 0x10, 0x60, 0x1a, 0x01};
    static const uint8_t legacy[8] = {0xa0, 0x10, 0x60, 0x12, 0x08, 0xc0, 0x0a, 0x01};
    qd_ak_motor_config config = test_config();
    qd_ak_can_frame f;
    CHECK(qd_ak_motor_encode(&config.joint[1], 4.0078125f, -7.984375f, 65.0f, 6.0f,
                             16.03125f, &f));
    CHECK(f.bus == 2 && f.extended == 1 && f.id == 0x801 && f.dlc == 8);
    CHECK(memcmp(f.data, v3, 8) == 0);
    CHECK(qd_ak_motor_encode(&config.joint[0], 4.0078125f, -7.984375f, 65.0f, 6.0f,
                             16.03125f, &f));
    CHECK(f.bus == 1 && f.extended == 0 && f.id == 1 && f.dlc == 8);
    CHECK(memcmp(f.data, legacy, 8) == 0);
}

static void sign_and_zero_map_canonical_to_motor_coordinates(void)
{
    static const uint8_t v3[8] = {0x20, 0x8c, 0x00, 0xa0, 0x10, 0x60, 0x1a, 0x01};
    qd_ak_motor_config config = test_config();
    qd_ak_can_frame f;
    config.joint[1].sign = -1.0f;
    config.joint[1].zero = 0.5f;
    /* motor = -canonical + 0.5; gains are sign invariant. */
    CHECK(qd_ak_motor_encode(&config.joint[1], -3.5078125f, 7.984375f, 65.0f, 6.0f,
                             -16.03125f, &f));
    CHECK(memcmp(f.data, v3, 8) == 0);
}

static void span_edges_keep_top_code_and_reject_instead_of_clipping(void)
{
    qd_ak_motor_config config = test_config();
    qd_ak_joint_config *j = &config.joint[0];
    qd_ak_can_frame f, before;
    CHECK(qd_ak_motor_encode(j, 16.0f, 32.0f, 512.0f, 8.0f, 64.0f, &f));
    CHECK(f.data[0] == 0xff && f.data[1] == 0xff && f.data[2] == 0xff &&
          f.data[3] == 0xff && f.data[4] == 0xff && f.data[5] == 0xff &&
          f.data[6] == 0xff && f.data[7] == 0xff);
    CHECK(qd_ak_motor_encode(j, -16.0f, -32.0f, 0.0f, 0.0f, -64.0f, &f));
    CHECK(f.data[0] == 0 && f.data[1] == 0 && f.data[2] == 0 && f.data[7] == 0);
    memcpy(&before, &f, sizeof(f));
    CHECK(!qd_ak_motor_encode(j, nextafterf(16.0f, 32.0f), 0, 0, 0, 0, &f));
    CHECK(!qd_ak_motor_encode(j, 0, -32.01f, 0, 0, 0, &f));
    CHECK(!qd_ak_motor_encode(j, 0, 0, 512.5f, 0, 0, &f));
    CHECK(!qd_ak_motor_encode(j, 0, 0, -0.5f, 0, 0, &f));
    CHECK(!qd_ak_motor_encode(j, 0, 0, 0, 8.01f, 0, &f));
    CHECK(!qd_ak_motor_encode(j, 0, 0, 0, 0, 64.5f, &f));
    CHECK(!qd_ak_motor_encode(j, NAN, 0, 0, 0, 0, &f));
    CHECK(!qd_ak_motor_encode(j, 0, INFINITY, 0, 0, 0, &f));
    CHECK(!qd_ak_motor_encode(j, 0, 0, 0, NAN, 0, &f));
    j->zero = 1.0f; /* motor = 15.5 + 1 exceeds the span */
    CHECK(!qd_ak_motor_encode(j, 15.5f, 0, 0, 0, 0, &f));
    CHECK(memcmp(&f, &before, sizeof(f)) == 0);
}

static void config_validation_rejects_unsafe_tables(void)
{
    qd_ak_motor_config c = test_config();
    CHECK(qd_ak_config_valid(&c));
    CHECK(!qd_ak_config_valid(NULL));
    c.verified = 0; CHECK(!qd_ak_config_valid(&c));
    c = test_config(); c.joint[4].sign = 0.5f; CHECK(!qd_ak_config_valid(&c));
    c = test_config(); c.joint[4].bus = 4; CHECK(!qd_ak_config_valid(&c));
    c = test_config(); c.joint[4].motor_id = 0; CHECK(!qd_ak_config_valid(&c));
    c = test_config(); c.joint[4].motor_id = 256; CHECK(!qd_ak_config_valid(&c));
    c = test_config(); c.joint[4].protocol = 3; CHECK(!qd_ak_config_valid(&c));
    c = test_config(); c.joint[4].kd_max = 0.0f; CHECK(!qd_ak_config_valid(&c));
    c = test_config(); c.joint[4].reply_effort = NAN; CHECK(!qd_ak_config_valid(&c));
    c = test_config(); c.joint[4].zero = 16.0f; CHECK(!qd_ak_config_valid(&c));
    c = test_config(); c.joint[0].reply_id = 0x800; CHECK(!qd_ak_config_valid(&c));
    /* Joints 1 and 4 share bus 2; same ID on another bus is allowed. */
    c = test_config(); c.joint[4].motor_id = 1; CHECK(!qd_ak_config_valid(&c));
    c = test_config(); c.joint[4].motor_id = 9; c.joint[5].motor_id = 9;
    CHECK(qd_ak_config_valid(&c));
    /* A reply must not look like a command to another motor on the bus. */
    c = test_config(); c.joint[0].reply_id = 2; CHECK(!qd_ak_config_valid(&c));
    c = test_config(); c.joint[1].reply_id = 0x802; CHECK(!qd_ak_config_valid(&c));
    c = test_config(); c.joint[1].reply_id = 0x2902; CHECK(!qd_ak_config_valid(&c));
}

static void unverified_or_unsendable_adapter_never_transmits(void)
{
    capture cap = {0};
    qd_ak_motor motor;
    qd_ak_motor_config config = test_config();
    qd_measurement m;
    qd_command command = {0};
    float total[QD_JOINT_COUNT] = {0};
    config.verified = 0;
    qd_ak_motor_init(&motor, &config, capture_send, &cap);
    qd_board board = qd_ak_motor_board(&motor);
    qd_ak_motor_clock(&motor, 1);
    feed_all(&motor);
    CHECK(!board.ready(board.user));
    CHECK(!board.read(board.user, &m));
    CHECK(!board.apply(board.user, &command, total));
    board.disarm(board.user);
    CHECK(!qd_ak_motor_poll(&motor));
    CHECK(cap.attempts == 0 && motor.rejected_frames == QD_JOINT_COUNT);
    config = test_config();
    qd_ak_motor_init(&motor, &config, NULL, NULL);
    CHECK(!motor.config_valid);
}

static void replies_decode_to_canonical_si_with_sign_and_zero(void)
{
    capture cap = {0};
    qd_ak_motor motor;
    qd_ak_motor_config config = test_config();
    config.joint[4].sign = -1.0f;
    config.joint[4].zero = 0.5f;
    qd_ak_motor_init(&motor, &config, capture_send, &cap);
    qd_ak_motor_clock(&motor, 1000);
    qd_ak_can_frame f = v3_reply(1, -80, 12, -32, 0xec, 0);
    CHECK(qd_ak_motor_receive(&motor, &f));
    CHECK(motor.joint[1].position == -10.0f && motor.joint[1].velocity == 3.0f &&
          motor.joint[1].effort == -2.0f && motor.joint[1].temperature_c == -20 &&
          motor.joint[1].stamp_ms == 1000);
    f = v3_reply(4, -80, 12, -32, 30, 0);
    CHECK(qd_ak_motor_receive(&motor, &f));
    CHECK(motor.joint[4].position == 10.5f && motor.joint[4].velocity == -3.0f &&
          motor.joint[4].effort == 2.0f);
    f = legacy_reply(3, 0xffff, 0, 0xfff, 65, 0);
    CHECK(qd_ak_motor_receive(&motor, &f));
    CHECK(motor.joint[3].position == 16.0f && motor.joint[3].velocity == -32.0f &&
          motor.joint[3].effort == 64.0f && motor.joint[3].temperature_c == 25);
}

static void malformed_or_foreign_frames_change_no_state(void)
{
    capture cap = {0};
    qd_ak_motor motor;
    qd_ak_motor_config config = test_config();
    qd_ak_motor_init(&motor, &config, capture_send, &cap);
    qd_ak_motor_clock(&motor, 5);
    qd_ak_joint_state before[QD_JOINT_COUNT];
    memcpy(before, motor.joint, sizeof(before));
    qd_ak_can_frame good = v3_reply(1, 1, 1, 1, 0, 0), f;
    f = good; f.bus = 1; CHECK(!qd_ak_motor_receive(&motor, &f));
    f = good; f.id = 0x2905; CHECK(!qd_ak_motor_receive(&motor, &f));
    f = good; f.extended = 0; CHECK(!qd_ak_motor_receive(&motor, &f));
    f = good; f.dlc = 7; CHECK(!qd_ak_motor_receive(&motor, &f));
    f = legacy_reply(0, 0, 0, 0, 0, 0); f.data[0] = 7; CHECK(!qd_ak_motor_receive(&motor, &f));
    f = legacy_reply(0, 0, 0, 0, 0, 0); f.extended = 1; CHECK(!qd_ak_motor_receive(&motor, &f));
    CHECK(!qd_ak_motor_receive(&motor, NULL));
    CHECK(memcmp(before, motor.joint, sizeof(before)) == 0);
    CHECK(motor.rejected_frames == 6);
}

static void stale_or_faulted_motor_blocks_ready_and_read(void)
{
    capture cap = {0};
    qd_ak_motor motor;
    qd_ak_motor_config config = test_config();
    qd_measurement m;
    qd_ak_motor_init(&motor, &config, capture_send, &cap);
    qd_board board = qd_ak_motor_board(&motor);
    qd_ak_motor_clock(&motor, 100);
    CHECK(!board.ready(board.user));
    feed_all(&motor);
    CHECK(board.ready(board.user) && board.read(board.user, &m));
    qd_ak_motor_clock(&motor, 100 + QD_AK_FRESH_MS);
    CHECK(board.ready(board.user));
    qd_ak_motor_clock(&motor, 101 + QD_AK_FRESH_MS);
    CHECK(!board.ready(board.user) && !board.read(board.user, &m));
    /* Once stale, a later clock value can never revive old data. */
    qd_ak_motor_clock(&motor, 100);
    CHECK(!board.ready(board.user));
    feed_all(&motor);
    qd_ak_can_frame f = v3_reply(7, 0, 0, 0, 25, 2);
    CHECK(qd_ak_motor_receive(&motor, &f));
    CHECK(!board.ready(board.user) && !board.read(board.user, &m));
}

static void rest_frames_are_exit_for_legacy_and_zero_torque_for_v3(void)
{
    capture cap = {0};
    qd_ak_motor motor;
    qd_ak_motor_config config = test_config();
    qd_ak_motor_init(&motor, &config, capture_send, &cap);
    CHECK(qd_ak_motor_poll(&motor));
    CHECK(cap.count == QD_JOINT_COUNT);
    for (unsigned j = 0; j < QD_JOINT_COUNT; ++j) {
        const qd_ak_can_frame *f = &cap.frames[j];
        CHECK(f->bus == j % 3u + 1u);
        if (j % 3u == 0u)
            CHECK(f->id == j / 3u + 1u && is_legacy_special(f, 0xfd));
        else
            CHECK(f->id == (0x800u | (j / 3u + 1u)) && is_v3_rest(f));
    }
}

static void rest_continues_to_every_motor_after_a_failed_send(void)
{
    capture cap = {0};
    qd_ak_motor motor;
    qd_ak_motor_config config = test_config();
    qd_ak_motor_init(&motor, &config, capture_send, &cap);
    cap.fail_at = 3;
    CHECK(!qd_ak_motor_poll(&motor));
    CHECK(cap.attempts == QD_JOINT_COUNT && cap.count == QD_JOINT_COUNT - 1u);
    CHECK(motor.send_failures == 1);
}

static qd_frame frame_of(uint8_t type, uint32_t sequence, const qd_command *command)
{
    qd_frame f = {0};
    f.type = type;
    f.session = 7;
    f.sequence = sequence;
    if (command != NULL) {
        f.payload_length = QD_COMMAND_SIZE;
        if (!qd_pack_command(command, f.payload))
            memset(f.payload, 0, QD_COMMAND_SIZE);
    }
    return f;
}

static qd_command hold_command(float kd)
{
    qd_command c = {0};
    c.enabled = 1;
    for (unsigned j = 0; j < QD_JOINT_COUNT; ++j) {
        c.kp[j] = 10.0f;
        c.kd[j] = kd;
    }
    return c;
}

static void controller_arms_enters_legacy_once_and_rests_on_stop(void)
{
    capture cap = {0};
    qd_ak_motor motor;
    qd_ak_motor_config config = test_config();
    qd_controller controller;
    qd_frame reply;
    qd_ak_motor_init(&motor, &config, capture_send, &cap);
    qd_board board = qd_ak_motor_board(&motor);
    qd_controller_init(&controller, &board);
    CHECK(cap.count == QD_JOINT_COUNT); /* init disarm sends rest frames */
    qd_ak_motor_clock(&motor, 10);
    feed_all(&motor);
    qd_frame hello = frame_of(QD_HELLO, 1, NULL);
    CHECK(qd_controller_on_frame(&controller, &hello, 10, &reply) == QD_REPLY);
    unsigned mark = cap.count;
    qd_command c = hold_command(1.0f);
    qd_frame cmd = frame_of(QD_COMMAND, 2, &c);
    CHECK(qd_controller_on_frame(&controller, &cmd, 10, &reply) == QD_ACCEPTED);
    CHECK(controller.armed);
    CHECK(cap.count - mark == QD_JOINT_COUNT + 4u);
    for (unsigned k = mark, j = 0; j < QD_JOINT_COUNT; ++j) {
        if (j % 3u == 0u)
            CHECK(is_legacy_special(&cap.frames[k++], 0xfc));
        CHECK(!is_legacy_special(&cap.frames[k], 0xfc) && !is_legacy_special(&cap.frames[k], 0xfd));
        ++k;
    }
    mark = cap.count;
    cmd = frame_of(QD_COMMAND, 3, &c);
    CHECK(qd_controller_on_frame(&controller, &cmd, 12, &reply) == QD_ACCEPTED);
    CHECK(cap.count - mark == QD_JOINT_COUNT); /* already in motor mode */
    mark = cap.count;
    qd_frame stop = frame_of(QD_STOP, 4, NULL);
    CHECK(qd_controller_on_frame(&controller, &stop, 13, &reply) == QD_ACCEPTED);
    CHECK(!controller.armed && cap.count - mark == QD_JOINT_COUNT);
    CHECK(is_legacy_special(&cap.frames[mark], 0xfd));
    mark = cap.count;
    cmd = frame_of(QD_COMMAND, 5, &c);
    CHECK(qd_controller_on_frame(&controller, &cmd, 14, &reply) == QD_ACCEPTED);
    CHECK(cap.count - mark == QD_JOINT_COUNT + 4u && is_legacy_special(&cap.frames[mark], 0xfc));
}

static void watchdog_and_stale_feedback_rest_all_motors(void)
{
    capture cap = {0};
    qd_ak_motor motor;
    qd_ak_motor_config config = test_config();
    qd_controller controller;
    qd_frame reply;
    qd_ak_motor_init(&motor, &config, capture_send, &cap);
    qd_board board = qd_ak_motor_board(&motor);
    qd_controller_init(&controller, &board);
    qd_ak_motor_clock(&motor, 10);
    feed_all(&motor);
    qd_frame hello = frame_of(QD_HELLO, 1, NULL);
    (void)qd_controller_on_frame(&controller, &hello, 10, &reply);
    qd_command c = hold_command(1.0f);
    qd_frame cmd = frame_of(QD_COMMAND, 2, &c);
    CHECK(qd_controller_on_frame(&controller, &cmd, 10, &reply) == QD_ACCEPTED);
    /* Motor replies stop: the next tick sees stale state and rests. */
    unsigned mark = cap.count;
    qd_ak_motor_clock(&motor, 10 + QD_AK_FRESH_MS + 1u);
    qd_controller_tick(&controller, 10 + QD_AK_FRESH_MS + 1u);
    CHECK(!controller.armed && (controller.fault_flags & QD_FAULT_BOARD) != 0u);
    CHECK(cap.count - mark == QD_JOINT_COUNT && is_legacy_special(&cap.frames[mark], 0xfd));
    /* Fresh replies but no commands: the 100 ms watchdog rests. */
    hello = frame_of(QD_HELLO, 9, NULL);
    hello.session = 8;
    qd_ak_motor_clock(&motor, 200);
    feed_all(&motor);
    (void)qd_controller_on_frame(&controller, &hello, 200, &reply);
    cmd = frame_of(QD_COMMAND, 10, &c);
    cmd.session = 8;
    CHECK(qd_controller_on_frame(&controller, &cmd, 200, &reply) == QD_ACCEPTED);
    mark = cap.count;
    qd_ak_motor_clock(&motor, 300);
    feed_all(&motor);
    qd_controller_tick(&controller, 300);
    CHECK(!controller.armed && (controller.fault_flags & QD_FAULT_WATCHDOG) != 0u);
    CHECK(cap.count - mark == QD_JOINT_COUNT);
}

static void unsupported_gain_sends_no_command_and_rests(void)
{
    capture cap = {0};
    qd_ak_motor motor;
    qd_ak_motor_config config = test_config();
    qd_controller controller;
    qd_frame reply;
    config.joint[5].kd_max = 4.0f; /* below the USB limit of 8 */
    qd_ak_motor_init(&motor, &config, capture_send, &cap);
    qd_board board = qd_ak_motor_board(&motor);
    qd_controller_init(&controller, &board);
    qd_ak_motor_clock(&motor, 10);
    feed_all(&motor);
    qd_frame hello = frame_of(QD_HELLO, 1, NULL);
    (void)qd_controller_on_frame(&controller, &hello, 10, &reply);
    unsigned mark = cap.count;
    qd_command c = hold_command(4.5f);
    qd_frame cmd = frame_of(QD_COMMAND, 2, &c);
    CHECK(qd_controller_on_frame(&controller, &cmd, 10, &reply) == QD_REJECTED);
    CHECK(!controller.armed && (controller.fault_flags & QD_FAULT_BOARD) != 0u);
    /* Only the 12 rest frames from disarm; no partial command. */
    CHECK(cap.count - mark == QD_JOINT_COUNT);
    CHECK(is_legacy_special(&cap.frames[mark], 0xfd) && is_v3_rest(&cap.frames[mark + 1]));
}

static void failed_send_mid_apply_rests_every_motor(void)
{
    capture cap = {0};
    qd_ak_motor motor;
    qd_ak_motor_config config = test_config();
    qd_controller controller;
    qd_frame reply;
    qd_ak_motor_init(&motor, &config, capture_send, &cap);
    qd_board board = qd_ak_motor_board(&motor);
    qd_controller_init(&controller, &board);
    qd_ak_motor_clock(&motor, 10);
    feed_all(&motor);
    qd_frame hello = frame_of(QD_HELLO, 1, NULL);
    (void)qd_controller_on_frame(&controller, &hello, 10, &reply);
    cap.fail_at = cap.attempts + 7u;
    unsigned before = cap.attempts;
    qd_command c = hold_command(1.0f);
    qd_frame cmd = frame_of(QD_COMMAND, 2, &c);
    CHECK(qd_controller_on_frame(&controller, &cmd, 10, &reply) == QD_REJECTED);
    CHECK(!controller.armed && motor.send_failures == 1);
    /* 7 apply attempts, then 12 rest attempts; motor mode cleared. */
    CHECK(cap.attempts - before == 7u + QD_JOINT_COUNT);
    for (unsigned j = 0; j < QD_JOINT_COUNT; ++j)
        CHECK(!motor.joint[j].motor_mode);
}

int main(void)
{
    static const struct { const char *name; void (*run)(void); } tests[] = {
        {"manual_bit_layouts_match_hand_computed_vectors", manual_bit_layouts_match_hand_computed_vectors},
        {"sign_and_zero_map_canonical_to_motor_coordinates", sign_and_zero_map_canonical_to_motor_coordinates},
        {"span_edges_keep_top_code_and_reject_instead_of_clipping", span_edges_keep_top_code_and_reject_instead_of_clipping},
        {"config_validation_rejects_unsafe_tables", config_validation_rejects_unsafe_tables},
        {"unverified_or_unsendable_adapter_never_transmits", unverified_or_unsendable_adapter_never_transmits},
        {"replies_decode_to_canonical_si_with_sign_and_zero", replies_decode_to_canonical_si_with_sign_and_zero},
        {"malformed_or_foreign_frames_change_no_state", malformed_or_foreign_frames_change_no_state},
        {"stale_or_faulted_motor_blocks_ready_and_read", stale_or_faulted_motor_blocks_ready_and_read},
        {"rest_frames_are_exit_for_legacy_and_zero_torque_for_v3", rest_frames_are_exit_for_legacy_and_zero_torque_for_v3},
        {"rest_continues_to_every_motor_after_a_failed_send", rest_continues_to_every_motor_after_a_failed_send},
        {"controller_arms_enters_legacy_once_and_rests_on_stop", controller_arms_enters_legacy_once_and_rests_on_stop},
        {"watchdog_and_stale_feedback_rest_all_motors", watchdog_and_stale_feedback_rest_all_motors},
        {"unsupported_gain_sends_no_command_and_rests", unsupported_gain_sends_no_command_and_rests},
        {"failed_send_mid_apply_rests_every_motor", failed_send_mid_apply_rests_every_motor}
    };
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
        unsigned before = failed;
        tests[i].run();
        printf("%s %s\n", before == failed ? "PASS" : "FAIL", tests[i].name);
    }
    printf("%u failure(s)\n", failed);
    return failed ? 1 : 0;
}
