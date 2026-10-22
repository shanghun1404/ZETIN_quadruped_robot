#include "qd_ak_motor.h"

#include <math.h>
#include <string.h>

#define QD_AK_V3_MIT_PACKET 8u /* V3 manual 4.2: force control mode ID */
#define QD_AK_ENTER 0xfcu      /* LEGACY manual 5.3 special codes */
#define QD_AK_EXIT 0xfdu

static bool span_valid(float value)
{
    return isfinite(value) && value > 0.0f;
}

static bool joint_config_valid(const qd_ak_joint_config *joint)
{
    bool legacy = joint->protocol == QD_AK_LEGACY_MIT;
    if (!legacy && joint->protocol != QD_AK_V3_MIT)
        return false;
    if (joint->bus < 1u || joint->bus > 3u || joint->motor_id < 1u ||
        joint->motor_id > 255u || joint->reply_id > (legacy ? 0x7ffu : 0x1fffffffu))
        return false;
    if (joint->sign != 1.0f && joint->sign != -1.0f)
        return false;
    if (!span_valid(joint->position_max) || !span_valid(joint->velocity_max) ||
        !span_valid(joint->torque_max) || !span_valid(joint->kp_max) ||
        !span_valid(joint->kd_max) || !span_valid(joint->reply_position) ||
        !span_valid(joint->reply_velocity) || !span_valid(joint->reply_effort))
        return false;
    /* Canonical zero must itself be commandable. */
    return isfinite(joint->zero) && fabsf(joint->zero) < joint->position_max;
}

static uint32_t command_id(const qd_ak_joint_config *joint)
{
    return joint->protocol == QD_AK_LEGACY_MIT ? joint->motor_id
                                               : QD_AK_V3_MIT_PACKET << 8 | joint->motor_id;
}

bool qd_ak_config_valid(const qd_ak_motor_config *config)
{
    if (config == NULL || config->verified != QD_AK_CONFIG_VERIFIED)
        return false;
    for (unsigned i = 0; i < QD_JOINT_COUNT; ++i) {
        const qd_ak_joint_config *a = &config->joint[i];
        if (!joint_config_valid(a))
            return false;
        for (unsigned k = 0; k < QD_JOINT_COUNT; ++k) {
            const qd_ak_joint_config *b = &config->joint[k];
            if (k == i || b->bus != a->bus)
                continue;
            if (b->motor_id == a->motor_id)
                return false;
            /* A reply on the bus must never be a command to another motor. */
            if (a->protocol == b->protocol && a->reply_id == command_id(b))
                return false;
            if (a->protocol == QD_AK_V3_MIT && b->protocol == QD_AK_V3_MIT &&
                a->reply_id == b->reply_id)
                return false;
        }
    }
    return true;
}

static bool in_span(float value, float maximum)
{
    return isfinite(value) && value >= -maximum && value <= maximum;
}

static bool in_gain(float value, float maximum)
{
    return isfinite(value) && value >= 0.0f && value <= maximum;
}

/* Manual float_to_uint; the top code stays in range at value == maximum. */
static uint32_t to_uint(float value, float minimum, float maximum, unsigned bits)
{
    uint32_t top = (1u << bits) - 1u;
    uint32_t code = (uint32_t)((value - minimum) * ((float)(1u << bits) / (maximum - minimum)));
    return code > top ? top : code;
}

/* Manual uint_to_float. */
static float from_uint(uint32_t code, float minimum, float maximum, unsigned bits)
{
    return (float)code * (maximum - minimum) / (float)((1u << bits) - 1u) + minimum;
}

static int32_t read_i16(const uint8_t *data)
{
    uint32_t raw = (uint32_t)data[0] << 8 | data[1];
    return (int32_t)raw - ((raw & 0x8000u) != 0u ? 65536 : 0);
}

/* Motor coordinates, already range-checked. */
static void pack_mit(const qd_ak_joint_config *joint, float position, float velocity,
                     float kp, float kd, float torque, qd_ak_can_frame *out)
{
    uint32_t p = to_uint(position, -joint->position_max, joint->position_max, 16);
    uint32_t v = to_uint(velocity, -joint->velocity_max, joint->velocity_max, 12);
    uint32_t kp_code = to_uint(kp, 0.0f, joint->kp_max, 12);
    uint32_t kd_code = to_uint(kd, 0.0f, joint->kd_max, 12);
    uint32_t t = to_uint(torque, -joint->torque_max, joint->torque_max, 12);
    qd_ak_can_frame frame = {0};
    uint8_t *d = frame.data;
    frame.bus = joint->bus;
    frame.id = command_id(joint);
    frame.dlc = 8;
    if (joint->protocol == QD_AK_LEGACY_MIT) {
        d[0] = (uint8_t)(p >> 8);
        d[1] = (uint8_t)p;
        d[2] = (uint8_t)(v >> 4);
        d[3] = (uint8_t)((v & 0xfu) << 4 | kp_code >> 8);
        d[4] = (uint8_t)kp_code;
        d[5] = (uint8_t)(kd_code >> 4);
        d[6] = (uint8_t)((kd_code & 0xfu) << 4 | t >> 8);
        d[7] = (uint8_t)t;
    } else {
        frame.extended = 1;
        d[0] = (uint8_t)(kp_code >> 4);
        d[1] = (uint8_t)((kp_code & 0xfu) << 4 | kd_code >> 8);
        d[2] = (uint8_t)kd_code;
        d[3] = (uint8_t)(p >> 8);
        d[4] = (uint8_t)p;
        d[5] = (uint8_t)(v >> 4);
        d[6] = (uint8_t)((v & 0xfu) << 4 | t >> 8);
        d[7] = (uint8_t)t;
    }
    *out = frame;
}

static void legacy_special(const qd_ak_joint_config *joint, uint8_t code,
                           qd_ak_can_frame *out)
{
    qd_ak_can_frame frame = {0};
    frame.bus = joint->bus;
    frame.id = joint->motor_id;
    frame.dlc = 8;
    memset(frame.data, 0xff, 7);
    frame.data[7] = code;
    *out = frame;
}

bool qd_ak_motor_encode(const qd_ak_joint_config *joint, float position,
                        float velocity, float kp, float kd, float effort,
                        qd_ak_can_frame *out)
{
    if (joint == NULL || out == NULL || !joint_config_valid(joint))
        return false;
    float q = joint->sign * position + joint->zero;
    float dq = joint->sign * velocity;
    float torque = joint->sign * effort;
    if (!in_span(q, joint->position_max) || !in_span(dq, joint->velocity_max) ||
        !in_gain(kp, joint->kp_max) || !in_gain(kd, joint->kd_max) ||
        !in_span(torque, joint->torque_max))
        return false;
    pack_mit(joint, q, dq, kp, kd, torque, out);
    return true;
}

void qd_ak_motor_init(qd_ak_motor *motor, const qd_ak_motor_config *config,
                      qd_ak_send send, void *send_user)
{
    if (motor == NULL)
        return;
    memset(motor, 0, sizeof(*motor));
    if (config != NULL)
        motor->config = *config;
    motor->send = send;
    motor->send_user = send_user;
    motor->config_valid = send != NULL && qd_ak_config_valid(config);
}

void qd_ak_motor_clock(qd_ak_motor *motor, uint32_t now_ms)
{
    if (motor == NULL)
        return;
    motor->now_ms = now_ms;
    for (unsigned i = 0; i < QD_JOINT_COUNT; ++i) {
        qd_ak_joint_state *state = &motor->joint[i];
        if (state->seen && (uint32_t)(now_ms - state->stamp_ms) > QD_AK_FRESH_MS)
            state->seen = false;
    }
}

bool qd_ak_motor_receive(qd_ak_motor *motor, const qd_ak_can_frame *frame)
{
    if (motor == NULL || frame == NULL)
        return false;
    for (unsigned i = 0; motor->config_valid && frame->dlc == 8u && i < QD_JOINT_COUNT; ++i) {
        const qd_ak_joint_config *joint = &motor->config.joint[i];
        const uint8_t *d = frame->data;
        bool legacy = joint->protocol == QD_AK_LEGACY_MIT;
        if (frame->bus != joint->bus || frame->id != joint->reply_id ||
            frame->extended != (legacy ? 0u : 1u) || (legacy && d[0] != joint->motor_id))
            continue;
        float q, dq, torque;
        int32_t temperature;
        if (legacy) {
            q = from_uint((uint32_t)d[1] << 8 | d[2], -joint->reply_position,
                          joint->reply_position, 16);
            dq = from_uint((uint32_t)d[3] << 4 | d[4] >> 4, -joint->reply_velocity,
                           joint->reply_velocity, 12);
            torque = from_uint((uint32_t)(d[4] & 0xfu) << 8 | d[5], -joint->reply_effort,
                               joint->reply_effort, 12);
            temperature = (int32_t)d[6] - 40;
        } else {
            q = (float)read_i16(d) * joint->reply_position;
            dq = (float)read_i16(d + 2) * joint->reply_velocity;
            torque = (float)read_i16(d + 4) * joint->reply_effort;
            temperature = (int32_t)d[6] - ((d[6] & 0x80u) != 0u ? 256 : 0);
        }
        qd_ak_joint_state *state = &motor->joint[i];
        state->position = joint->sign * (q - joint->zero);
        state->velocity = joint->sign * dq;
        state->effort = joint->sign * torque;
        state->temperature_c = temperature;
        state->error_code = d[7];
        state->stamp_ms = motor->now_ms;
        state->seen = true;
        return true;
    }
    ++motor->rejected_frames;
    return false;
}

static bool transmit(qd_ak_motor *motor, const qd_ak_can_frame *frame)
{
    if (motor->send(motor->send_user, frame))
        return true;
    ++motor->send_failures;
    return false;
}

/* Every joint gets a rest frame even after a failed send. */
static bool rest_all(qd_ak_motor *motor)
{
    if (motor == NULL || !motor->config_valid)
        return false;
    bool ok = true;
    for (unsigned i = 0; i < QD_JOINT_COUNT; ++i) {
        const qd_ak_joint_config *joint = &motor->config.joint[i];
        qd_ak_can_frame frame;
        if (joint->protocol == QD_AK_LEGACY_MIT)
            legacy_special(joint, QD_AK_EXIT, &frame);
        else
            pack_mit(joint, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, &frame);
        if (!transmit(motor, &frame))
            ok = false;
        motor->joint[i].motor_mode = false;
    }
    return ok;
}

bool qd_ak_motor_poll(qd_ak_motor *motor)
{
    return rest_all(motor);
}

static bool all_usable(const qd_ak_motor *motor)
{
    if (motor == NULL || !motor->config_valid)
        return false;
    for (unsigned i = 0; i < QD_JOINT_COUNT; ++i)
        if (!motor->joint[i].seen || motor->joint[i].error_code != 0u)
            return false;
    return true;
}

static bool board_ready(void *user)
{
    return all_usable(user);
}

static bool board_read(void *user, qd_measurement *measured)
{
    const qd_ak_motor *motor = user;
    if (measured == NULL || !all_usable(motor))
        return false;
    for (unsigned i = 0; i < QD_JOINT_COUNT; ++i) {
        measured->position[i] = motor->joint[i].position;
        measured->velocity[i] = motor->joint[i].velocity;
        measured->effort[i] = motor->joint[i].effort;
    }
    return true;
}

/* MIT motors close PD locally; qd_guard_command already checked the total. */
static bool board_apply(void *user, const qd_command *command,
                        const float total_torque[QD_JOINT_COUNT])
{
    qd_ak_motor *motor = user;
    qd_ak_can_frame frames[QD_JOINT_COUNT];
    (void)total_torque;
    if (motor == NULL || command == NULL || !motor->config_valid)
        return false;
    for (unsigned i = 0; i < QD_JOINT_COUNT; ++i)
        if (!qd_ak_motor_encode(&motor->config.joint[i], command->position[i],
                                command->velocity[i], command->kp[i], command->kd[i],
                                command->effort[i], &frames[i]))
            return false;
    for (unsigned i = 0; i < QD_JOINT_COUNT; ++i) {
        const qd_ak_joint_config *joint = &motor->config.joint[i];
        if (joint->protocol == QD_AK_LEGACY_MIT && !motor->joint[i].motor_mode) {
            qd_ak_can_frame enter;
            legacy_special(joint, QD_AK_ENTER, &enter);
            if (!transmit(motor, &enter))
                return false;
            motor->joint[i].motor_mode = true;
        }
        if (!transmit(motor, &frames[i]))
            return false;
    }
    return true;
}

static void board_disarm(void *user)
{
    (void)rest_all(user);
}

qd_board qd_ak_motor_board(qd_ak_motor *motor)
{
    qd_board board = {motor, board_ready, board_read, board_apply, board_disarm};
    return board;
}
