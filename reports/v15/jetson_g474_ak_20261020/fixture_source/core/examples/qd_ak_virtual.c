#include "qd_ak_virtual.h"

#include <math.h>

#define QD_AK_PI 3.14159265358979323846

static bool valid_mode(unsigned mode)
{
    return mode == QD_AK_MIT || mode == QD_AK_SERVO;
}

static double position_limit(unsigned joint)
{
    return joint % 3u == 0u ? 12.5 : 12.56;
}

static bool valid_route(const qd_ak_frame *frame)
{
    return frame != NULL && frame->joint_index < QD_JOINT_COUNT &&
           frame->bus == frame->joint_index % 3u + 1u;
}

static uint32_t motor_id(const qd_ak_frame *frame)
{
    return frame->joint_index / 3u + 1u;
}

static bool legacy_mit(unsigned mode, const qd_ak_frame *frame)
{
    return mode == QD_AK_MIT && frame->bus == 1u;
}

static uint16_t read_u16(const uint8_t *data)
{
    return (uint16_t)((uint16_t)data[0] << 8 | data[1]);
}

static int32_t read_i16(const uint8_t *data)
{
    uint32_t raw = read_u16(data);
    return (int32_t)raw - ((raw & 0x8000u) != 0u ? 65536 : 0);
}

static int64_t read_i32(const uint8_t *data)
{
    uint32_t raw = (uint32_t)data[0] << 24 | (uint32_t)data[1] << 16 |
                   (uint32_t)data[2] << 8 | data[3];
    return (int64_t)raw - ((raw & 0x80000000u) != 0u ? INT64_C(4294967296) : 0);
}

static uint16_t encode_mit_position(double position, double limit)
{
    double scaled = (position + limit) * 65535.0 / (2.0 * limit);
    return scaled <= 0.0 ? 0u : scaled >= 65535.0 ? 65535u : (uint16_t)scaled;
}

static double decode_mit_position(uint16_t position, unsigned joint)
{
    double limit = position_limit(joint);
    return (double)position * (2.0 * limit) / 65535.0 - limit;
}

bool qd_ak_build_command(unsigned mode, const qd_command *command,
                         unsigned joint, qd_ak_frame *out)
{
    if (!valid_mode(mode) || command == NULL || out == NULL ||
        joint >= QD_JOINT_COUNT || command->enabled != 0u)
        return false;
    for (unsigned i = 0; i < QD_JOINT_COUNT; ++i) {
        float limit = (float)position_limit(i);
        if (!isfinite(command->position[i]) || command->position[i] < -limit ||
            command->position[i] > limit || command->velocity[i] != 0.0f ||
            command->kp[i] != 0.0f || command->kd[i] != 0.0f ||
            command->effort[i] != 0.0f)
            return false;
    }

    qd_ak_frame frame = {0};
    frame.bus = joint % 3u + 1u;
    frame.joint_index = joint;
    frame.arbitration_id = motor_id(&frame);
    if (mode == QD_AK_SERVO) {
        uint32_t raw = (uint32_t)(int32_t)lround(
            (double)command->position[joint] * 180.0 / QD_AK_PI * 10000.0);
        frame.extended = 1;
        frame.arbitration_id |= 0x400u;
        frame.dlc = 4;
        frame.data[0] = (uint8_t)(raw >> 24);
        frame.data[1] = (uint8_t)(raw >> 16);
        frame.data[2] = (uint8_t)(raw >> 8);
        frame.data[3] = (uint8_t)raw;
    } else {
        uint16_t raw = encode_mit_position(command->position[joint], position_limit(joint));
        frame.dlc = 8;
        if (frame.bus == 1u) {
            frame.data[0] = (uint8_t)(raw >> 8);
            frame.data[1] = (uint8_t)raw;
            frame.data[2] = 0x80;
            frame.data[6] = 0x08;
        } else {
            frame.extended = 1;
            frame.arbitration_id |= 0x800u;
            frame.data[3] = (uint8_t)(raw >> 8);
            frame.data[4] = (uint8_t)raw;
            frame.data[5] = 0x80;
            frame.data[6] = 0x08;
        }
    }
    *out = frame;
    return true;
}

bool qd_ak_virtual_reply(unsigned mode, const qd_ak_frame *received,
                         qd_ak_frame *out)
{
    if (!valid_mode(mode) || !valid_route(received) || out == NULL)
        return false;
    bool legacy = legacy_mit(mode, received);
    uint32_t expected_id = motor_id(received) |
        (mode == QD_AK_SERVO ? 0x400u : legacy ? 0u : 0x800u);
    if (received->arbitration_id != expected_id ||
        received->extended != (legacy ? 0u : 1u) ||
        received->dlc != (mode == QD_AK_SERVO ? 4u : 8u))
        return false;

    const uint8_t *data = received->data;
    uint16_t mit_position = 0;
    double position;
    if (mode == QD_AK_SERVO) {
        int64_t raw = read_i32(data);
        long maximum = lround((double)(float)position_limit(received->joint_index) *
                              180.0 / QD_AK_PI * 10000.0);
        if (raw < -(int64_t)maximum || raw > maximum)
            return false;
        position = (double)raw / 10000.0 * QD_AK_PI / 180.0;
    } else if (legacy) {
        if (data[2] != 0x80 || data[3] != 0 || data[4] != 0 ||
            data[5] != 0 || data[6] != 0x08 || data[7] != 0)
            return false;
        mit_position = read_u16(data);
        position = decode_mit_position(mit_position, received->joint_index);
    } else {
        if (data[0] != 0 || data[1] != 0 || data[2] != 0 ||
            data[5] != 0x80 || data[6] != 0x08 || data[7] != 0)
            return false;
        mit_position = read_u16(data + 3);
        position = decode_mit_position(mit_position, received->joint_index);
    }

    /* Explicit kinematic emulation: target angle becomes virtual angle.
     * No PD dynamics, motor enable, physical measurements, or Iq calibration. */
    qd_ak_frame reply = {0};
    reply.bus = received->bus;
    reply.joint_index = received->joint_index;
    reply.dlc = 8;
    if (legacy) {
        reply.arbitration_id = motor_id(received);
        reply.data[0] = (uint8_t)motor_id(received);
        reply.data[1] = (uint8_t)(mit_position >> 8);
        reply.data[2] = (uint8_t)mit_position;
        reply.data[3] = 0x80;
        reply.data[4] = 0x08;
        reply.data[6] = 65; /* Legacy MIT temperature has a +40 offset. */
    } else {
        uint16_t raw = (uint16_t)(int16_t)lround(position * 180.0 / QD_AK_PI * 10.0);
        reply.arbitration_id = 0x2900u | motor_id(received);
        reply.extended = 1;
        reply.data[0] = (uint8_t)(raw >> 8);
        reply.data[1] = (uint8_t)raw;
        reply.data[6] = 25;
    }
    *out = reply;
    return true;
}

bool qd_ak_decode_reply(unsigned mode, const qd_ak_frame *received,
                        qd_measurement *out)
{
    if (!valid_mode(mode) || !valid_route(received) || out == NULL)
        return false;
    bool legacy = legacy_mit(mode, received);
    if (received->arbitration_id != (motor_id(received) | (legacy ? 0u : 0x2900u)) ||
        received->extended != (legacy ? 0u : 1u) || received->dlc != 8u ||
        received->data[7] != 0u)
        return false;

    const uint8_t *data = received->data;
    double position;
    if (legacy) {
        if (data[0] != motor_id(received) || data[3] != 0x80 ||
            data[4] != 0x08 || data[5] != 0)
            return false;
        position = decode_mit_position(read_u16(data + 1), received->joint_index);
    } else {
        int32_t raw = read_i16(data);
        long maximum = lround((double)(float)position_limit(received->joint_index) *
                              180.0 / QD_AK_PI * 10.0);
        if (raw < -maximum || raw > maximum ||
            data[2] != 0 || data[3] != 0 || data[4] != 0 || data[5] != 0)
            return false;
        position = (double)raw * 0.1 * QD_AK_PI / 180.0;
    }
    out->position[received->joint_index] = (float)position;
    out->velocity[received->joint_index] = 0.0f;
    out->effort[received->joint_index] = 0.0f;
    return true;
}
