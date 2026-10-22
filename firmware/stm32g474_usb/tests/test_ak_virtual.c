#include "qd_ak_virtual.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static unsigned failed;
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: FAIL %s\n", __func__, __LINE__, #x); \
    ++failed; return; \
} } while (0)

static void legacy_mit_zero_matches_fixed_fixture(void)
{
    static const uint8_t command_bytes[8] = {0x7f, 0xff, 0x80, 0, 0, 0, 8, 0};
    static const uint8_t reply_bytes[8] = {1, 0x7f, 0xff, 0x80, 8, 0, 65, 0};
    qd_command command = {0};
    qd_ak_frame frame, reply;
    CHECK(qd_ak_build_command(QD_AK_MIT, &command, 0, &frame));
    CHECK(frame.bus == 1 && frame.arbitration_id == 1);
    CHECK(frame.joint_index == 0 && frame.extended == 0 && frame.dlc == 8);
    CHECK(memcmp(frame.data, command_bytes, 8) == 0);
    CHECK(qd_ak_virtual_reply(QD_AK_MIT, &frame, &reply));
    CHECK(reply.extended == 0 && reply.arbitration_id == 1 && reply.dlc == 8);
    CHECK(memcmp(reply.data, reply_bytes, 8) == 0);
}

static void v3_mit_zero_uses_extended_gain_first_fixture(void)
{
    static const uint8_t bytes[8] = {0, 0, 0, 0x7f, 0xff, 0x80, 8, 0};
    static const uint8_t feedback[8] = {0, 0, 0, 0, 0, 0, 25, 0};
    qd_command command = {0};
    qd_ak_frame frame, reply;
    CHECK(qd_ak_build_command(QD_AK_MIT, &command, 1, &frame));
    CHECK(frame.bus == 2 && frame.joint_index == 1);
    CHECK(frame.extended == 1 && frame.arbitration_id == 0x801 && frame.dlc == 8);
    CHECK(memcmp(frame.data, bytes, 8) == 0);
    CHECK(qd_ak_virtual_reply(QD_AK_MIT, &frame, &reply));
    CHECK(reply.extended == 1 && reply.arbitration_id == 0x2901 && reply.dlc == 8);
    CHECK(memcmp(reply.data, feedback, 8) == 0);
}

static void servo_ninety_degrees_matches_big_endian_signed_fixtures(void)
{
    static const uint8_t positive[8] = {0, 0x0d, 0xbb, 0xa0, 0, 0, 0, 0};
    static const uint8_t negative[8] = {0xff, 0xf2, 0x44, 0x60, 0, 0, 0, 0};
    static const uint8_t feedback[8] = {3, 0x84, 0, 0, 0, 0, 25, 0};
    qd_command command = {0};
    qd_ak_frame frame, reply;
    command.position[0] = 1.57079632679489661923f;
    CHECK(qd_ak_build_command(QD_AK_SERVO, &command, 0, &frame));
    CHECK(frame.arbitration_id == 0x401 && frame.extended == 1 && frame.dlc == 4);
    CHECK(memcmp(frame.data, positive, 8) == 0);
    CHECK(qd_ak_virtual_reply(QD_AK_SERVO, &frame, &reply));
    CHECK(reply.arbitration_id == 0x2901 && reply.extended == 1 && reply.dlc == 8);
    CHECK(memcmp(reply.data, feedback, 8) == 0);
    command.position[0] = -1.57079632679489661923f;
    CHECK(qd_ak_build_command(QD_AK_SERVO, &command, 0, &frame));
    CHECK(memcmp(frame.data, negative, 8) == 0);
    CHECK(qd_ak_virtual_reply(QD_AK_SERVO, &frame, &reply));
    CHECK(reply.data[0] == 0xfc && reply.data[1] == 0x7c);
}

static void all_routes_round_trip_both_modes_and_only_update_one_joint(void)
{
    static const uint32_t routes[12][2] = {
        {1, 1}, {2, 1}, {3, 1}, {1, 2}, {2, 2}, {3, 2},
        {1, 3}, {2, 3}, {3, 3}, {1, 4}, {2, 4}, {3, 4}
    };
    static const float positions[12] = {
        -12.5f, -12.56f, 12.56f, 12.5f, -1.5f, 1.5f,
        -0.3f, 0.3f, 0.0f, 2.0f, -2.0f, 0.7f
    };
    qd_command command = {0};
    for (unsigned i = 0; i < QD_JOINT_COUNT; ++i)
        command.position[i] = positions[i];
    for (unsigned mode = QD_AK_MIT; mode <= QD_AK_SERVO; ++mode) {
        for (unsigned joint = 0; joint < QD_JOINT_COUNT; ++joint) {
            qd_ak_frame frame, reply;
            qd_measurement measured, before;
            for (unsigned i = 0; i < QD_JOINT_COUNT; ++i) {
                measured.position[i] = 100.0f + (float)i;
                measured.velocity[i] = 200.0f + (float)i;
                measured.effort[i] = 300.0f + (float)i;
            }
            before = measured;
            CHECK(qd_ak_build_command(mode, &command, joint, &frame));
            CHECK(frame.bus == routes[joint][0] && frame.joint_index == joint);
            CHECK((frame.arbitration_id & 0xffu) == routes[joint][1]);
            CHECK(qd_ak_virtual_reply(mode, &frame, &reply));
            CHECK(reply.bus == frame.bus && reply.joint_index == joint);
            CHECK(qd_ak_decode_reply(mode, &reply, &measured));
            CHECK(fabsf(measured.position[joint] - positions[joint]) <= 0.0014f);
            CHECK(measured.velocity[joint] == 0.0f && measured.effort[joint] == 0.0f);
            for (unsigned i = 0; i < QD_JOINT_COUNT; ++i) {
                if (i == joint) continue;
                CHECK(measured.position[i] == before.position[i]);
                CHECK(measured.velocity[i] == before.velocity[i]);
                CHECK(measured.effort[i] == before.effort[i]);
            }
        }
    }
}

static void invalid_commands_preserve_output_and_validate_all_twelve_joints(void)
{
    static const float bad_fields[] = {1.0f, -1.0f, FLT_MIN, NAN, INFINITY};
    qd_ak_frame output;
    unsigned char before[sizeof(output)];
    memset(&output, 0xa5, sizeof(output));
    memcpy(before, &output, sizeof(output));
    for (unsigned mode = QD_AK_MIT; mode <= QD_AK_SERVO; ++mode) {
        for (unsigned joint = 0; joint < QD_JOINT_COUNT; ++joint) {
            const float limit = joint % 3u == 0u ? 12.5f : 12.56f;
            const float invalid[] = {
                NAN, INFINITY, -INFINITY,
                nextafterf(limit, INFINITY), nextafterf(-limit, -INFINITY)
            };
            for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
                qd_command command = {0};
                command.position[joint] = invalid[i];
                CHECK(!qd_ak_build_command(mode, &command, 0, &output));
                CHECK(memcmp(&output, before, sizeof(output)) == 0);
            }
            for (unsigned field = 0; field < 4; ++field) {
                for (size_t i = 0; i < sizeof(bad_fields) / sizeof(bad_fields[0]); ++i) {
                    qd_command command = {0};
                    float *fields[] = {command.velocity, command.kp, command.kd, command.effort};
                    fields[field][joint] = bad_fields[i];
                    CHECK(!qd_ak_build_command(mode, &command, 0, &output));
                    CHECK(memcmp(&output, before, sizeof(output)) == 0);
                }
            }
        }
        qd_command enabled = {0};
        enabled.enabled = 1;
        CHECK(!qd_ak_build_command(mode, &enabled, 0, &output));
        CHECK(memcmp(&output, before, sizeof(output)) == 0);
    }
}

static void malformed_frames_are_rejected_without_modifying_outputs(void)
{
    qd_command command = {0};
    qd_ak_frame output;
    qd_measurement measured;
    unsigned char frame_before[sizeof(output)], measured_before[sizeof(measured)];
    memset(&output, 0x5a, sizeof(output));
    memset(&measured, 0x3c, sizeof(measured));
    memcpy(frame_before, &output, sizeof(output));
    memcpy(measured_before, &measured, sizeof(measured));
    for (unsigned mode = QD_AK_MIT; mode <= QD_AK_SERVO; ++mode) {
        for (unsigned joint = 0; joint < 3; ++joint) {
            qd_ak_frame good, good_reply;
            CHECK(qd_ak_build_command(mode, &command, joint, &good));
            CHECK(qd_ak_virtual_reply(mode, &good, &good_reply));
            for (unsigned field = 0; field < 6; ++field) {
                qd_ak_frame bad = good;
                switch (field) {
                case 0: bad.bus = 0; break;
                case 1: bad.arbitration_id ^= 0x10u; break;
                case 2: bad.joint_index = 11; break;
                case 3: bad.extended ^= 1u; break;
                case 4: bad.dlc = 7; break;
                default: bad.bus = 4; break;
                }
                CHECK(!qd_ak_virtual_reply(mode, &bad, &output));
                CHECK(memcmp(&output, frame_before, sizeof(output)) == 0);
            }
            for (unsigned field = 0; field < 7; ++field) {
                qd_ak_frame bad = good_reply;
                switch (field) {
                case 0: bad.bus = 0; break;
                case 1: bad.arbitration_id ^= 0x10u; break;
                case 2: bad.joint_index = 11; break;
                case 3: bad.extended ^= 1u; break;
                case 4: bad.dlc = 7; break;
                case 5: bad.data[7] = 1; break;
                default:
                    if (bad.extended == 0) bad.data[0] = 2;
                    else bad.arbitration_id = 0x2c01;
                    break;
                }
                CHECK(!qd_ak_decode_reply(mode, &bad, &measured));
                CHECK(memcmp(&measured, measured_before, sizeof(measured)) == 0);
            }
        }
    }
}

static void null_invalid_mode_and_invalid_joint_are_transactional(void)
{
    qd_command command = {0};
    qd_ak_frame output, valid;
    qd_measurement measured;
    unsigned char before[sizeof(output)], measured_before[sizeof(measured)];
    memset(&output, 0xc3, sizeof(output));
    memset(&measured, 0x66, sizeof(measured));
    memcpy(before, &output, sizeof(output));
    memcpy(measured_before, &measured, sizeof(measured));
    CHECK(qd_ak_build_command(QD_AK_MIT, &command, 0, &valid));
    CHECK(!qd_ak_build_command(2, &command, 0, &output));
    CHECK(!qd_ak_build_command(QD_AK_MIT, NULL, 0, &output));
    CHECK(!qd_ak_build_command(QD_AK_MIT, &command, 12, &output));
    CHECK(!qd_ak_build_command(QD_AK_MIT, &command, 0, NULL));
    CHECK(!qd_ak_virtual_reply(2, &valid, &output));
    CHECK(!qd_ak_virtual_reply(QD_AK_MIT, NULL, &output));
    CHECK(!qd_ak_virtual_reply(QD_AK_MIT, &valid, NULL));
    CHECK(!qd_ak_decode_reply(2, &valid, &measured));
    CHECK(!qd_ak_decode_reply(QD_AK_MIT, NULL, &measured));
    CHECK(!qd_ak_decode_reply(QD_AK_MIT, &valid, NULL));
    CHECK(memcmp(&output, before, sizeof(output)) == 0);
    CHECK(memcmp(&measured, measured_before, sizeof(measured)) == 0);
}

int main(void)
{
    static const struct { const char *name; void (*run)(void); } tests[] = {
        {"legacy_mit_zero_matches_fixed_fixture", legacy_mit_zero_matches_fixed_fixture},
        {"v3_mit_zero_uses_extended_gain_first_fixture", v3_mit_zero_uses_extended_gain_first_fixture},
        {"servo_ninety_degrees_matches_big_endian_signed_fixtures", servo_ninety_degrees_matches_big_endian_signed_fixtures},
        {"all_routes_round_trip_both_modes_and_only_update_one_joint", all_routes_round_trip_both_modes_and_only_update_one_joint},
        {"invalid_commands_preserve_output_and_validate_all_twelve_joints", invalid_commands_preserve_output_and_validate_all_twelve_joints},
        {"malformed_frames_are_rejected_without_modifying_outputs", malformed_frames_are_rejected_without_modifying_outputs},
        {"null_invalid_mode_and_invalid_joint_are_transactional", null_invalid_mode_and_invalid_joint_are_transactional}
    };
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
        unsigned before = failed;
        tests[i].run();
        printf("%s %s\n", before == failed ? "PASS" : "FAIL", tests[i].name);
    }
    printf("%u failure(s)\n", failed);
    return failed ? 1 : 0;
}
