#include "qd_can_diagnostic.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static unsigned failed;
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: FAIL %s\n", __func__, __LINE__, #x); \
    ++failed; return; \
} } while (0)

static void canonical_joints_have_distinct_bus_and_motor_ids(void)
{
    static const uint32_t routes[QD_JOINT_COUNT][2] = {
        {1, 1}, {2, 1}, {3, 1}, /* FL HAA/HFE/KFE */
        {1, 2}, {2, 2}, {3, 2}, /* FR HAA/HFE/KFE */
        {1, 3}, {2, 3}, {3, 3}, /* RL HAA/HFE/KFE */
        {1, 4}, {2, 4}, {3, 4}  /* RR HAA/HFE/KFE */
    };
    qd_command command = {0};
    qd_can_diagnostic_frame frames[QD_JOINT_COUNT];

    CHECK(qd_can_build_diagnostic(&command, frames));
    for (size_t i = 0; i < QD_JOINT_COUNT; ++i) {
        CHECK(frames[i].bus == routes[i][0]);
        CHECK(frames[i].arbitration_id == routes[i][1]);
        CHECK(frames[i].joint_index == i);
        for (size_t j = 0; j < i; ++j) {
            CHECK(frames[i].bus != frames[j].bus ||
                  frames[i].arbitration_id != frames[j].arbitration_id);
        }
    }
}

static void position_tags_use_floor_and_fixed_zero_gain_fields(void)
{
    static const struct {
        float position;
        uint8_t high;
        uint8_t low;
    } cases[QD_JOINT_COUNT] = {
        {-12.5f, 0x00, 0x00}, {-10.0f, 0x19, 0x99},
        {-7.5f, 0x33, 0x33}, {-5.0f, 0x4c, 0xcc},
        {-2.5f, 0x66, 0x66}, {0.0f, 0x7f, 0xff},
        {2.5f, 0x99, 0x99}, {5.0f, 0xb3, 0x32},
        {7.5f, 0xcc, 0xcc}, {10.0f, 0xe6, 0x65},
        {12.5f, 0xff, 0xff}, {-0.0f, 0x7f, 0xff}
    };
    static const uint8_t tail[6] = {0x80, 0, 0, 0, 0x08, 0};
    qd_command command = {0};
    qd_can_diagnostic_frame frames[QD_JOINT_COUNT];

    for (size_t i = 0; i < QD_JOINT_COUNT; ++i)
        command.position[i] = cases[i].position;
    CHECK(qd_can_build_diagnostic(&command, frames));
    for (size_t i = 0; i < QD_JOINT_COUNT; ++i) {
        CHECK(frames[i].data[0] == cases[i].high);
        CHECK(frames[i].data[1] == cases[i].low);
        CHECK(memcmp(frames[i].data + 2, tail, sizeof(tail)) == 0);
    }
}

static void signed_zero_control_fields_are_accepted(void)
{
    qd_command command = {0};
    qd_can_diagnostic_frame frames[QD_JOINT_COUNT];

    for (size_t i = 0; i < QD_JOINT_COUNT; ++i) {
        command.velocity[i] = -0.0f;
        command.kp[i] = -0.0f;
        command.kd[i] = -0.0f;
        command.effort[i] = -0.0f;
    }
    CHECK(qd_can_build_diagnostic(&command, frames));
}

static void position_just_below_upper_bound_does_not_round_up(void)
{
    qd_command command = {0};
    qd_can_diagnostic_frame frames[QD_JOINT_COUNT];

    command.position[0] = nextafterf(12.5f, -INFINITY);
    CHECK(qd_can_build_diagnostic(&command, frames));
    CHECK(frames[0].data[0] == 0xff);
    CHECK(frames[0].data[1] == 0xfe);
}

static void nonzero_enabled_is_rejected_without_output_changes(void)
{
    static const uint32_t values[] = {1, 2, UINT32_MAX};
    qd_command command = {0};
    qd_can_diagnostic_frame frames[QD_JOINT_COUNT];
    unsigned char before[sizeof(frames)];

    memset(frames, 0xa5, sizeof(frames));
    memcpy(before, frames, sizeof(frames));
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        command.enabled = values[i];
        CHECK(!qd_can_build_diagnostic(&command, frames));
        CHECK(memcmp(frames, before, sizeof(frames)) == 0);
    }
}

static void invalid_positions_leave_every_output_byte_unchanged(void)
{
    const float values[] = {
        NAN, INFINITY, -INFINITY,
        nextafterf(-12.5f, -INFINITY), nextafterf(12.5f, INFINITY)
    };
    qd_can_diagnostic_frame frames[QD_JOINT_COUNT];
    unsigned char before[sizeof(frames)];

    memset(frames, 0x5a, sizeof(frames));
    memcpy(before, frames, sizeof(frames));
    for (size_t joint = 0; joint < QD_JOINT_COUNT; ++joint) {
        for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
            qd_command command = {0};
            command.position[joint] = values[i];
            CHECK(!qd_can_build_diagnostic(&command, frames));
            CHECK(memcmp(frames, before, sizeof(frames)) == 0);
        }
    }
}

static void nonzero_or_nonfinite_control_fields_are_rejected_atomically(void)
{
    static const float values[] = {
        1.0f, -1.0f, FLT_MIN, -FLT_MIN, NAN, INFINITY, -INFINITY
    };
    qd_can_diagnostic_frame frames[QD_JOINT_COUNT];
    unsigned char before[sizeof(frames)];

    memset(frames, 0x3c, sizeof(frames));
    memcpy(before, frames, sizeof(frames));
    for (size_t joint = 0; joint < QD_JOINT_COUNT; ++joint) {
        for (size_t field = 0; field < 4; ++field) {
            for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
                qd_command command = {0};
                float *fields[] = {
                    command.velocity, command.kp, command.kd, command.effort
                };
                fields[field][joint] = values[i];
                CHECK(!qd_can_build_diagnostic(&command, frames));
                CHECK(memcmp(frames, before, sizeof(frames)) == 0);
            }
        }
    }
}

static void null_arguments_are_rejected_without_output_changes(void)
{
    qd_command command = {0};
    qd_can_diagnostic_frame frames[QD_JOINT_COUNT];
    unsigned char before[sizeof(frames)];

    memset(frames, 0xc3, sizeof(frames));
    memcpy(before, frames, sizeof(frames));
    CHECK(!qd_can_build_diagnostic(NULL, frames));
    CHECK(memcmp(frames, before, sizeof(frames)) == 0);
    CHECK(!qd_can_build_diagnostic(&command, NULL));
    CHECK(!qd_can_build_diagnostic(NULL, NULL));
}

int main(void)
{
    static const struct {
        const char *name;
        void (*run)(void);
    } tests[] = {
        {"canonical_joints_have_distinct_bus_and_motor_ids",
         canonical_joints_have_distinct_bus_and_motor_ids},
        {"position_tags_use_floor_and_fixed_zero_gain_fields",
         position_tags_use_floor_and_fixed_zero_gain_fields},
        {"signed_zero_control_fields_are_accepted",
         signed_zero_control_fields_are_accepted},
        {"position_just_below_upper_bound_does_not_round_up",
         position_just_below_upper_bound_does_not_round_up},
        {"nonzero_enabled_is_rejected_without_output_changes",
         nonzero_enabled_is_rejected_without_output_changes},
        {"invalid_positions_leave_every_output_byte_unchanged",
         invalid_positions_leave_every_output_byte_unchanged},
        {"nonzero_or_nonfinite_control_fields_are_rejected_atomically",
         nonzero_or_nonfinite_control_fields_are_rejected_atomically},
        {"null_arguments_are_rejected_without_output_changes",
         null_arguments_are_rejected_without_output_changes}
    };

    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
        unsigned before = failed;
        tests[i].run();
        printf("%s %s\n", before == failed ? "PASS" : "FAIL", tests[i].name);
    }
    printf("%u failure(s)\n", failed);
    return failed ? 1 : 0;
}
