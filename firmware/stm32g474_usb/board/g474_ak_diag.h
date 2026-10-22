#ifndef G474_AK_DIAG_H
#define G474_AK_DIAG_H
#include <stdint.h>

#define G474_AK_DIAG_MAGIC 0x47414b4du /* "GAKM" */
#define G474_AK_DIAG_VERSION 1u
#define G474_AK_UNKNOWN_LOG 16u

/* Frames the adapter did not accept; used to discover reply identifiers. */
typedef struct {
    uint32_t bus, extended, id, dlc, data_lo, data_hi, tick_ms;
} g474_can_log;

typedef struct {
    uint32_t tx_queued, tx_full, tx_errors, rx_frames, rx_lost, bus_off;
    uint32_t last_psr, last_ecr;
} g474_bus_diag;

/* Read over SWD. probe_*: write bus/protocol/id, then increment
 * probe_request; while disarmed the firmware sends ONE rest frame. */
typedef struct {
    uint32_t magic, version, boot_stage, clock_hz;
    uint32_t rx_bytes, rx_overflows, uart_errors, uart_tx_errors;
    uint32_t frames_received, commands_received, commands_accepted, commands_rejected;
    uint32_t feedback_sent, missed_ticks, armed, fault_flags, config_valid;
    uint32_t adapter_rejected, adapter_send_failures, polls, can_ring_overflows;
    uint32_t probe_request, probe_done, probe_bus, probe_protocol, probe_id, probe_sent;
    uint32_t unknown_count;
    g474_bus_diag bus[3];
    g474_can_log unknown[G474_AK_UNKNOWN_LOG];
} g474_ak_diag;

_Static_assert(sizeof(g474_ak_diag) == (28u + 3u * 8u + G474_AK_UNKNOWN_LOG * 7u) * 4u,
               "diagnostic layout changed");
#endif
