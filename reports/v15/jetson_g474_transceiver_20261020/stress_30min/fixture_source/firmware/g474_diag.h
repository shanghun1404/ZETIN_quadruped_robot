#ifndef G474_DIAG_H
#define G474_DIAG_H
#include <stdint.h>
#define CAPTURE_ROUNDS 8u
#define CAPTURE_SAMPLES 800u
typedef struct {
 uint32_t queued,completed,received,rx_mismatch,timeouts,tx_cancelled;
 uint32_t last_psr,last_ecr,txbrp,txbto,txbcf,cccr,nbtp,test_reg;
} can_bus_diag;
typedef struct {
 uint32_t queued,completed,received,last_id,expected_lo,expected_hi,received_lo,received_hi;
} can_joint_diag;
typedef struct {
 uint32_t magic,version,clock_hz,boot_stage,requested_mode,current_mode,mode_changes;
 uint32_t rx_bytes,rx_overflows,uart_errors,tx_errors,frames_received;
 uint32_t commands_received,commands_accepted,commands_routed,route_rejected;
 uint32_t feedback_sent,missed_ticks,apply_calls,ready_checks,session;
 uint32_t capture_requested,capture_done,capture_samples_per_round,capture_rounds;
 uint32_t can_errors,gpio_moder,gpio_afrl,gpio_afrh,pwr_cr3,capture_sequence;
 uint32_t requested_profile,current_profile,virtual_replies,virtual_decoded,virtual_errors,parser_rejected,max_route_cycles;
 uint32_t max_parser_cycles,max_rx_queued,max_loop_cycles;
 uint32_t gpioa_moder,gpioa_afrl,gpioa_afrh;
 can_bus_diag bus[3];
 can_joint_diag joint[12];
} can_diag;
typedef struct { uint32_t cycle,pins; } pin_sample;
_Static_assert(sizeof(can_diag)==728,"diagnostic layout changed");
#endif
