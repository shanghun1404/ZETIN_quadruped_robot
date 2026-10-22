#include "qd_g474_port.h"

/* This file is intentionally unable to enable physical motors. */
bool qd_g474_board_ready(void) { return false; }
bool qd_g474_board_read(qd_measurement *measured) { (void)measured; return false; }
bool qd_g474_board_apply(const qd_command *command,const float total[QD_JOINT_COUNT]) {
    (void)command; (void)total; return false;
}
void qd_g474_board_disarm(void) {
    /* No power/enable GPIO is known. Implement the actual safe disable path
     * before changing ready(). This default never enables hardware. */
}
void qd_g474_board_disarm_from_isr(void) {
    /* The completed board port must implement a bounded IRQ-safe disable. */
}
