#ifndef QD_G474_PORT_H
#define QD_G474_PORT_H
#include "qd_protocol.h"

/* Call after CubeMX peripheral init; poll continuously in the main loop. */
void qd_g474_start(void);
void qd_g474_poll(void);

/* Replace the disabled definitions after the actual CAN and calibration
 * contracts are measured. No motor IDs, directions or zeros are assumed. */
bool qd_g474_board_ready(void);
bool qd_g474_board_read(qd_measurement *measured);
bool qd_g474_board_apply(const qd_command *command,const float total[QD_JOINT_COUNT]);
void qd_g474_board_disarm(void);
/* Must revoke hardware enable synchronously, without waiting for CAN/UART.
 * Used for UART overflow/error and watchdog while foreground is delayed. */
void qd_g474_board_disarm_from_isr(void);
#endif
