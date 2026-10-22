#include "qd_ak_motor.h"

/* UNVERIFIED: every field is zero, so the adapter never transmits and
 * never becomes ready. Fill all twelve joints from measurements on the
 * installed motors (see README "AK MIT 실제 모터 어댑터"), then set
 * verified = QD_AK_CONFIG_VERIFIED. */
const qd_ak_motor_config g474_ak_config = {0};
