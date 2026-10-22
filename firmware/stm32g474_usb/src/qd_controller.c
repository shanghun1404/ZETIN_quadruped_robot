#include "qd_protocol.h"
#include <float.h>
#include <math.h>
#include <string.h>

static bool measured_valid(const qd_measurement *measured) {
    for (unsigned i=0; i<QD_JOINT_COUNT; ++i)
        if (!isfinite(measured->position[i]) || !isfinite(measured->velocity[i]) ||
            !isfinite(measured->effort[i])) return false;
    return true;
}
static float clamp(float value,float low,float high) {
    return fminf(fmaxf(value,low),high);
}
static double float_spacing(float value) {
    float magnitude=fabsf(value);
    return (double)nextafterf(magnitude,INFINITY)-(double)magnitude;
}
static double torque_roundoff(const qd_command *c,const qd_measurement *m,
                              unsigned i,float cap) {
    /* Half an outward binary32 spacing bounds each value's wire rounding.
     * Include gain rounding and reconstruction at the endpoint. */
    double allowance=4.0*FLT_EPSILON*cap+0.5*float_spacing(c->effort[i]);
    if (c->kp[i]>0.0f)
        allowance += 0.5*c->kp[i]*(float_spacing(c->position[i])+float_spacing(m->position[i])) +
                     0.5*float_spacing(c->kp[i])*fabs((double)c->position[i]-m->position[i]);
    if (c->kd[i]>0.0f)
        allowance += 0.5*c->kd[i]*(float_spacing(c->velocity[i])+float_spacing(m->velocity[i])) +
                     0.5*float_spacing(c->kd[i])*fabs((double)c->velocity[i]-m->velocity[i]);
    return allowance;
}
bool qd_guard_command(const qd_command *command,const qd_measurement *measured,
                      float total_torque[QD_JOINT_COUNT]) {
    uint8_t checked[QD_COMMAND_SIZE];
    if (command==NULL || measured==NULL || total_torque==NULL ||
        !qd_pack_command(command,checked) || !measured_valid(measured)) return false;
    float pending[QD_JOINT_COUNT];
    for (unsigned i=0; i<QD_JOINT_COUNT; ++i) {
        double torque=(double)command->kp[i]*((double)command->position[i]-measured->position[i]) +
                      (double)command->kd[i]*((double)command->velocity[i]-measured->velocity[i]) + command->effort[i];
        float cap=(i%3u==0u)?24.8f:48.0f;
        float upper=0.99f*cap*clamp(1.0f-measured->velocity[i]/50.0f,0.0f,1.0f);
        float lower=0.99f*cap*clamp(-1.0f-measured->velocity[i]/50.0f,-1.0f,0.0f);
        double roundoff=torque_roundoff(command,measured,i,cap);
        float raw_upper=cap*clamp(1.0f-measured->velocity[i]/50.0f,0.0f,1.0f);
        float raw_lower=cap*clamp(-1.0f-measured->velocity[i]/50.0f,-1.0f,0.0f);
        /* The allowance can consume only the remaining 1% curve margin.
         * In particular it is zero at a zero acceleration boundary. */
        double upper_slack=upper==0.0f?0.0:fmin(roundoff,fmax(0.0,(double)raw_upper-upper));
        double lower_slack=lower==0.0f?0.0:fmin(roundoff,fmax(0.0,(double)lower-raw_lower));
        pending[i]=(float)torque;
        if (!isfinite(torque) || !isfinite(pending[i]) ||
            pending[i]<lower-lower_slack || pending[i]>upper+upper_slack) return false;
    }
    memcpy(total_torque,pending,sizeof(pending)); return true;
}
static void disarm(qd_controller *controller) {
    controller->armed=false;
    memset(controller->total_torque,0,sizeof(controller->total_torque));
    if (controller->board.disarm!=NULL) controller->board.disarm(controller->board.user);
}
void qd_controller_fault(qd_controller *controller,uint32_t fault_flags) {
    controller->fault_flags |= fault_flags; disarm(controller);
}
void qd_controller_init(qd_controller *controller,const qd_board *board) {
    memset(controller,0,sizeof(*controller));
    if (board!=NULL) controller->board=*board;
    disarm(controller);
}
static bool ready(qd_controller *controller) {
    qd_board *b=&controller->board;
    return b->ready!=NULL && b->read!=NULL && b->apply!=NULL && b->disarm!=NULL && b->ready(b->user);
}
static bool read_actual(qd_controller *controller) {
    qd_measurement measured;
    controller->measurement_valid=controller->board.read!=NULL &&
        controller->board.read(controller->board.user,&measured) && measured_valid(&measured);
    if (controller->measurement_valid) controller->measured=measured;
    return controller->measurement_valid;
}
static void supervise(qd_controller *controller,uint32_t now_ms,bool reapply) {
    if (!controller->armed) return;
    if ((uint32_t)(now_ms-controller->last_command_ms)>=QD_WATCHDOG_MS) {
        qd_controller_fault(controller,QD_FAULT_WATCHDOG); return;
    }
    if (!ready(controller)) { qd_controller_fault(controller,QD_FAULT_BOARD); return; }
    if (!read_actual(controller)) { qd_controller_fault(controller,QD_FAULT_STATE); return; }
    if (!qd_guard_command(&controller->command,&controller->measured,controller->total_torque)) {
        qd_controller_fault(controller,QD_FAULT_GUARD); return;
    }
    if (reapply && !controller->board.apply(controller->board.user,&controller->command,controller->total_torque))
        qd_controller_fault(controller,QD_FAULT_BOARD);
}
void qd_controller_tick(qd_controller *controller,uint32_t now_ms) { supervise(controller,now_ms,true); }
enum qd_result qd_controller_on_frame(qd_controller *controller,const qd_frame *frame,
                                     uint32_t now_ms,qd_frame *reply) {
    supervise(controller,now_ms,false);
    if (frame==NULL || frame->session==0) return QD_IGNORED;
    if (frame->type==QD_HELLO && frame->payload_length==0) {
        if (frame->session!=controller->session) {
            disarm(controller); controller->session=frame->session; controller->have_command=false;
            controller->acknowledged_sequence=0; controller->tx_sequence=0; controller->last_command_ms=0;
            controller->fault_flags=0; memset(&controller->command,0,sizeof(controller->command));
        }
        if (reply!=NULL) {
            memset(reply,0,sizeof(*reply)); reply->type=QD_HELLO_ACK; reply->session=frame->session;
            reply->sequence=frame->sequence; reply->stamp_ns=frame->stamp_ns;
            reply->payload_length=4; reply->payload[0]=1;
        }
        return QD_REPLY;
    }
    if (controller->session==0 || frame->session!=controller->session) return QD_IGNORED;
    if (frame->type==QD_STOP && frame->payload_length==0) { disarm(controller); return QD_ACCEPTED; }
    if (frame->type!=QD_COMMAND) return QD_IGNORED;
    if (controller->have_command && !qd_sequence_newer(frame->sequence,controller->acknowledged_sequence)) return QD_IGNORED;
    qd_command command;
    if (!qd_unpack_command(frame->payload,frame->payload_length,&command)) {
        qd_controller_fault(controller,QD_FAULT_COMMAND); return QD_REJECTED;
    }
    if (controller->fault_flags!=0) { disarm(controller); return QD_REJECTED; }
    if (command.enabled) {
        if (!ready(controller)) { qd_controller_fault(controller,QD_FAULT_BOARD); return QD_REJECTED; }
        if (!read_actual(controller)) { qd_controller_fault(controller,QD_FAULT_STATE); return QD_REJECTED; }
        if (!qd_guard_command(&command,&controller->measured,controller->total_torque)) {
            qd_controller_fault(controller,QD_FAULT_GUARD); return QD_REJECTED;
        }
        if (!controller->board.apply(controller->board.user,&command,controller->total_torque)) {
            qd_controller_fault(controller,QD_FAULT_BOARD); return QD_REJECTED;
        }
        controller->armed=true;
    } else disarm(controller);
    controller->command=command; controller->have_command=true;
    controller->acknowledged_sequence=frame->sequence; controller->last_command_ms=now_ms;
    return QD_ACCEPTED;
}
bool qd_controller_feedback(qd_controller *controller,uint64_t stamp_ns,qd_frame *out) {
    if (out==NULL || controller->session==0) return false;
    if (!read_actual(controller)) {
        if (controller->armed) qd_controller_fault(controller,QD_FAULT_STATE);
        return false;
    }
    qd_feedback feedback={0}; feedback.acknowledged_sequence=controller->acknowledged_sequence;
    feedback.fault_flags=controller->fault_flags; feedback.armed=controller->armed?1u:0u;
    feedback.measured=controller->measured;
    memset(out,0,sizeof(*out)); out->type=QD_FEEDBACK; out->session=controller->session;
    out->sequence=controller->tx_sequence++; out->stamp_ns=stamp_ns; out->payload_length=QD_FEEDBACK_SIZE;
    return qd_pack_feedback(&feedback,out->payload);
}
