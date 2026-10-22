#define _POSIX_C_SOURCE 200809L
#include "qd_protocol.h"
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Native transport/MCU emulator only. This never accesses UART or CAN. */
typedef struct {
    qd_controller controller;
    bool virtual_enable;
    bool frame_driven;
    const char *measurement_path;
    uint8_t output[QD_MAX_FRAME*16u];
    size_t output_used;
    bool io_failed;
    unsigned applies;
} emulator;
static uint64_t now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t)!=0) { perror("clock_gettime"); exit(1); }
    return (uint64_t)t.tv_sec*UINT64_C(1000000000)+(uint64_t)t.tv_nsec;
}
static bool virtual_ready(void *user) { return ((emulator *)user)->virtual_enable; }
static bool virtual_read(void *user,qd_measurement *measured) {
    emulator *e=user;
    if (e->measurement_path==NULL) { memset(measured,0,sizeof(*measured)); return true; }
    FILE *file=fopen(e->measurement_path,"rb");
    if (file==NULL) return false;
    uint8_t payload[QD_FEEDBACK_SIZE];
    size_t n=fread(payload,1,sizeof(payload),file);
    bool exact=(n==sizeof(payload) && fgetc(file)==EOF && !ferror(file));
    fclose(file);
    qd_feedback feedback;
    if (!exact || !qd_unpack_feedback(payload,n,&feedback)) return false;
    *measured=feedback.measured; return true;
}
static bool virtual_apply(void *user,const qd_command *command,const float *total) {
    emulator *e=user; (void)command; (void)total; ++e->applies;
    return e->virtual_enable;
}
static void virtual_disarm(void *user) { (void)user; }
static void queue_frame(emulator *e,const qd_frame *frame) {
    uint8_t bytes[QD_MAX_FRAME]; size_t n=qd_encode_frame(frame,bytes,sizeof(bytes));
    if (n==0 || n>sizeof(e->output)-e->output_used) {
        e->io_failed=true; qd_controller_fault(&e->controller,QD_FAULT_IO); return;
    }
    memcpy(e->output+e->output_used,bytes,n); e->output_used+=n;
}
static void handle_frame(void *user,const qd_frame *frame) {
    emulator *e=user; uint64_t t=now_ns(); qd_frame reply;
    enum qd_result result=qd_controller_on_frame(&e->controller,frame,(uint32_t)(t/1000000u),&reply);
    if (result==QD_REPLY) queue_frame(e,&reply);
    else if (e->frame_driven && (frame->type==QD_COMMAND || frame->type==QD_STOP)) {
        if (qd_controller_feedback(&e->controller,t,&reply)) queue_frame(e,&reply);
    }
}
static bool drain(emulator *e) {
    if (e->output_used==0) return true;
    ssize_t n=write(STDOUT_FILENO,e->output,e->output_used);
    if (n<0) return errno==EAGAIN || errno==EWOULDBLOCK || errno==EINTR;
    e->output_used-=(size_t)n;
    memmove(e->output,e->output+n,e->output_used); return true;
}
int main(int argc,char **argv) {
    emulator e={0};
    for (int i=1; i<argc; ++i) {
        if (strcmp(argv[i],"--virtual-enable")==0) e.virtual_enable=true;
        else if (strcmp(argv[i],"--frame-driven")==0) e.frame_driven=true;
        else if (strcmp(argv[i],"--measurement")==0 && i+1<argc) e.measurement_path=argv[++i];
        else { fprintf(stderr,"usage: %s [--frame-driven] [--virtual-enable] [--measurement feedback-payload.bin]\n",argv[0]); return 2; }
    }
    fprintf(stderr,"virtual MCU only; virtual enable=%u; measurements=%s\n",e.virtual_enable?1u:0u,
            e.measurement_path==NULL?"explicit fixed zero state":e.measurement_path);
    signal(SIGPIPE,SIG_IGN);
    int flags=fcntl(STDOUT_FILENO,F_GETFL);
    if (flags<0 || fcntl(STDOUT_FILENO,F_SETFL,flags|O_NONBLOCK)<0) { perror("stdout nonblocking"); return 1; }
    qd_board board={&e,virtual_ready,virtual_read,virtual_apply,virtual_disarm};
    qd_controller_init(&e.controller,&board); qd_parser parser; qd_parser_init(&parser);
    const uint64_t period=UINT64_C(5000000);
    bool eof=false; uint64_t logical_tick=0,next_tick=now_ns()+period;
    while (!e.io_failed && (!eof || e.output_used!=0)) {
        struct pollfd fds[2]={{eof?-1:STDIN_FILENO,POLLIN,0},
                             {STDOUT_FILENO,e.output_used?POLLOUT:0,0}};
        uint64_t t=now_ns();
        int timeout=t>=next_tick?0:(int)((next_tick-t+999999u)/1000000u);
        int result=poll(fds,2,timeout);
        if (result<0 && errno!=EINTR) { perror("poll"); e.io_failed=true; break; }
        if (!eof && (fds[0].revents&(POLLIN|POLLHUP))) {
            uint8_t bytes[1024]; ssize_t n=read(STDIN_FILENO,bytes,sizeof(bytes));
            if (n>0) qd_parser_feed(&parser,bytes,(size_t)n,handle_frame,&e);
            else if (n==0) eof=true;
            else if (errno!=EINTR && errno!=EAGAIN) { perror("stdin"); e.io_failed=true; }
        }
        t=now_ns();
        if (t>=next_tick) {
            uint64_t ticks=1u+(t-next_tick)/period;
            logical_tick+=ticks*period; next_tick+=ticks*period;
            qd_controller_tick(&e.controller,(uint32_t)(t/1000000u));
            if (!eof && !e.frame_driven) {
                qd_frame feedback;
                /* Scheduling delay produces a visible gap, not repeated
                 * measurements stamped as earlier logical ticks. */
                if (qd_controller_feedback(&e.controller,logical_tick,&feedback)) queue_frame(&e,&feedback);
            }
        }
        if (!drain(&e)) { perror("stdout"); e.io_failed=true; }
    }
    e.controller.armed=false; virtual_disarm(&e);
    fprintf(stderr,"frames=%u rejected_frames=%u ack=%u fault=%u armed=0 virtual_applies=%u\n",
            parser.emitted_frames,parser.rejected_frames,e.controller.acknowledged_sequence,
            e.controller.fault_flags,e.applies);
    return e.io_failed?1:0;
}
